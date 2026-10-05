"""Sieve3 V2 integrated exits for the promoted MTF MACD-volume long entry.

Preflight contract
------------------
Canonical Sieve2 source:
    sieve2_mtf_std_daily_macd_volume_breakout_long_1h
Promotion evidence:
    20260612T071622_entry_sieve2_top_level_clear100_cores8_20260612_01_retry01,
    result row 33 and its backtest-result-2026-06-12_08-38-35.zip archive.
Oracle hashes:
    Executed source SHA-256:
    c8e78309683f277548e9ac9e3a929ae475a6daa4c197d6060eed4eaf7455b18f
    Result ZIP SHA-256:
    ea1713135c6646f6db870cc7fa3e3e673310ec70061eb31f627e49d1d1960f09
    Locked params-member SHA-256:
    932d20bed431b8f58d4b8bc0309fa9f289fdc9ea66e3a1fdf2df503932b15e02
    Effective 14-value buy-lock SHA-256 (sorted compact JSON):
    e97f93bfde27c64de9ffa158a06ce4f9017363dcddcd9fabea71071cdbcf0a58
Promotion-oracle correction:
    The baseline lookup and JSONL labels say TP/SL 3%/3%. The authoritative
    executed source embedded in the result ZIP is byte-identical to the run
    snapshot and resolves ROI/stoploss to 3%/4%. ``promotion_oracle_3_4``
    preserves that historical fact; ``baseline_3_3`` remains the required V2
    comparison control. Entry logic and promoted buy values are unchanged.
Entry lock:
    The 14 promoted buy values below are immutable in Sieve3 V2
    (``optimize=False, load=False``). The lock includes parameters that are
    inactive for this fixed entry mode because they are part of the oracle;
    their dead RSI/Bollinger/ATR calculation paths are not reproduced.
Tested legacy controls:
    sieve3_exit_breakeven_from_mtf_std_daily_macd_volume_breakout_long_1h.py,
    job 20260708T190447_entry_sieve3_exit_speed_batch_045;
    sieve3_exit_level_zone_reversal_from_mtf_std_daily_macd_volume_breakout_long_1h.py,
    job 20260624T185245_entry_sieve3_exit_speed_batch_024.
Refined pilot controls:
    sieve3_rework_exit_{invalidation,layered,target_zone}_from_mtf_std_daily_*.py,
    job 20260711T230600_entry_sieve3_exit_rework_comparison_001.
Consolidation:
    Thirty-two plans retain fixed controls, source-native MTF and causal OHLCV
    targets, multi-source invalidation, layered realization, runners, and
    bounded progress failure. Twenty are integrated/layered policies and twelve
    isolate baseline or full-exit controls. The authoritative source contains no
    Volume Profile subsystem, so V2 does not import, compute, or test one.
Target semantics:
    Daily/4h prior highs, causal 1h prior-48/prior-96 highs, and measured
    breakout objectives are frozen from the last closed placement candle when
    the entry fill is confirmed. Named providers remain exact; ``nearest``
    selects the closest valid frozen objective across those families.
Invalidation:
    Full thesis exits require coherent combinations of source-breakout loss,
    LTF momentum/participation failure, 4h context failure, and daily context
    failure. No one-bar MACD, EMA, or volume flip can be a sole full exit.
Hyperopt surface:
    One sell-space ``exit_policy_plan`` parameter. Each category is a complete,
    trader-readable policy; no inactive branch thresholds are optimized.
Split decision:
    One V2 file is sufficient because the daily MACD-volume breakout, its
    1h/4h/1d context, and causal structural objectives form one policy ecosystem.
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
    merge_informative_pair,
    stoploss_from_absolute,
)


SIEVE_STAGE = "sieve3"
SOURCE_STRATEGY = "sieve2_mtf_std_daily_macd_volume_breakout_long_1h"
SOURCE_RESULT_BATCH = "20260612T071622_entry_sieve2_top_level_clear100_cores8_20260612_01_retry01"
RESEARCH_PATH = "sieve3_exit_integrated"
ENTRY_SOURCE_STAGE = "sieve2"
EXIT_HYPOTHESIS = (
    "Frozen MTF prior highs, causal structural highs, and measured-range targets "
    "interact with multi-source invalidation through complete layered long-management "
    "policies."
)
ENTRY_MODE = "daily_macd_volume_breakout"
ENTRY_TAG = "mtf_std_daily_macd_volume_breakout_long_1h"
STATE_KEY = "sieve3_v2_mtf_macd_volume_long"
STATE_VERSION = 3
LEVEL_BAND = 0.005
MIN_TARGET_MOVE = 0.002
PARTIAL_TAGS = {"s3v2_partial_target", "s3v2_partial_invalidation"}


def _safe_div(numerator: Series, denominator: Series) -> Series:
    return numerator / denominator.replace(0.0, np.nan)


def _ema(close: Series, length: int) -> Series:
    return close.ewm(span=length, adjust=False, min_periods=max(2, length // 2)).mean()


def _macd_hist(close: Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Series:
    macd = _ema(close, fast) - _ema(close, slow)
    return macd - _ema(macd, signal)


def _num(frame: DataFrame, column: str) -> Series:
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)



def _finite_float(value: Any) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric) or not np.isfinite(numeric):
        return None
    return float(numeric)


DEFAULT_FALLBACK_ROI = 0.12
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
        invalidation: str = "any_two",
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
    "promotion_oracle_3_4": ExitPlan(
        "baseline",
        invalidation="none",
        invalidation_action="hold",
        hard_stop=0.04,
        fixed_tp=0.03,
    ),
    "d1_high_touch_full": ExitPlan(
        "target_full",
        "d1_high",
        confirmation="touch",
        target_action="full",
        invalidation="none",
        invalidation_action="hold",
        target_band=0.005,
    ),
    "d1_high_reversal_full": ExitPlan(
        "target_full",
        "d1_high",
        confirmation="reversal1",
        target_action="full",
        invalidation="none",
        invalidation_action="hold",
    ),
    "h4_high_reversal_full": ExitPlan(
        "target_full",
        "h4_high",
        confirmation="reversal1",
        target_action="full",
        invalidation="none",
        invalidation_action="hold",
    ),
    "prior48_touch_full": ExitPlan(
        "target_full",
        "prior48",
        confirmation="touch",
        target_action="full",
        invalidation="none",
        invalidation_action="hold",
        target_band=0.005,
    ),
    "prior96_reversal_full": ExitPlan(
        "target_full",
        "prior96",
        confirmation="reversal1",
        target_action="full",
        invalidation="none",
        invalidation_action="hold",
    ),
    "range_full_reversal_full": ExitPlan(
        "target_full",
        "range_full",
        confirmation="reversal1",
        target_action="full",
        invalidation="none",
        invalidation_action="hold",
    ),
    "source_ltf_full": ExitPlan(
        "invalidation_full", invalidation="source_ltf", invalidation_action="full"
    ),
    "ltf_h4_full": ExitPlan("invalidation_full", invalidation="ltf_h4", invalidation_action="full"),
    "h4_d1_full": ExitPlan("invalidation_full", invalidation="h4_d1", invalidation_action="full"),
    "any_two_full": ExitPlan(
        "invalidation_full", invalidation="any_two", invalidation_action="full"
    ),
    "d1_touch_p33_be_any_two": ExitPlan(
        "target_partial",
        "d1_high",
        confirmation="touch",
        target_action="partial",
        invalidation="any_two",
        partial_fraction=0.33,
        remainder="breakeven",
    ),
    "d1_reversal_p50_trail_any_two": ExitPlan(
        "target_partial",
        "d1_high",
        confirmation="reversal1",
        target_action="partial",
        invalidation="any_two",
        partial_fraction=0.50,
        remainder="trail",
    ),
    "h4_reversal_p33_be_ltf_h4": ExitPlan(
        "target_partial",
        "h4_high",
        confirmation="reversal1",
        target_action="partial",
        invalidation="ltf_h4",
        partial_fraction=0.33,
        remainder="breakeven",
    ),
    "prior48_touch_p33_be_any_two": ExitPlan(
        "target_partial",
        "prior48",
        confirmation="touch",
        target_action="partial",
        invalidation="any_two",
        partial_fraction=0.33,
        remainder="breakeven",
    ),
    "prior48_reversal_p50_trail_any_two": ExitPlan(
        "target_partial",
        "prior48",
        confirmation="reversal1",
        target_action="partial",
        invalidation="any_two",
        partial_fraction=0.50,
        remainder="trail",
    ),
    "prior96_reversal_p33_trail_h4_d1": ExitPlan(
        "target_partial",
        "prior96",
        confirmation="reversal1",
        target_action="partial",
        invalidation="h4_d1",
        partial_fraction=0.33,
        remainder="trail",
    ),
    "range_half_touch_p33_be_source_ltf": ExitPlan(
        "target_partial",
        "range_half",
        confirmation="touch",
        target_action="partial",
        invalidation="source_ltf",
        partial_fraction=0.33,
        remainder="breakeven",
    ),
    "range_half_reversal_p50_trail_any_two": ExitPlan(
        "target_partial",
        "range_half",
        confirmation="reversal1",
        target_action="partial",
        invalidation="any_two",
        partial_fraction=0.50,
        remainder="trail",
    ),
    "range_full_reversal_p33_trail_any_two": ExitPlan(
        "target_partial",
        "range_full",
        confirmation="reversal1",
        target_action="partial",
        invalidation="any_two",
        partial_fraction=0.33,
        remainder="trail",
    ),
    "nearest_reversal_p50_trail_any_two": ExitPlan(
        "target_partial",
        "nearest",
        confirmation="reversal1",
        target_action="partial",
        invalidation="any_two",
        partial_fraction=0.50,
        remainder="trail",
    ),
    "any_two_reduce33_d1": ExitPlan(
        "invalidation_reduce",
        "d1_high",
        confirmation="touch",
        target_action="full",
        invalidation="any_two",
        invalidation_action="partial",
        partial_fraction=0.33,
        remainder="target",
    ),
    "source_ltf_reduce33_prior48": ExitPlan(
        "invalidation_reduce",
        "prior48",
        confirmation="touch",
        target_action="full",
        invalidation="source_ltf",
        invalidation_action="partial",
        partial_fraction=0.33,
        remainder="target",
    ),
    "h4_d1_reduce50_nearest": ExitPlan(
        "invalidation_reduce",
        "nearest",
        confirmation="touch",
        target_action="full",
        invalidation="h4_d1",
        invalidation_action="partial",
        partial_fraction=0.50,
        remainder="target",
    ),
    "d1_touch_tighten_any_two": ExitPlan(
        "target_tighten",
        "d1_high",
        confirmation="touch",
        target_action="tighten",
        invalidation="any_two",
        invalidation_action="full",
        remainder="tighten",
    ),
    "prior48_touch_tighten_source_ltf": ExitPlan(
        "target_tighten",
        "prior48",
        confirmation="touch",
        target_action="tighten",
        invalidation="source_ltf",
        invalidation_action="full",
        remainder="tighten",
    ),
    "range_half_touch_tighten_h4_d1": ExitPlan(
        "target_tighten",
        "range_half",
        confirmation="touch",
        target_action="tighten",
        invalidation="h4_d1",
        invalidation_action="full",
        remainder="tighten",
    ),
    "h4_then_d1_runner_any_two": ExitPlan(
        "dual_target",
        "h4_high",
        "d1_high",
        "touch",
        "partial",
        "any_two",
        "full",
        0.33,
        "trail",
    ),
    "prior48_then_d1_runner_any_two": ExitPlan(
        "dual_target",
        "prior48",
        "d1_high",
        "touch",
        "partial",
        "any_two",
        "full",
        0.50,
        "trail",
    ),
    "progress48_nearest_any_two": ExitPlan(
        "progress_failure",
        "nearest",
        confirmation="touch",
        target_action="full",
        invalidation="any_two",
        invalidation_action="full",
        progress_candles=48,
    ),
    "progress96_d1_any_two": ExitPlan(
        "progress_failure",
        "d1_high",
        confirmation="touch",
        target_action="full",
        invalidation="any_two",
        invalidation_action="full",
        progress_candles=96,
    ),
}


TARGET_PROVIDER_CHAINS: dict[str, tuple[str, ...]] = {
    "d1_high": ("d1_high",),
    "h4_high": ("h4_high",),
    "prior48": ("prior48",),
    "prior96": ("prior96",),
    "range_half": ("range_half",),
    "range_full": ("range_full",),
    "nearest": (
        "h4_high",
        "d1_high",
        "prior48",
        "prior96",
        "range_half",
        "range_full",
    ),
}


ACTIVE_SELL_PARAMS = ("exit_policy_plan",)


class Sieve3V2IntegratedFromMtfStdDailyMacdVolumeBreakoutLong1H(IStrategy):
    SIEVE_STAGE = SIEVE_STAGE
    SOURCE_STRATEGY = SOURCE_STRATEGY
    SOURCE_RESULT_BATCH = SOURCE_RESULT_BATCH
    RESEARCH_PATH = RESEARCH_PATH
    ENTRY_SOURCE_STAGE = ENTRY_SOURCE_STAGE
    EXIT_HYPOTHESIS = EXIT_HYPOTHESIS

    INTERFACE_VERSION = 3
    timeframe = "1h"
    startup_candle_count = 400
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

    ema_fast_len = 43
    ema_slow_len = 139
    # Legacy inert entry parameter: bb_len=52 (fixed entry mode/gate).
    # Legacy inert entry parameter: bb_width_max=0.13 (fixed entry mode/gate).
    volume_ratio_min = 1.67
    # Legacy inert entry parameter: retest_buffer_pct=0.026 (fixed entry mode/gate).
    # Legacy inert entry parameter: rsi_len=23 (fixed entry mode/gate).
    # Legacy inert entry parameter: rsi_long_min=42 (fixed entry mode/gate).
    # Legacy inert entry parameter: rsi_short_max=41 (fixed entry mode/gate).
    # Legacy fixed-off entry parameter: use_daily_trend=False (CategoricalParameter, buy space).
    # Legacy fixed-off entry parameter: use_h4_compression=False (CategoricalParameter, buy space).
    # Legacy inert entry parameter: use_volume_filter=True (fixed entry mode/gate).
    # Legacy inert entry parameter: use_retest=True (fixed entry mode/gate).
    # Legacy inert entry parameter: use_momentum_filter=True (fixed entry mode/gate).

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
        dp = getattr(self, "dp", None)
        if dp is None:
            return []
        pairs = dp.current_whitelist()
        return [(pair, "4h") for pair in pairs] + [(pair, "1d") for pair in pairs]

    def _features(self, frame: DataFrame, timeframe: str) -> DataFrame:
        close = _num(frame, "close")
        high = _num(frame, "high")
        low = _num(frame, "low")
        volume = _num(frame, "volume").fillna(0.0)
        ema_fast = int(self.ema_fast_len)
        ema_slow = int(self.ema_slow_len)
        frame[f"ema_fast_{timeframe}"] = _ema(close, ema_fast)
        frame[f"ema_slow_{timeframe}"] = _ema(close, ema_slow)
        frame[f"macd_hist_{timeframe}"] = _macd_hist(close).fillna(0.0)
        frame[f"volume_ratio_{timeframe}"] = _safe_div(
            volume, volume.rolling(20, min_periods=5).mean()
        ).fillna(0.0)
        frame[f"prior_high_{timeframe}"] = high.shift(1).rolling(20, min_periods=5).max()
        frame[f"prior_low_{timeframe}"] = low.shift(1).rolling(20, min_periods=5).min()
        return frame

    def populate_indicators(self, dataframe: DataFrame, metadata: dict[str, object]) -> DataFrame:
        frame = dataframe.copy()
        dp = getattr(self, "dp", None)
        if dp is not None and metadata and metadata.get("pair"):
            pair = str(metadata["pair"])
            for timeframe in ("4h", "1d"):
                informative = dp.get_pair_dataframe(pair=pair, timeframe=timeframe)
                if informative is not None and not informative.empty:
                    informative = self._features(informative.copy(), timeframe)
                    frame = merge_informative_pair(
                        frame, informative, self.timeframe, timeframe, ffill=True
                    )
                    frame = frame.rename(
                        columns=lambda column, tf=timeframe: column.replace(f"_{tf}_{tf}", f"_{tf}")
                    )
        frame = self._features(frame, "1h")
        high = _num(frame, "high")
        low = _num(frame, "low")
        frame["s3v2_prior_high_48"] = high.shift(1).rolling(48, min_periods=12).max()
        frame["s3v2_prior_high_96"] = high.shift(1).rolling(96, min_periods=24).max()
        frame["s3v2_prior_low_48"] = low.shift(1).rolling(48, min_periods=12).min()
        return frame

    def _daily_trend_ok(self, frame: DataFrame) -> Series:
        return pd.Series(True, index=frame.index, dtype='bool')

    def _h4_compression_ok(self, frame: DataFrame) -> Series:
        return pd.Series(True, index=frame.index, dtype='bool')

    def _volume_ok(self, frame: DataFrame) -> Series:
        return _num(frame, 'volume_ratio_1h').ge(float(self.volume_ratio_min))

    def _entry_condition(self, frame: DataFrame) -> Series:
        close = _num(frame, "close")
        breakout = close.gt(_num(frame, "high").shift(1).rolling(20, min_periods=5).max())
        return (
            self._daily_trend_ok(frame)
            & self._h4_compression_ok(frame)
            & self._volume_ok(frame)
            & _num(frame, "macd_hist_1d").gt(0.0)
            & breakout
        ).fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict[str, object]) -> DataFrame:
        _ = metadata
        dataframe.loc[self._entry_condition(dataframe), ["enter_long", "enter_tag"]] = (
            1,
            ENTRY_TAG,
        )
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict[str, object]) -> DataFrame:
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
            raise RuntimeError("integrated MTF target snapshot requires a closed entry candle")
        required = {
            "prior_high_1h",
            "prior_high_1d",
            "prior_high_4h",
            "s3v2_prior_high_48",
            "s3v2_prior_high_96",
            "s3v2_prior_low_48",
        }
        missing = sorted(required - set(row.index))
        if missing:
            raise KeyError(f"integrated MTF target snapshot missing required source columns: {missing}")
        broken = _finite_float(row["prior_high_1h"])
        support = _finite_float(row["s3v2_prior_low_48"])
        if support is not None and support >= entry_rate:
            support = None

        raw: dict[str, float | None] = {
            "d1_high": self._directional_target(row["prior_high_1d"], entry_rate),
            "h4_high": self._directional_target(row["prior_high_4h"], entry_rate),
            "prior48": self._directional_target(row["s3v2_prior_high_48"], entry_rate),
            "prior96": self._directional_target(row["s3v2_prior_high_96"], entry_rate),
            "range_half": None,
            "range_full": None,
        }
        if broken is not None and support is not None and broken > support:
            measured_range = broken - support
            raw["range_half"] = self._directional_target(broken + measured_range * 0.5, entry_rate)
            raw["range_full"] = self._directional_target(broken + measured_range, entry_rate)

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
        source_ltf: bool,
        ltf_h4: bool,
        h4_d1: bool,
        any_two: bool,
    ) -> bool:
        return {
            "none": False,
            "source_ltf": source_ltf,
            "ltf_h4": ltf_h4,
            "h4_d1": h4_d1,
            "any_two": any_two,
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

        decision_frame = target_frame
        close = _num(decision_frame, "close")
        broken = _finite_float(state.get("broken_resistance"))
        if broken is None or decision_frame.empty:
            source_condition = pd.Series(False, index=decision_frame.index, dtype="bool")
        else:
            source_condition = close.lt(broken * (1.0 - LEVEL_BAND))

        ltf_condition = (
            close.lt(_num(decision_frame, "ema_fast_1h"))
            & _num(decision_frame, "macd_hist_1h").le(0.0)
            & _num(decision_frame, "volume_ratio_1h").lt(1.0)
        )
        h4_condition = _num(decision_frame, "ema_fast_4h").le(
            _num(decision_frame, "ema_slow_4h")
        ) & _num(decision_frame, "macd_hist_4h").le(0.0)
        d1_condition = _num(decision_frame, "macd_hist_1d").le(0.0) & _num(
            decision_frame, "ema_fast_1d"
        ).le(_num(decision_frame, "ema_slow_1d"))

        source_failure = self._last_n(source_condition, 2)
        ltf_failure = self._last_n(ltf_condition, 2)
        h4_failure = self._last_n(h4_condition, 1)
        d1_failure = self._last_n(d1_condition, 1)
        source_ltf = source_failure and ltf_failure
        ltf_h4 = ltf_failure and h4_failure
        h4_d1 = h4_failure and d1_failure
        failure_count = sum((source_failure, ltf_failure, h4_failure, d1_failure))
        any_two = failure_count >= 2
        hard_failure = (source_failure and d1_failure) or (
            source_failure and ltf_failure and h4_failure
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
            "hard_invalidation": False if plan.role == "baseline" else hard_failure,
            "invalidation": self._invalidation_event(
                plan.invalidation, source_ltf, ltf_h4, h4_d1, any_two
            ),
            "source_failure": source_failure,
            "ltf_failure": ltf_failure,
            "h4_failure": h4_failure,
            "d1_failure": d1_failure,
            "failure_count": failure_count,
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
    def _priority_full_exit_tag(state: dict[str, Any], events: dict[str, Any]) -> str | None:
        if state.get("terminal_exit_pending"):
            return str(state.get("terminal_exit_tag") or "s3v2_hard_invalidation")
        if events.get("hard_invalidation"):
            return "s3v2_hard_invalidation"
        return None

    @classmethod
    def evaluate_policy(
        cls,
        plan: ExitPlan,
        state: dict[str, Any],
        events: dict[str, Any],
    ) -> ExitDecision:
        priority_tag = cls._priority_full_exit_tag(state, events)
        if priority_tag is not None:
            return ExitDecision("full", priority_tag)
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
        decision = self.evaluate_policy(plan, state, events)
        state["partial_pending"] = partial_pending
        if partial_pending:
            state["phase"] = "REALIZATION_PENDING"
            if decision.action == "full":
                state["terminal_exit_pending"] = True
                state["terminal_exit_tag"] = decision.tag
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
