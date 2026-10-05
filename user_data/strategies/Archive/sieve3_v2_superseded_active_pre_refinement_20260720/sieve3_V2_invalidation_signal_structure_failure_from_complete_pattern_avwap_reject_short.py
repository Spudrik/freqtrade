"""Signal-candle and entry-frozen swing invalidations for the AVWAP rejection short."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import math
import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter, IStrategy, stoploss_from_absolute
from user_data.Indicators.complex_volume_profile import add_volume_profile


ENTRY_TAG = "avwap_reject_short"
SIDE = "short"
SOURCE_STRATEGY = (
    "user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve/runs/"
    "sieve2_avwap_reject_short__auto_generic_1h_2020_q2_q3/strategy/"
    "sieve2_avwap_reject_short.py:Sieve2AvwapRejectShort"
)
SOURCE_PARAMS_FILE = (
    "user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve/params/"
    "Sieve2AvwapRejectShort__auto_generic_1h_2020_q2_q3.json"
)
LOCKED_BUY_PARAMS = {
    "reclaim_buffer_pct": 0.006,
    "zone_near_pct": 0.005,
    "rolling_level_lookback": 72,
    "avwap_anchor_lookback": 240,
    "use_volume_guard": True,
    "volume_window": 48,
    "volume_ratio_min": 1.0,
    "pressure_min": 0.2,
    "vp_window": 96,
    "vp_bins": 48,
    "vp_guard_mode": "context",
    "vp_score_min": 0.15,
    "vp_context_min": 0.45,
}


def _num(frame: DataFrame, column: str) -> Series:
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)



LINEAGE_STATUS = "verified_archived_executable_plus_promoted_effective_buy_params"
EXIT_FAMILY = "invalidation_signal_structure_failure"
EXIT_HYPOTHESIS = "Exit when price closes through the rejection candle high or its pre-entry swing resistance."
PRIMARY_TRIGGER = "locked AVWAP-from-high rejection short"
PRIMARY_GUARD = "locked bearish volume pressure plus 1h bearish VP context"
TARGET_PROVIDER = "none"
INVALIDATION_PROVIDER = "entry signal high or entry-frozen 72-candle rolling resistance"
ACTIVE_SELL_PARAMS = ("failure_plan",)


SIEVE_STAGE = "sieve3"
ENTRY_SOURCE_STAGE = "sieve2"
RESEARCH_PATH = "sieve3_exit_invalidation_signal_structure_failure"
SOURCE_RESULT_BATCH = "historical_promotion_lineage_unknown"


class Sieve3V2InvalidationSignalStructureFailureFromCompletePatternAvwapRejectShort(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = "1h"
    startup_candle_count = 336
    process_only_new_candles = True
    can_short = True
    minimal_roi = {"0": 100.0}
    stoploss = -0.99
    use_exit_signal = True
    exit_profit_only = False
    use_custom_stoploss = True
    position_adjustment_enable = False
    max_entry_position_adjustment = 0
    trailing_stop = False
    ignore_roi_if_entry_signal = False

    reclaim_buffer_pct = 0.006
    zone_near_pct = 0.005
    rolling_level_lookback = 72
    avwap_anchor_lookback = 240
    volume_window = 48
    volume_ratio_min = 1.0
    pressure_min = 0.2
    vp_window = 96
    vp_bins = 48
    vp_value_area_pct = 0.7
    vp_price_source = "hlc3"
    vp_smooth_bins = 3
    vp_hvn_threshold = 0.7
    vp_lvn_threshold = 0.35
    vp_pressure_delta_min = 0.05
    vp_node_near_pct = 0.01
    vp_volume_percentile_min = 0.55
    vp_score_window = 48
    vp_fast_traverse_atr_mult = 1.2
    vp_entry_score_margin = 0.02
    vp_guard_mode = "context"
    vp_score_min = 0.15
    vp_context_min = 0.45
    # Legacy use_close_direction_guard=False and 4h/1d VP gates were fixed off;
    # their parameters and branches are inactive for this locked entry.
    # Legacy generic liquidity, supply/demand, and alternate-concept parameters are
    # inactive because the selected entry is the fixed AVWAP rejection.

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = (pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs)
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe = self._add_volume_pressure(dataframe)
        dataframe = self._add_rolling_levels(dataframe)
        dataframe = self._add_avwap(dataframe)
        return self._add_volume_profile(dataframe)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        condition = self._avwap_reject(dataframe)
        condition &= self._volume_guard(dataframe)
        condition &= self._vp_guard(dataframe, "vp", SIDE, self.vp_guard_mode, self.vp_score_min, self.vp_context_min)
        valid = condition.fillna(False) & dataframe["volume"].gt(0.0) & dataframe["close"].notna()
        dataframe.loc[valid, "enter_short"] = 1
        dataframe.loc[valid, "enter_tag"] = ENTRY_TAG
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None
        return dataframe

    def _add_volume_pressure(self, dataframe: DataFrame) -> DataFrame:
        window = int(self.volume_window)
        volume_mean = _num(dataframe, "volume").rolling(window, min_periods=max(2, window // 3)).mean()
        candle_range = _num(dataframe, "high").sub(_num(dataframe, "low")).replace(0.0, np.nan)
        close_location = _num(dataframe, "close").sub(_num(dataframe, "low")).div(candle_range).clip(0.0, 1.0)
        dataframe["entry_volume_ratio"] = _num(dataframe, "volume").div(volume_mean.replace(0.0, np.nan))
        dataframe["entry_pressure"] = close_location.sub(0.5).mul(2.0)
        return dataframe

    def _add_rolling_levels(self, dataframe: DataFrame) -> DataFrame:
        lookback = int(self.rolling_level_lookback)
        min_periods = max(4, lookback // 4)
        dataframe["rolling_resistance"] = _num(dataframe, "high").shift(1).rolling(lookback, min_periods=min_periods).max()
        return dataframe

    def _add_avwap(self, dataframe: DataFrame) -> DataFrame:
        lookback = int(self.avwap_anchor_lookback)
        high = _num(dataframe, "high")
        high_reset = high.shift(1).ge(high.shift(1).rolling(lookback, min_periods=2).max())
        dataframe["avwap_from_high"] = self._anchored_vwap_series(dataframe, high_reset)
        return dataframe

    @staticmethod
    def _anchored_vwap_series(dataframe: DataFrame, reset: Series) -> Series:
        typical = _num(dataframe, "high").add(_num(dataframe, "low")).add(_num(dataframe, "close")).div(3.0)
        volume = _num(dataframe, "volume").clip(lower=0.0)
        groups = reset.fillna(False).astype(bool).cumsum()
        return typical.mul(volume).groupby(groups).cumsum().div(volume.groupby(groups).cumsum().replace(0.0, np.nan))

    def _add_volume_profile(self, dataframe: DataFrame) -> DataFrame:
        return add_volume_profile(
            dataframe,
            window=self.vp_window,
            bins=self.vp_bins,
            value_area_pct=self.vp_value_area_pct,
            price_source=self.vp_price_source,
            smooth_bins=self.vp_smooth_bins,
            hvn_threshold=self.vp_hvn_threshold,
            lvn_threshold=self.vp_lvn_threshold,
            pressure_delta_min=self.vp_pressure_delta_min,
            node_near_pct=self.vp_node_near_pct,
            volume_percentile_min=self.vp_volume_percentile_min,
            score_window=self.vp_score_window,
            fast_traverse_atr_mult=self.vp_fast_traverse_atr_mult,
            entry_score_margin=self.vp_entry_score_margin,
            prefix="vp",
        )

    def _avwap_reject(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, "close")
        avwap = _num(dataframe, "avwap_from_high")
        trigger = avwap.mul(1.0 - self.reclaim_buffer_pct)
        return close.le(trigger) & close.shift(1).gt(avwap.shift(1)) & _num(dataframe, "high").ge(avwap.mul(1.0 - self.zone_near_pct))

    def _volume_guard(self, dataframe: DataFrame) -> Series:
        return _num(dataframe, "entry_volume_ratio").ge(self.volume_ratio_min) & _num(dataframe, "entry_pressure").le(-self.pressure_min)

    @staticmethod
    def _score_guard(dataframe: DataFrame, prefix: str, side: str, score_min: float) -> Series:
        score = _num(dataframe, f"{prefix}_score_{side}")
        opposite = _num(dataframe, f"{prefix}_score_{'short' if side == 'long' else 'long'}")
        return score.ge(score_min) & score.ge(opposite)

    @staticmethod
    def _context_guard(dataframe: DataFrame, prefix: str, side: str, context_min: float) -> Series:
        context_name = "context_score_bull" if side == "long" else "context_score_bear"
        opposite_name = "context_score_bear" if side == "long" else "context_score_bull"
        context = _num(dataframe, f"{prefix}_{context_name}")
        opposite = _num(dataframe, f"{prefix}_{opposite_name}")
        market = _num(dataframe, f"{prefix}_market_context")
        market_ok = market.ge(0) if side == "long" else market.le(0)
        return context.ge(context_min) & context.ge(opposite) & market_ok

    def _vp_guard(self, dataframe: DataFrame, prefix: str, side: str, mode: str, score_min: float, context_min: float) -> Series:
        score_ok = self._score_guard(dataframe, prefix, side, score_min)
        context_ok = self._context_guard(dataframe, prefix, side, context_min)
        if mode == "score":
            return score_ok
        if mode == "context":
            return context_ok
        if mode == "score_or_context":
            return score_ok | context_ok
        if mode == "balance":
            return _num(dataframe, f"{prefix}_context_score_balance").ge(context_min)
        direction = _num(dataframe, f"{prefix}_market_context")
        return direction.ge(0) if side == "long" else direction.le(0)

    def _closed_frame(self, pair: str, current_time: datetime) -> DataFrame:
        if not getattr(self, "dp", None):
            raise RuntimeError("exit callback requires Freqtrade's analyzed dataframe")
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            raise RuntimeError("exit callback received no analyzed candles")
        dates = pd.to_datetime(dataframe["date"], utc=True, errors="raise")
        close_times = dates + pd.to_timedelta(
            timeframe_to_minutes(self.timeframe),
            unit="m",
        )
        now = pd.Timestamp(current_time)
        now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
        return dataframe.loc[close_times.le(now)].sort_values("date")

    def _post_entry_rows(self, dataframe: DataFrame, trade: Any) -> DataFrame:
        filled_at = pd.Timestamp(
            getattr(trade, "date_entry_fill_utc", None) or trade.open_date_utc
        )
        filled_at = (
            filled_at.tz_localize("UTC")
            if filled_at.tzinfo is None
            else filled_at.tz_convert("UTC")
        )
        candle_opens = pd.to_datetime(dataframe["date"], utc=True, errors="raise")
        return dataframe.loc[candle_opens.ge(filled_at)]

    @staticmethod
    def _entry_row(dataframe: DataFrame, trade: Any) -> Series:
        opened = pd.Timestamp(trade.open_date_utc)
        if opened.tzinfo is None:
            opened = opened.tz_localize("UTC")
        dates = pd.to_datetime(dataframe["date"], utc=True, errors="raise")
        rows = dataframe.loc[dates.lt(opened)]
        if rows.empty:
            raise RuntimeError("no closed signal candle exists before trade open")
        return rows.iloc[-1]

    FAILURE_PLANS = {
        "signal_high_one_close": ("high", 1),
        "signal_high_two_closes": ("high", 2),
        "swing_resistance_one_close": ("rolling_resistance", 1),
        "swing_resistance_two_closes": ("rolling_resistance", 2),
    }
    ACTIVE_SELL_PARAMS = ('failure_plan',)

    failure_plan = CategoricalParameter(tuple(FAILURE_PLANS), default="signal_high_one_close", space="sell", optimize=True, load=True)
    failure_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | None:
        _ = (current_rate, current_profit, kwargs)
        level_column, confirmations = self.FAILURE_PLANS[str(self.failure_plan.value)]
        dataframe = self._closed_frame(pair, current_time)
        entry_row = self._entry_row(dataframe, trade)
        level = float(entry_row[level_column])
        if not math.isfinite(level) or level <= 0.0:
            return None
        post_entry = self._post_entry_rows(dataframe, trade)
        if len(post_entry) < confirmations:
            return None
        closes = pd.to_numeric(post_entry.iloc[-confirmations:]["close"], errors="coerce")
        failed = closes.gt(level)
        return f"signal_structure_failure_{self.failure_plan.value}" if bool(failed.all()) else None

    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs: Any) -> float:
        _ = (pair, current_time, current_profit, after_fill, kwargs)
        stop_price = float(trade.open_rate) * 1.03
        return stoploss_from_absolute(stop_price, current_rate=current_rate, is_short=True, leverage=float(trade.leverage or 1.0))
