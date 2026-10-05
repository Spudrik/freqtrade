"""Profit-armed bullish weakening/reversal exits for the AVWAP rejection short."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import BooleanParameter, CategoricalParameter, IStrategy, stoploss_from_absolute
from user_data.Indicators.complex_volume_profile import add_volume_profile


ENTRY_MODE = "entry_avwap_reject_short"


CONCEPT = "avwap_reject"
PERIOD_KIND = "fixed"
PERIOD_CHOICES = ("day", "week", "month")
GUARD_MODE_CHOICES = ("direction", "score", "context", "score_or_context", "balance")
PRICE_SOURCE_CHOICES = ("close", "hl2", "hlc3", "ohlc4")
VP_COLUMNS = (
    "score_long",
    "score_short",
    "context_score_bull",
    "context_score_bear",
    "context_score_balance",
    "market_context",
)


SOURCE_ENTRY_STEM = "complete_pattern_avwap_reject_short"
SOURCE_ENTRY_CLASS = "Sieve2AvwapRejectShort"
SOURCE_STRATEGY = (
    "user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve/runs/"
    "sieve2_avwap_reject_short__auto_generic_1h_2020_q2_q3/strategy/"
    "sieve2_avwap_reject_short.py:Sieve2AvwapRejectShort"
)
SOURCE_PARAMS_FILE = (
    "user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve/params/"
    "Sieve2AvwapRejectShort__auto_generic_1h_2020_q2_q3.json"
)
SOURCE_RESULT_ARCHIVE = (
    "user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve/backtests/"
    "sieve2_avwap_reject_short__auto_generic_1h_2020_q2_q3/full_cycle_2020_2026/"
    "tp_2_sl_2/backtest-result-2026-05-21_18-40-29.zip"
)
LINEAGE_STATUS = "verified_archived_executable_plus_promoted_effective_buy_params"


LOCKED_BUY_PARAMS = {'breakout_buffer_pct': 0.004, 'reclaim_buffer_pct': 0.006, 'sweep_buffer_pct': 0.006, 'zone_near_pct': 0.005, 'rolling_level_lookback': 72, 'equal_level_lookback': 72, 'equal_level_tolerance_pct': 0.006, 'equal_level_min_touches': 2, 'confluence_period': 'day', 'avwap_anchor_lookback': 240, 'avwap_band_mult': 1.25, 'zone_impulse_window': 36, 'zone_impulse_atr_min': 0.8, 'zone_body_fraction_min': 0.55, 'zone_volume_ratio_min': 1.1, 'zone_max_age_bars': 72, 'use_volume_guard': True, 'volume_window': 48, 'volume_ratio_min': 1.0, 'pressure_min': 0.2, 'use_close_direction_guard': False, 'vp_window': 96, 'vp_bins': 48, 'vp_value_area_pct': 0.7, 'vp_price_source': 'hlc3', 'vp_smooth_bins': 3, 'vp_hvn_threshold': 0.7, 'vp_lvn_threshold': 0.35, 'vp_pressure_delta_min': 0.05, 'vp_node_near_pct': 0.01, 'vp_volume_percentile_min': 0.55, 'vp_score_window': 48, 'vp_fast_traverse_atr_mult': 1.2, 'vp_entry_score_margin': 0.02, 'use_vp_1h_guard': True, 'vp_guard_mode': 'context', 'vp_score_min': 0.15, 'vp_context_min': 0.45, 'use_vp_4h_guard': False, 'vp_4h_window': 48, 'vp_4h_bins': 36, 'use_vp_1d_guard': False, 'vp_1d_window': 30, 'vp_1d_bins': 36}




def _num(frame: DataFrame, column: str, default: float | Series=...) -> Series:
    if column not in frame.columns:
        if default is ...:
            raise KeyError(f'missing required column: {column!r}')
        return pd.to_numeric(
            pd.Series(default, index=frame.index), errors='coerce'
        ).replace([np.inf, -np.inf], np.nan)
    return pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)

LINEAGE_STATUS = "verified_archived_executable_plus_promoted_effective_buy_params"








EXIT_FAMILY = 'profit_armed_bullish_reversal'
PRIMARY_TRIGGER = 'locked AVWAP-from-high rejection short'
PRIMARY_GUARD = 'locked bearish volume pressure plus 1h bearish VP context'
PRIMARY_TARGET = 'profit gate followed by two-candle bullish reversal'
PRIMARY_INVALIDATION = 'fixed 3% adverse move'
TARGET_PROVIDER = PRIMARY_TARGET
INVALIDATION_PROVIDER = PRIMARY_INVALIDATION
ENTRY_SOURCE_STAGE = 'sieve2'
SOURCE_RESULT_BATCH = 'historical_promotion_lineage_unknown'
RESEARCH_PATH = 'sieve3_exit_profit_armed_bullish_reversal'
EXIT_HYPOTHESIS = 'Bank profit when two bullish candles and pressure show the bearish rejection is weakening.'
ACTIVE_SELL_PARAMS = ('reversal_plan', 'require_bull_vp_confirmation')
SIEVE_STAGE = 'sieve3'
SIDE = 'short'
ENTRY_TAG = 'avwap_reject_short'

class Sieve3V2ProfitArmedBullishReversalFromCompletePatternAvwapRejectShort(IStrategy):
    """Standalone authoritative promoted entry surface."""

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


    breakout_buffer_pct = 0.004
    reclaim_buffer_pct = 0.006
    sweep_buffer_pct = 0.006
    zone_near_pct = 0.005
    rolling_level_lookback = 72
    equal_level_lookback = 72
    equal_level_tolerance_pct = 0.006
    equal_level_min_touches = 2
    confluence_period = 'day'
    avwap_anchor_lookback = 240
    avwap_band_mult = 1.25
    zone_impulse_window = 36
    zone_impulse_atr_min = 0.8
    zone_body_fraction_min = 0.55
    zone_volume_ratio_min = 1.1
    zone_max_age_bars = 72
    use_volume_guard = True
    volume_window = 48
    volume_ratio_min = 1.0
    pressure_min = 0.2
    # Legacy fixed-off entry parameter: use_close_direction_guard=False (BooleanParameter, buy space).
    vp_window = 96
    vp_bins = 48
    vp_value_area_pct = 0.7
    vp_price_source = 'hlc3'
    vp_smooth_bins = 3
    vp_hvn_threshold = 0.7
    vp_lvn_threshold = 0.35
    vp_pressure_delta_min = 0.05
    vp_node_near_pct = 0.01
    vp_volume_percentile_min = 0.55
    vp_score_window = 48
    vp_fast_traverse_atr_mult = 1.2
    vp_entry_score_margin = 0.02
    use_vp_1h_guard = True
    vp_guard_mode = 'context'
    vp_score_min = 0.15
    vp_context_min = 0.45
    # Legacy fixed-off entry parameter: use_vp_4h_guard=False (BooleanParameter, buy space).
    vp_4h_window = 48
    vp_4h_bins = 36
    # Legacy fixed-off entry parameter: use_vp_1d_guard=False (BooleanParameter, buy space).
    vp_1d_window = 30
    vp_1d_bins = 36

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = (pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs)
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        if not getattr(self, "dp", None):
            return []
        pairs = self.dp.current_whitelist()
        return [(pair, "4h") for pair in pairs] + [(pair, "1d") for pair in pairs]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = self._add_volume_pressure(dataframe)
        dataframe = self._add_prior_period_levels(dataframe)
        dataframe = self._add_rolling_levels(dataframe)
        dataframe = self._add_avwap(dataframe)
        dataframe = self._add_supply_demand(dataframe)
        dataframe = self._add_liquidity_levels(dataframe)
        dataframe = self._add_volume_profile(dataframe, "vp", int(self.vp_window), int(self.vp_bins))
        dataframe = self._merge_informative_vp(dataframe, metadata, "4h", "vp4h", int(self.vp_4h_window), int(self.vp_4h_bins))
        dataframe = self._merge_informative_vp(dataframe, metadata, "1d", "vp1d", int(self.vp_1d_window), int(self.vp_1d_bins))
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        condition = self._avwap_reject(dataframe)
        if bool(self.use_volume_guard):
            condition &= self._volume_guard(dataframe)
        if bool(self.use_vp_1h_guard):
            condition &= self._vp_guard(dataframe, "vp", SIDE, str(self.vp_guard_mode), float(self.vp_score_min), float(self.vp_context_min))
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

    @staticmethod
    def _atr(dataframe: DataFrame, period: int = 14) -> Series:
        high = _num(dataframe, "high")
        low = _num(dataframe, "low")
        close = _num(dataframe, "close")
        previous_close = close.shift(1)
        true_range = pd.concat([high.sub(low), high.sub(previous_close).abs(), low.sub(previous_close).abs()], axis=1).max(axis=1)
        return true_range.rolling(period, min_periods=max(2, period // 2)).mean()

    @staticmethod
    def _bars_since(signal: Series) -> Series:
        clean = pd.Series(signal, index=signal.index).fillna(False).astype(bool)
        positions = pd.Series(np.arange(len(clean), dtype="float64"), index=clean.index)
        return positions.sub(positions.where(clean).ffill()).fillna(9999.0)

    @staticmethod
    def _period_key(dates: Series, period: str) -> Series:
        if period == "week":
            iso = dates.dt.isocalendar()
            return iso["year"].astype("string").str.cat(iso["week"].astype("string").str.zfill(2), sep="-")
        if period == "month":
            return dates.dt.strftime("%Y-%m")
        return dates.dt.strftime("%Y-%m-%d")

    def _add_volume_pressure(self, dataframe: DataFrame) -> DataFrame:
        window = int(self.volume_window)
        volume_mean = _num(dataframe, "volume").rolling(window, min_periods=max(2, window // 3)).mean()
        candle_range = _num(dataframe, "high").sub(_num(dataframe, "low")).replace(0.0, np.nan)
        close_location = _num(dataframe, "close").sub(_num(dataframe, "low")).div(candle_range).clip(0.0, 1.0)
        dataframe["entry_close_location"] = close_location
        dataframe["entry_volume_ratio"] = _num(dataframe, "volume").div(volume_mean.replace(0.0, np.nan))
        dataframe["entry_pressure"] = close_location.sub(0.5).mul(2.0)
        return dataframe

    def _add_prior_period_levels(self, dataframe: DataFrame) -> DataFrame:
        dates = pd.to_datetime(dataframe["date"], utc=True, errors="coerce")
        high = _num(dataframe, "high")
        low = _num(dataframe, "low")
        for period in PERIOD_CHOICES:
            key = self._period_key(dates, period)
            grouped = pd.DataFrame({"period": key, "high": high, "low": low}).groupby("period", sort=True).agg(period_high=("high", "max"), period_low=("low", "min"))
            grouped["prior_high"] = grouped["period_high"].shift(1)
            grouped["prior_low"] = grouped["period_low"].shift(1)
            dataframe[f"prior_{period}_high"] = key.map(grouped["prior_high"]).astype("float64")
            dataframe[f"prior_{period}_low"] = key.map(grouped["prior_low"]).astype("float64")
        return dataframe

    def _add_rolling_levels(self, dataframe: DataFrame) -> DataFrame:
        lookback = int(self.rolling_level_lookback)
        min_periods = max(4, lookback // 4)
        dataframe["rolling_resistance"] = _num(dataframe, "high").shift(1).rolling(lookback, min_periods=min_periods).max()
        dataframe["rolling_support"] = _num(dataframe, "low").shift(1).rolling(lookback, min_periods=min_periods).min()
        return dataframe

    def _add_avwap(self, dataframe: DataFrame) -> DataFrame:
        lookback = int(self.avwap_anchor_lookback)
        low = _num(dataframe, "low")
        high = _num(dataframe, "high")
        low_reset = low.shift(1).le(low.shift(1).rolling(lookback, min_periods=2).min())
        high_reset = high.shift(1).ge(high.shift(1).rolling(lookback, min_periods=2).max())
        dataframe["avwap_from_low"] = self._anchored_vwap_series(dataframe, low_reset)
        dataframe["avwap_from_high"] = self._anchored_vwap_series(dataframe, high_reset)
        typical = high.add(low).add(_num(dataframe, "close")).div(3.0)
        deviation = typical.rolling(lookback, min_periods=max(5, lookback // 5)).std()
        dataframe["avwap_lower_band"] = dataframe["avwap_from_low"].sub(deviation.mul(float(self.avwap_band_mult)))
        dataframe["avwap_upper_band"] = dataframe["avwap_from_high"].add(deviation.mul(float(self.avwap_band_mult)))
        return dataframe

    @staticmethod
    def _anchored_vwap_series(dataframe: DataFrame, reset: Series) -> Series:
        typical = _num(dataframe, "high").add(_num(dataframe, "low")).add(_num(dataframe, "close")).div(3.0)
        volume = _num(dataframe, "volume").clip(lower=0.0)
        groups = reset.fillna(False).astype(bool).cumsum()
        return typical.mul(volume).groupby(groups).cumsum().div(volume.groupby(groups).cumsum().replace(0.0, np.nan))

    def _add_supply_demand(self, dataframe: DataFrame) -> DataFrame:
        window = int(self.zone_impulse_window)
        open_ = _num(dataframe, "open")
        high = _num(dataframe, "high")
        low = _num(dataframe, "low")
        close = _num(dataframe, "close")
        body = close.sub(open_).abs()
        body_fraction = body.div(high.sub(low).replace(0.0, np.nan))
        volume_ratio = _num(dataframe, "volume").div(_num(dataframe, "volume").rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan))
        quality = body_fraction.ge(float(self.zone_body_fraction_min)) & volume_ratio.ge(float(self.zone_volume_ratio_min))
        impulse_distance = self._atr(dataframe).mul(float(self.zone_impulse_atr_min))
        bull_impulse = close.gt(open_) & close.sub(open_).ge(impulse_distance) & quality
        bear_impulse = close.lt(open_) & open_.sub(close).ge(impulse_distance) & quality
        body_low = pd.concat([open_, close], axis=1).min(axis=1)
        body_high = pd.concat([open_, close], axis=1).max(axis=1)
        dataframe["demand_zone_low"] = low.where(bull_impulse).ffill().shift(1)
        dataframe["demand_zone_high"] = body_low.where(bull_impulse).ffill().shift(1)
        dataframe["supply_zone_low"] = body_high.where(bear_impulse).ffill().shift(1)
        dataframe["supply_zone_high"] = high.where(bear_impulse).ffill().shift(1)
        dataframe["demand_zone_age"] = self._bars_since(bull_impulse).shift(1)
        dataframe["supply_zone_age"] = self._bars_since(bear_impulse).shift(1)
        return dataframe

    def _add_liquidity_levels(self, dataframe: DataFrame) -> DataFrame:
        lookback = int(self.equal_level_lookback)
        min_periods = max(4, lookback // 4)
        tolerance = float(self.equal_level_tolerance_pct)
        prior_high = _num(dataframe, "high").shift(1)
        prior_low = _num(dataframe, "low").shift(1)
        high_level = prior_high.rolling(lookback, min_periods=min_periods).max()
        low_level = prior_low.rolling(lookback, min_periods=min_periods).min()
        high_touches = prior_high.sub(high_level).abs().le(high_level.abs().mul(tolerance)).rolling(lookback, min_periods=min_periods).sum()
        low_touches = prior_low.sub(low_level).abs().le(low_level.abs().mul(tolerance)).rolling(lookback, min_periods=min_periods).sum()
        min_touches = int(self.equal_level_min_touches)
        dataframe["equal_high_level"] = high_level.where(high_touches.ge(min_touches))
        dataframe["equal_low_level"] = low_level.where(low_touches.ge(min_touches))
        dataframe["range_resistance"] = high_level
        dataframe["range_support"] = low_level
        return dataframe

    def _add_volume_profile(self, dataframe: DataFrame, prefix: str, window: int, bins: int) -> DataFrame:
        return add_volume_profile(
            dataframe,
            window=window,
            bins=bins,
            value_area_pct=float(self.vp_value_area_pct),
            price_source=str(self.vp_price_source),
            smooth_bins=int(self.vp_smooth_bins),
            hvn_threshold=float(self.vp_hvn_threshold),
            lvn_threshold=float(self.vp_lvn_threshold),
            pressure_delta_min=float(self.vp_pressure_delta_min),
            node_near_pct=float(self.vp_node_near_pct),
            volume_percentile_min=float(self.vp_volume_percentile_min),
            score_window=int(self.vp_score_window),
            fast_traverse_atr_mult=float(self.vp_fast_traverse_atr_mult),
            entry_score_margin=float(self.vp_entry_score_margin),
            prefix=prefix,
        )

    def _merge_informative_vp(self, dataframe: DataFrame, metadata: dict, timeframe: str, prefix: str, window: int, bins: int) -> DataFrame:
        if not getattr(self, "dp", None) or "date" not in dataframe.columns:
            return dataframe
        pair = metadata.get("pair") if metadata else None
        if not pair:
            return dataframe
        informative = self.dp.get_pair_dataframe(pair=pair, timeframe=timeframe)
        if informative is None or informative.empty or "date" not in informative.columns:
            return dataframe
        informative = self._add_volume_profile(informative.copy(), prefix, window, bins)
        merge_columns = [f"{prefix}_{name}" for name in VP_COLUMNS if f"{prefix}_{name}" in informative.columns]
        if not merge_columns:
            return dataframe
        informative = informative[["date", *merge_columns]].copy().sort_values("date")
        informative["date_merge"] = informative["date"] + pd.to_timedelta(timeframe_to_minutes(timeframe), unit="m")
        base = dataframe.reset_index().rename(columns={"index": "__row_index"}).sort_values("date")
        merged = pd.merge_asof(base, informative[["date_merge", *merge_columns]].sort_values("date_merge"), left_on="date", right_on="date_merge", direction="backward").sort_values("__row_index")
        for column in merge_columns:
            dataframe[column] = merged[column].to_numpy()
        return dataframe

    def _avwap_reject(self, dataframe: DataFrame) -> Series:
        close = _num(dataframe, "close")
        avwap = _num(dataframe, "avwap_from_high", np.nan)
        trigger = avwap.mul(1.0 - float(self.reclaim_buffer_pct))
        return close.le(trigger) & close.shift(1).gt(avwap.shift(1)) & _num(dataframe, "high").ge(avwap.mul(1.0 - float(self.zone_near_pct)))

    def _volume_guard(self, dataframe: DataFrame) -> Series:
        return _num(dataframe, "entry_volume_ratio").ge(float(self.volume_ratio_min)) & _num(dataframe, "entry_pressure").le(-float(self.pressure_min))

    @staticmethod
    def _close_direction_guard(dataframe: DataFrame) -> Series:
        close = _num(dataframe, "close")
        return close.lt(close.shift(1))

    @staticmethod
    def _score_guard(dataframe: DataFrame, prefix: str, side: str, score_min: float) -> Series:
        score = _num(dataframe, f"{prefix}_score_{side}")
        opposite = _num(dataframe, f"{prefix}_score_{'short' if side == 'long' else 'long'}")
        return score.ge(score_min) & score.ge(opposite)

    @staticmethod
    def _context_guard(dataframe: DataFrame, prefix: str, side: str, context_min: float) -> Series:
        if side == "long":
            context = _num(dataframe, f"{prefix}_context_score_bull")
            opposite = _num(dataframe, f"{prefix}_context_score_bear")
            market_ok = _num(dataframe, f"{prefix}_market_context").ge(0)
        else:
            context = _num(dataframe, f"{prefix}_context_score_bear")
            opposite = _num(dataframe, f"{prefix}_context_score_bull")
            market_ok = _num(dataframe, f"{prefix}_market_context").le(0)
        return context.ge(context_min) & context.ge(opposite) & market_ok

    def _vp_guard(self, dataframe: DataFrame, prefix: str, side: str, mode: str, score_min: float, context_min: float) -> Series:
        if mode == "score": return self._score_guard(dataframe, prefix, side, score_min)
        if mode == "context": return self._context_guard(dataframe, prefix, side, context_min)
        if mode == "score_or_context": return self._score_guard(dataframe, prefix, side, score_min) | self._context_guard(dataframe, prefix, side, context_min)
        if mode == "balance": return _num(dataframe, f"{prefix}_context_score_balance").ge(context_min)
        direction = _num(dataframe, f"{prefix}_market_context")
        return direction.ge(0) if side == "long" else direction.le(0)

    def _analyzed_frame(self, pair: str, current_time: datetime, trade: Any) -> DataFrame:
        if not getattr(self, "dp", None):
            raise RuntimeError("exit callback requires Freqtrade's analyzed dataframe")
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            raise RuntimeError("exit callback received no analyzed candles")
        dates = pd.to_datetime(dataframe["date"], utc=True, errors="raise")
        close_times = dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        cutoff = pd.Timestamp(current_time)
        cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
        entered = pd.Timestamp(getattr(trade, "date_entry_fill_utc", None) or trade.open_date_utc)
        entered = entered.tz_localize("UTC") if entered.tzinfo is None else entered.tz_convert("UTC")
        return dataframe.loc[dates.ge(entered) & close_times.le(cutoff)].sort_values("date")

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

    REVERSAL_PLANS = {
        "profit_1_bull_pressure_0": (0.01, 0.0),
        "profit_2_bull_pressure_0": (0.02, 0.0),
        "profit_2_bull_pressure_20": (0.02, 0.20),
        "profit_3_bull_pressure_20": (0.03, 0.20),
        "profit_4_bull_pressure_35": (0.04, 0.35),
    }
    reversal_plan = CategoricalParameter(tuple(REVERSAL_PLANS), default="profit_2_bull_pressure_20", space="sell", optimize=True, load=True)
    reversal_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    require_bull_vp_confirmation = BooleanParameter(default=False, space="sell", optimize=True, load=True)
    require_bull_vp_confirmation.batch_tags = ('family:exits', 'mode:sieve3_exit')
    ACTIVE_SELL_PARAMS = ('reversal_plan', 'require_bull_vp_confirmation')

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | None:
        _ = (current_rate, kwargs)
        profit_gate, pressure_gate = self.REVERSAL_PLANS[str(self.reversal_plan.value)]
        if current_profit < profit_gate:
            return None
        dataframe = self._analyzed_frame(pair, current_time, trade)
        if len(dataframe) < 2:
            return None
        last_two = dataframe.iloc[-2:]
        open_ = pd.to_numeric(last_two["open"], errors="coerce")
        close = pd.to_numeric(last_two["close"], errors="coerce")
        pressure = pd.to_numeric(last_two["entry_pressure"], errors="coerce")
        bullish_reversal = bool((close.gt(open_) & pressure.ge(pressure_gate)).all() and close.iloc[-1] > close.iloc[-2])
        if not bullish_reversal:
            return None
        if bool(self.require_bull_vp_confirmation.value):
            row = dataframe.iloc[-1]
            bull = float(row["vp_context_score_bull"])
            bear = float(row["vp_context_score_bear"])
            if not (bull >= bear and float(row["vp_market_context"]) >= 0.0):
                return None
        return f"profit_armed_bullish_reversal_{self.reversal_plan.value}"

    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs: Any) -> float:
        _ = (pair, current_time, current_profit, after_fill, kwargs)
        stop_price = float(trade.open_rate) * 1.03
        return stoploss_from_absolute(stop_price, current_rate=current_rate, is_short=True, leverage=float(trade.leverage or 1.0))
