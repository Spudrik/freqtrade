from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


from freqtrade.exchange import timeframe_to_prev_date  # noqa: E402
from freqtrade.strategy import CategoricalParameter, IStrategy, stoploss_from_absolute
from user_data.Indicators.pattern_bos_choch import add_bos_choch  # noqa: E402


# Source: selected Sieve2 snapshot in the 2026-05-21 8h BOS result archive.
# Exit family: fixed targets with broken-support reclaim as initial protection.
# Trigger/guard: bearish 8h BOS; bearish state plus locked volume/pressure guards.
# Target: fixed favorable move; invalidation: entry-frozen ms_break_level.
# Active HyperOpt parameter: exit_plan. No branch-local inactive parameters.
# Split from swing-high invalidation because the level and expected risk differ.
ENTRY_MODE = "entry_bos_bear_continuation_short_8h"
ENTRY_TAG = "bos_bear_continuation_short_8h"
SIDE = "short"
TIMEFRAME = "8h"

SIEVE_STAGE = "sieve3"
ENTRY_SOURCE_STAGE = "sieve2"
SOURCE_STRATEGY = (
    "Sieve2BOSBearContinuationShort8H snapshot:"
    "backtest-result-2026-05-21_18-40-06_Sieve2BOSBearContinuationShort8H.py"
)
SOURCE_RESULT_BATCH = "20260521T012740_entry_all_resume"
SOURCE_PARAMS_FILE = (
    "user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve/params/"
    "Sieve2BOSBearContinuationShort8H__pattern_continuation_8h_2020_23.json"
)
LINEAGE_STATUS = "selected_sieve2_params_verified; promotion_label_ambiguous"
RESEARCH_PATH = 'sieve3_exit_source_invalidation'
EXIT_FAMILY = "source_invalidation"
EXIT_HYPOTHESIS = (
    "A bearish BOS continuation is invalid as soon as price reclaims its frozen "
    "break level, so fixed targets should be tested against that tight initial stop."
)
PRIMARY_TRIGGER = "confirmed 8h ms_bos_to_bear"
PRIMARY_GUARD = "ms_state <= 0 with locked 24-candle volume and pressure guards"
TARGET_PROVIDER = "fixed favorable move from trade.open_rate"
PRIMARY_TARGET = TARGET_PROVIDER
INVALIDATION_PROVIDER = "entry-frozen ms_break_level; an already-reclaimed fill exits"
PRIMARY_INVALIDATION = INVALIDATION_PROVIDER
ACTIVE_SELL_PARAMS = ("exit_plan",)
STATE_KEY = "s3v2_bos_bear_8h_broken_support_stop"

LOCKED_BUY_PARAMS: dict[str, Any] = {
    "use_sieve2_vp_guard": False, "sieve2_vp_guard_mode": "score_or_context",
    "sieve2_vp_window": 96, "sieve2_vp_bins": 36,
    "sieve2_vp_score_min": 0.25, "sieve2_vp_context_min": 0.28,
    "use_sieve2_market_guard": False,
    "sieve2_market_guard_mode": "pressure_or_trend",
    "sieve2_market_window": 24, "sieve2_market_pressure_min": 0.07,
    "sieve2_market_trend_min": 0.25,
    "sieve2_rs_benchmark_pair": "BTC/USDT:USDT", "sieve2_rs_score_min": 0.45,
    "use_volume_guard": True, "volume_guard_window": 24, "volume_ratio_min": 1.0,
    "use_pressure_guard": True, "pressure_window": 24, "pressure_min": 0.1,
    "use_accumulation_guard": False, "use_body_direction_guard": False,
    "use_close_direction_guard": False, "strength": 3,
    "min_prominence_atr": 0.35, "min_pivot_spacing_bars": 2,
    "max_pivot_age_bars": 96, "breakout_buffer_atr": 0.3,
    "use_state_guard": True,
}

EXIT_PLANS = {
    "broken_support_stop_target_2": 0.02,
    "broken_support_stop_target_3": 0.03,
    "broken_support_stop_target_4": 0.04,
    "broken_support_stop_target_5": 0.05,
    "broken_support_stop_target_6": 0.06,
    "broken_support_stop_target_8": 0.08,
    "broken_support_stop_target_10": 0.10,
    "broken_support_stop_target_12": 0.12,
}




def _num(frame: DataFrame, column: str, default: float | Series | None = None) -> Series:
    if column not in frame.columns:
        if default is None:
            raise KeyError(f"required numeric column is missing: {column!r}")
        if isinstance(default, Series):
            return pd.to_numeric(default, errors="coerce")
        return pd.Series(float(default), index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _bool(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        raise KeyError(f"required boolean column is missing: {column!r}")
    return pd.Series(frame[column], index=frame.index).astype("boolean").fillna(False).astype(bool)


class Sieve3V2SourceInvalidationBrokenSupportFixedTargetsFromBosBearContinuationShort8H(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 160
    process_only_new_candles = True
    can_short = True

    minimal_roi: dict[str, float] = {}
    stoploss = -0.99
    use_exit_signal = True
    use_custom_stoploss = True
    use_custom_roi = True
    position_adjustment_enable = False
    max_entry_position_adjustment = 0
    trailing_stop = False
    ignore_roi_if_entry_signal = False

    
    
    
    
    
    
    
    
    
    
    
    
    
    use_volume_guard = True
    volume_guard_window = 24
    volume_ratio_min = 1.0
    use_pressure_guard = True
    pressure_window = 24
    pressure_min = 0.1
    # Legacy fixed-off entry parameter: use_accumulation_guard=False (BooleanParameter, buy space).
    # Legacy fixed-off entry parameter: use_body_direction_guard=False (BooleanParameter, buy space).
    # Legacy fixed-off entry parameter: use_close_direction_guard=False (BooleanParameter, buy space).
    strength = 3
    min_prominence_atr = 0.35
    min_pivot_spacing_bars = 2
    max_pivot_age_bars = 96
    breakout_buffer_atr = 0.3
    use_state_guard = True

    exit_plan = CategoricalParameter(tuple(EXIT_PLANS), default="broken_support_stop_target_4", space="sell", optimize=True, load=True)
    exit_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    ACTIVE_SELL_PARAMS = ('exit_plan',)

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = add_bos_choch(
            dataframe,
            strength=int(self.strength),
            min_prominence_atr=float(self.min_prominence_atr),
            min_pivot_spacing_bars=int(self.min_pivot_spacing_bars),
            max_pivot_age_bars=int(self.max_pivot_age_bars),
            breakout_buffer_atr=float(self.breakout_buffer_atr),
            include_diagnostics=True,
            prefix="ms",
        )
        return dataframe

    def _common_guards(self, dataframe: DataFrame) -> Series:
        guard = pd.Series(True, index=dataframe.index, dtype="bool")
        close = _num(dataframe, "close")
        if bool(self.use_volume_guard):
            volume = _num(dataframe, "volume").clip(lower=0.0)
            window = int(self.volume_guard_window)
            baseline = volume.shift(1).rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan)
            guard &= volume.ge(baseline.mul(float(self.volume_ratio_min)))
        if bool(self.use_pressure_guard):
            open_ = _num(dataframe, "open")
            high = _num(dataframe, "high")
            low = _num(dataframe, "low")
            volume = _num(dataframe, "volume").clip(lower=0.0).fillna(0.0)
            candle_range = (high - low).replace(0.0, np.nan)
            body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
            close_location = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)
            directional_volume = (((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0).clip(-1.0, 1.0) * volume).fillna(0.0)
        if bool(self.use_pressure_guard):
            window = int(self.pressure_window)
            pressure_baseline = volume.rolling(window, min_periods=max(2, window // 3)).sum().replace(0.0, np.nan)
            pressure_ratio = directional_volume.rolling(window, min_periods=max(2, window // 3)).sum() / pressure_baseline
            guard &= pressure_ratio.le(-float(self.pressure_min))
        return guard.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        condition = _bool(dataframe, "ms_bos_to_bear")
        if bool(self.use_state_guard):
            condition &= _num(dataframe, "ms_state").le(0)
        condition &= self._common_guards(dataframe)
        
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

    def _freeze_break_level(self, pair: str, trade: Any) -> dict[str, float]:
        if getattr(self, "dp", None) is None:
            raise RuntimeError("broken-support exit runtime requires Freqtrade's data provider")
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if frame is None or frame.empty:
            raise RuntimeError("broken-support exit runtime requires a non-empty analyzed dataframe")
        if "date" not in frame.columns or "ms_break_level" not in frame.columns:
            raise KeyError("cannot freeze ms_break_level from entry candle")
        boundary = pd.Timestamp(timeframe_to_prev_date(self.timeframe, trade.open_date_utc))
        boundary = boundary.tz_localize("UTC") if boundary.tzinfo is None else boundary.tz_convert("UTC")
        dates = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        closed = frame.loc[dates.lt(boundary)]
        if closed.empty:
            raise ValueError("entry candle is unavailable for break-level freeze")
        stop_rate = float(closed.iloc[-1]["ms_break_level"])
        if not np.isfinite(stop_rate) or stop_rate <= 0.0:
            raise ValueError("short broken-support stop must be positive and finite")
        state = {"stop_rate": stop_rate}
        trade.set_custom_data(STATE_KEY, state)
        return state

    def _state(self, pair: str, trade: Any) -> dict[str, float]:
        state = trade.get_custom_data(STATE_KEY)
        if state is None:
            return self._freeze_break_level(pair, trade)
        if not isinstance(state, Mapping):
            raise ValueError("broken-support trade state must be a mapping")
        try:
            stop_rate = float(state["stop_rate"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("broken-support trade state has no numeric stop_rate") from exc
        if not np.isfinite(stop_rate) or stop_rate <= 0.0:
            raise ValueError("broken-support trade state stop_rate must be positive and finite")
        return {"stop_rate": stop_rate}

    def custom_roi(self, pair: str, trade: Any, current_time: datetime, trade_duration: int, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = pair, trade, current_time, trade_duration, entry_tag, side, kwargs
        return EXIT_PLANS[str(self.exit_plan.value)]

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | None:
        _ = current_time, current_profit, kwargs
        if current_rate >= float(self._state(pair, trade)["stop_rate"]):
            return "s3v2_broken_support_reclaimed"
        return None

    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs: Any) -> float:
        _ = current_time, current_profit, after_fill, kwargs
        stop_rate = float(self._state(pair, trade)["stop_rate"])
        return stoploss_from_absolute(stop_rate, current_rate=current_rate, is_short=True, leverage=float(trade.leverage or 1.0))

    def order_filled(self, pair: str, trade: Any, order: Any, current_time: datetime, **kwargs: Any) -> None:
        _ = current_time, kwargs
        if order.ft_order_side == trade.entry_side and trade.get_custom_data(STATE_KEY) is None:
            self._freeze_break_level(pair, trade)
