from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


from freqtrade.strategy import (
    CategoricalParameter,
    IStrategy,
    stoploss_from_absolute,
)
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2

ENTRY_MODE = "entry_geometry_wedge_breakout_long_1h"
ENTRY_TAG = "geometry_wedge_breakout_long_1h"
SIDE = "long"
TIMEFRAME = "1h"

SIEVE_STAGE = "sieve3"
SOURCE_ENTRY_STAGE = "sieve2"
SOURCE_RESULT_BATCH = "20260521T012740_entry_all_resume:523"
SOURCE_STRATEGY = (
    "D:\\FreqTradeStuffLargeData\\sieve_runtime\\entry_sieve\\backtests\\"
    "sieve2_geometry_wedge_breakout_long_1h__pattern_geometry_1h_2023_24_compression\\"
    "full_cycle_2020_2026\\tp_4_sl_2\\backtest-result-2026-05-26_02-54-35.zip!"
    "backtest-result-2026-05-26_02-54-35_Sieve2GeometryWedgeBreakoutLong1H.py:"
    "Sieve2GeometryWedgeBreakoutLong1H"
)
RESEARCH_PATH = "sieve3_exit_fixed_rr"
EXIT_HYPOTHESIS = "Compare named fixed reward/risk exits while preserving the locked geometry wedge breakout long entry."

# Preflight: wedge upper-rail breakout; locked volume/pressure guards; fixed price target;
# fixed loss invalidation. The single named-plan parameter keeps target and stop coherent.
ACTIVE_SELL_PARAMS = ("exit_plan",)
BRANCH_LOCAL_PARAMS: tuple[str, ...] = ()




def _num(frame: DataFrame, column: str) -> Series:
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)



def _bool(frame: DataFrame, column: str) -> Series:
    return pd.Series(frame[column], index=frame.index).astype("boolean").fillna(False).astype(bool)



ENTRY_SOURCE_STAGE = "sieve2"


class Sieve3V2FixedRrFromCompletePatternGeometryWedgeBreakoutLong1H(IStrategy):
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 180
    process_only_new_candles = True
    can_short = False

    minimal_roi = {"0": 100.0}
    stoploss = -0.99
    use_exit_signal = True
    use_custom_stoploss = True
    position_adjustment_enable = False
    max_entry_position_adjustment = 0
    trailing_stop = False
    ignore_roi_if_entry_signal = False


    use_volume_guard = True
    volume_guard_window = 24
    volume_ratio_min = 1.6
    use_pressure_guard = True
    pressure_window = 48
    pressure_min = 0.15
    # Legacy fixed-off entry parameter: use_accumulation_guard=False; its branch is unreachable.
    # Legacy fixed-off entry parameters: use_body_direction_guard=False and
    # use_close_direction_guard=False; their directional branches are unreachable.
    min_pattern_bars = 12
    max_pattern_bars = 72
    compression_max_width_atr = 2.0
    squeeze_active_width_atr = 2.0
    min_line_score = 0.55
    min_containment = 0.88

    ACTIVE_SELL_PARAMS = ('exit_plan',)

    exit_plan = CategoricalParameter(
        ["sym_2_2", "baseline_3_3", "sym_4_4", "sym_5_5", "reward_3_2", "reward_4_2", "reward_5_2", "reward_6_3", "reward_8_4", "reward_10_5", "defensive_2_3", "wide_12_6"],
        default="baseline_3_3",
        space="sell",
        optimize=True,
        load=True,
    )
    exit_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    EXIT_PLANS = {
        "sym_2_2": (0.02, 0.02), "baseline_3_3": (0.03, 0.03), "sym_4_4": (0.04, 0.04),
        "sym_5_5": (0.05, 0.05), "reward_3_2": (0.03, 0.02), "reward_4_2": (0.04, 0.02),
        "reward_5_2": (0.05, 0.02), "reward_6_3": (0.06, 0.03), "reward_8_4": (0.08, 0.04),
        "reward_10_5": (0.10, 0.05), "defensive_2_3": (0.02, 0.03), "wide_12_6": (0.12, 0.06),
    }

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float:
        _ = pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs
        return 1.0

    def informative_pairs(self) -> list[tuple[str, str]]:
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = add_pattern_geometry_v2(
            dataframe, timeframe=self.timeframe, output_slots=1,
            include_triangle_patterns=False, include_wedge_patterns=True,
            include_compression_patterns=False, include_rectangle_patterns=False,
            include_ascending_channel_patterns=False, include_descending_channel_patterns=False,
            min_pattern_bars=int(self.min_pattern_bars), max_pattern_bars=int(self.max_pattern_bars),
            compression_max_width_atr=float(self.compression_max_width_atr),
            squeeze_active_width_atr=float(self.squeeze_active_width_atr),
            min_line_score=float(self.min_line_score), min_containment=float(self.min_containment),
            output_prefix="pg2",
        )
        return dataframe

    def _common_guards(self, dataframe: DataFrame) -> Series:
        guard = pd.Series(True, index=dataframe.index, dtype="bool")
        close = _num(dataframe, "close")
        if self.use_volume_guard:
            volume = _num(dataframe, "volume").clip(lower=0.0)
            window = int(self.volume_guard_window)
            baseline = volume.shift(1).rolling(window, min_periods=max(2, window // 3)).mean().replace(0.0, np.nan)
            guard &= volume.ge(baseline.mul(float(self.volume_ratio_min)))
        if self.use_pressure_guard:
            open_ = _num(dataframe, "open")
            high = _num(dataframe, "high")
            low = _num(dataframe, "low")
            volume = _num(dataframe, "volume").clip(lower=0.0).fillna(0.0)
            candle_range = (high - low).replace(0.0, np.nan)
            body_pressure = ((close - open_) / candle_range).clip(-1.0, 1.0)
            close_location = (((close - low) / candle_range) * 2.0 - 1.0).clip(-1.0, 1.0)
            pressure = ((body_pressure.fillna(0.0) + close_location.fillna(0.0)) / 2.0).clip(-1.0, 1.0)
            directional_volume = (pressure * volume).fillna(0.0)
        if self.use_pressure_guard:
            window = int(self.pressure_window)
            pressure_baseline = volume.rolling(window, min_periods=max(2, window // 3)).sum().replace(0.0, np.nan)
            pressure_ratio = directional_volume.rolling(window, min_periods=max(2, window // 3)).sum() / pressure_baseline
            guard &= pressure_ratio.ge(float(self.pressure_min))
        return guard.fillna(False)

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        condition = (
            _bool(dataframe, "pg2_wedge_pattern_present")
            & _bool(dataframe, "pg2_wedge_squeeze_active")
            & _num(dataframe, "pg2_wedge_indicator_score").ge(float(self.min_line_score))
            & _num(dataframe, "pg2_wedge_direction").ge(0)
            & _num(dataframe, "close").gt(_num(dataframe, "pg2_wedge_upper"))
        )
        condition &= self._common_guards(dataframe)
        valid = condition.fillna(False) & dataframe["volume"].gt(0.0) & dataframe["close"].notna()
        dataframe.loc[valid, "enter_long"] = 1
        dataframe.loc[valid, "enter_tag"] = ENTRY_TAG
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        _ = metadata
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None
        return dataframe

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | None:
        _ = pair, trade, current_time, current_rate, kwargs
        target, loss = self.EXIT_PLANS[str(self.exit_plan.value)]
        if current_profit >= target:
            return f"fixed_rr_target_{self.exit_plan.value}"
        if current_profit <= -loss:
            return f"fixed_rr_stop_{self.exit_plan.value}"
        return None

    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs: Any) -> float | None:
        _ = pair, current_time, current_profit, after_fill, kwargs
        _, loss = self.EXIT_PLANS[str(self.exit_plan.value)]
        stop_price = float(trade.open_rate) * (1.0 - loss)
        if current_rate <= stop_price:
            return None
        return stoploss_from_absolute(stop_price, current_rate=current_rate, is_short=False, leverage=float(getattr(trade, "leverage", 1.0) or 1.0))


LOCKED_BUY_PARAMS = {
    "use_sieve2_vp_guard": False, "sieve2_vp_guard_mode": "score_or_context", "sieve2_vp_window": 96,
    "sieve2_vp_bins": 36, "sieve2_vp_score_min": 0.25, "sieve2_vp_context_min": 0.28,
    "use_sieve2_market_guard": False, "sieve2_market_guard_mode": "pressure_or_trend", "sieve2_market_window": 24,
    "sieve2_market_pressure_min": 0.07, "sieve2_market_trend_min": 0.25,
    "sieve2_rs_benchmark_pair": "BTC/USDT:USDT", "sieve2_rs_score_min": 0.45,
    "use_volume_guard": True, "volume_guard_window": 24, "volume_ratio_min": 1.6,
    "use_pressure_guard": True, "pressure_window": 48, "pressure_min": 0.15,
    "use_accumulation_guard": False, "use_body_direction_guard": False, "use_close_direction_guard": False,
    "min_pattern_bars": 12, "max_pattern_bars": 72, "compression_max_width_atr": 2.0,
    "squeeze_active_width_atr": 2.0, "min_line_score": 0.55, "min_containment": 0.88,
}
