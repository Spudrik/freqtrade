from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


from freqtrade.exchange import timeframe_to_minutes  # noqa: E402
from freqtrade.strategy import (  # noqa: E402
    CategoricalParameter,
    IStrategy,
    stoploss_from_absolute,
)
from user_data.Indicators.pattern_bos_choch import add_bos_choch  # noqa: E402


# Source: selected Sieve2 snapshot in the 2026-05-21 8h BOS result archive.
# Exit family: full exits at entry-frozen bearish BOS structure objectives.
# Trigger/guard: bearish 8h BOS; bearish state plus locked volume/pressure guards.
# Target: fill-anchored projections of the frozen BOS structure height.
# Invalidation: entry-frozen ms_invalidation_level above the short fill.
# Active HyperOpt parameter: exit_plan. No branch-local inactive parameters.
# All modes share one source provider and differ only by target/8h confirmation.
ENTRY_MODE = "entry_bos_bear_continuation_short_8h"
ENTRY_TAG = "bos_bear_continuation_short_8h"
SIDE = "short"
TIMEFRAME = "8h"

SIEVE_STAGE = "sieve3"
ENTRY_SOURCE_STAGE = 'sieve2'
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
RESEARCH_PATH = 'sieve3_exit_source_structure_target'
EXIT_HYPOTHESIS = (
    "The 8h bearish BOS should realize profit at its entry-known prior swing or "
    "measured continuation objective before its frozen swing-high invalidation."
)
PRIMARY_TRIGGER = "confirmed 8h ms_bos_to_bear"
PRIMARY_GUARD = "ms_state <= 0 with locked 24-candle volume and pressure guards"
TARGET_PROVIDER = "fill-anchored projections of entry-frozen BOS structure height"
INVALIDATION_PROVIDER = "entry-frozen ms_invalidation_level"
ACTIVE_SELL_PARAMS = ("exit_plan",)
STATE_KEY = "s3v2_bos_bear_8h_source_structure_target"

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
    "measured_half_touch_full": ("measured_half", "touch"),
    "measured_half_close_full": ("measured_half", "close"),
    "measured_half_two_close_full": ("measured_half", "two_close"),
    "measured_full_touch_full": ("measured_full", "touch"),
    "measured_full_close_full": ("measured_full", "close"),
    "measured_full_two_close_full": ("measured_full", "two_close"),
    "measured_one_half_touch_full": ("measured_one_half", "touch"),
    "measured_one_half_close_full": ("measured_one_half", "close"),
    "measured_one_half_two_close_full": ("measured_one_half", "two_close"),
}




def _num(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        raise KeyError(f"required numeric column is missing: {column!r}")
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _bool(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        raise KeyError(f"required boolean column is missing: {column!r}")
    return pd.Series(frame[column], index=frame.index).astype("boolean").fillna(False).astype(bool)


EXIT_FAMILY = 'source_structure_target'




class Sieve3V2SourceStructureTargetFromBosBearContinuationShort8H(IStrategy):
    ACTIVE_SELL_PARAMS = ('exit_plan',)
    INTERFACE_VERSION = 3
    timeframe = TIMEFRAME
    startup_candle_count = 160
    process_only_new_candles = True
    can_short = True

    minimal_roi = {"0": 1.0}
    stoploss = -0.99
    use_exit_signal = True
    exit_profit_only = False
    use_custom_stoploss = True
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
    # Legacy inactive entry parameter: use_accumulation_guard=False (proven inactive in the fixed entry path).
    # Legacy inactive entry parameter: use_body_direction_guard=False (proven inactive in the fixed entry path).
    # Legacy inactive entry parameter: use_close_direction_guard=False (proven inactive in the fixed entry path).
    strength = 3
    min_prominence_atr = 0.35
    min_pivot_spacing_bars = 2
    max_pivot_age_bars = 96
    breakout_buffer_atr = 0.3
    use_state_guard = True

    exit_plan = CategoricalParameter(tuple(EXIT_PLANS), default="measured_half_touch_full", space="sell", optimize=True, load=True)
    exit_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')

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

    def _closed_frame(self, pair: str, current_time: datetime) -> DataFrame:
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        required = {"date", "open", "high", "low", "close"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise KeyError(f"missing analyzed columns: {missing}")
        now = pd.Timestamp(current_time)
        if now.tzinfo is None:
            now = now.tz_localize("UTC")
        else:
            now = now.tz_convert("UTC")
        dates = pd.to_datetime(frame["date"], utc=True, errors="raise")
        close_times = dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        return frame.loc[close_times.le(now)].copy()

    @staticmethod
    def _finite(row: Series, column: str) -> float:
        value = float(row[column])
        if not np.isfinite(value):
            raise ValueError(f"entry source column {column} is not finite")
        return value

    def _freeze_structure(self, pair: str, trade: Any) -> dict[str, Any]:
        fill_time = getattr(trade, "date_entry_fill_utc", None) or getattr(trade, "open_date_utc", None)
        if fill_time is None:
            raise ValueError("source structure target requires an entry fill timestamp")
        fill_timestamp = pd.Timestamp(fill_time)
        if fill_timestamp.tzinfo is None:
            fill_timestamp = fill_timestamp.tz_localize("UTC")
        else:
            fill_timestamp = fill_timestamp.tz_convert("UTC")
        frame = self._closed_frame(pair, fill_timestamp)
        required = {"ms_break_level", "ms_invalidation_level"}
        missing = sorted(required.difference(frame.columns))
        if missing or frame.empty:
            raise KeyError(f"cannot freeze source structure; missing={missing}")
        row = frame.iloc[-1]
        entry = float(trade.open_rate)
        break_level = self._finite(row, "ms_break_level")
        invalidation = self._finite(row, "ms_invalidation_level")
        if invalidation <= 0.0 or invalidation <= entry:
            raise ValueError("short structure invalidation must be positive, finite, and above entry")
        structure_height = invalidation - break_level
        if structure_height <= 0.0:
            raise ValueError("bearish BOS structure height must be positive")
        measured_half = entry - structure_height * 0.5
        measured_full = entry - structure_height
        targets = {
            "measured_half": measured_half,
            "measured_full": measured_full,
            "measured_one_half": entry - structure_height * 1.5,
        }
        if any(not np.isfinite(value) or value <= 0.0 or value >= entry for value in targets.values()):
            raise ValueError("all short structure targets must be positive, finite, and below entry")
        state = {
            "entry_filled_at": fill_timestamp.isoformat(),
            "invalidation": invalidation,
            "targets": targets,
            "target_reached": {},
        }
        trade.set_custom_data(STATE_KEY, state)
        return state

    def _structure(self, pair: str, trade: Any) -> dict[str, Any]:
        state = trade.get_custom_data(STATE_KEY)
        if isinstance(state, Mapping):
            restored = dict(state)
            changed = False
            if not restored.get("entry_filled_at"):
                fill_time = getattr(trade, "date_entry_fill_utc", None) or getattr(trade, "open_date_utc", None)
                if fill_time is None:
                    raise ValueError("source structure target requires an entry fill timestamp")
                fill_timestamp = pd.Timestamp(fill_time)
                if fill_timestamp.tzinfo is None:
                    fill_timestamp = fill_timestamp.tz_localize("UTC")
                else:
                    fill_timestamp = fill_timestamp.tz_convert("UTC")
                restored["entry_filled_at"] = fill_timestamp.isoformat()
                changed = True
            target_reached = restored.get("target_reached")
            if target_reached is None:
                restored["target_reached"] = {}
                changed = True
            elif not isinstance(target_reached, Mapping):
                raise ValueError("restored target latch state must be a mapping")
            targets = restored.get("targets")
            if not isinstance(targets, Mapping):
                raise ValueError("restored source structure targets must be a mapping")
            entry = float(trade.open_rate)
            if any(
                not np.isfinite(float(value)) or float(value) <= 0.0 or float(value) >= entry
                for value in targets.values()
            ):
                raise ValueError("all restored short structure targets must be positive, finite, and below entry")
            if changed:
                trade.set_custom_data(STATE_KEY, restored)
            return restored
        return self._freeze_structure(pair, trade)

    def _post_entry_closed(
        self,
        pair: str,
        state: Mapping[str, Any],
        current_time: datetime,
    ) -> DataFrame:
        frame = self._closed_frame(pair, current_time)
        filled_at = pd.Timestamp(state["entry_filled_at"])
        if filled_at.tzinfo is None:
            filled_at = filled_at.tz_localize("UTC")
        else:
            filled_at = filled_at.tz_convert("UTC")
        now = pd.Timestamp(current_time)
        if now.tzinfo is None:
            now = now.tz_localize("UTC")
        else:
            now = now.tz_convert("UTC")
        dates = pd.to_datetime(frame["date"], utc=True, errors="raise")
        close_times = dates + pd.to_timedelta(timeframe_to_minutes(self.timeframe), unit="m")
        return frame.loc[dates.ge(filled_at) & close_times.le(now)].copy()

    @staticmethod
    def _target_reached(frame: DataFrame, target: float, confirmation: str) -> bool:
        if frame.empty:
            return False
        if confirmation == "touch":
            lows = pd.to_numeric(frame["low"], errors="coerce")
            return bool(lows.le(target).any())
        if confirmation == "close":
            closes = pd.to_numeric(frame["close"], errors="coerce")
            return bool(closes.le(target).any())
        if confirmation == "two_close":
            closes = pd.to_numeric(frame["close"], errors="coerce")
            confirmed = closes.le(target).rolling(window=2, min_periods=2).sum().ge(2)
            return bool(confirmed.any())
        raise ValueError(f"unsupported target confirmation: {confirmation}")

    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | None:
        _ = current_profit, kwargs
        state = self._structure(pair, trade)
        target_name, confirmation = EXIT_PLANS[str(self.exit_plan.value)]
        target = float(state["targets"][target_name])
        latch_key = f"{target_name}:{confirmation}"
        reached = dict(state["target_reached"])
        target_reached = bool(reached.get(latch_key))
        if not target_reached:
            target_reached = self._target_reached(
                self._post_entry_closed(pair, state, current_time),
                target,
                confirmation,
            )
            if target_reached:
                reached[latch_key] = True
                state["target_reached"] = reached
                trade.set_custom_data(STATE_KEY, state)
        if current_rate >= float(state["invalidation"]):
            return "s3v2_source_invalidation"
        if target_reached:
            return f"s3v2_{target_name}_{confirmation}"
        return None

    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs: Any) -> float:
        _ = current_time, current_profit, after_fill, kwargs
        stop_rate = float(self._structure(pair, trade)["invalidation"])
        return stoploss_from_absolute(stop_rate, current_rate=current_rate, is_short=True, leverage=float(trade.leverage or 1.0))

    def order_filled(self, pair: str, trade: Any, order: Any, current_time: datetime, **kwargs: Any) -> None:
        _ = current_time, kwargs
        if order.ft_order_side == trade.entry_side and trade.get_custom_data(STATE_KEY) is None:
            self._freeze_structure(pair, trade)
