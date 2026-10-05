"""Entry-frozen wedge-height measured-move exits for the locked 8h breakout."""
from __future__ import annotations
from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter, IStrategy, stoploss_from_absolute
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2


ENTRY_TAG = "geometry_wedge_breakout_long_8h"; STATE_KEY = "s3v2_wedge_measured_move"
SIEVE_STAGE = "sieve3"; ENTRY_SOURCE_STAGE = "sieve2"
SOURCE_STRATEGY = "Sieve2GeometryWedgeBreakoutLong8H tested snapshot 2026-05-23"
SOURCE_RESULT_BATCH = "20260521T012740_entry_all_resume line 246"
RESEARCH_PATH = "sieve3_exit_wedge_measured_move"
EXIT_HYPOTHESIS = "Exit in a valid upside zone projected from the entry wedge's full start width."
LOCKED_BUY_PARAMS = {"use_sieve2_vp_guard": False, "sieve2_vp_guard_mode": "score_or_context", "sieve2_vp_window": 96, "sieve2_vp_bins": 36, "sieve2_vp_score_min": 0.25, "sieve2_vp_context_min": 0.28, "use_sieve2_market_guard": False, "sieve2_market_guard_mode": "pressure_or_trend", "sieve2_market_window": 24, "sieve2_market_pressure_min": 0.07, "sieve2_market_trend_min": 0.25, "sieve2_rs_benchmark_pair": "BTC/USDT:USDT", "sieve2_rs_score_min": 0.45, "use_volume_guard": True, "volume_guard_window": 12, "volume_ratio_min": 1.6, "use_pressure_guard": True, "pressure_window": 24, "pressure_min": 0.1, "use_accumulation_guard": False, "use_body_direction_guard": False, "use_close_direction_guard": False, "min_pattern_bars": 12, "max_pattern_bars": 72, "compression_max_width_atr": 2.0, "squeeze_active_width_atr": 2.0, "min_line_score": 0.4, "min_containment": 0.88}
ACTIVE_SELL_PARAMS = ("exit_plan",)

def _num(
    frame: DataFrame, column: str, default: float | Series | object = ...
) -> Series:
    if column not in frame.columns:
        if default is ...:
            raise KeyError(f"Required dataframe column not found: {column}")
        return pd.to_numeric(
            pd.Series(default, index=frame.index), errors="coerce"
        ).replace([np.inf, -np.inf], np.nan)
    return pd.to_numeric(frame[column], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
def _bool(f: DataFrame, c: str) -> Series:
    if c not in f.columns: raise KeyError(f"Required dataframe column not found: {c}")
    return pd.Series(f[c], index=f.index).astype("boolean").fillna(False).astype(bool)

class Sieve3V2WedgeMeasuredMoveFromCompletePatternGeometryWedgeBreakoutLong8H(IStrategy):
    INTERFACE_VERSION = 3; timeframe = "8h"; startup_candle_count = 180; process_only_new_candles = True; can_short = False
    minimal_roi = {"0": 100.0}; stoploss = -0.99; use_exit_signal = True; use_custom_stoploss = True; trailing_stop = False; position_adjustment_enable = False
    
    
    
    
    
    
    
    exit_plan = CategoricalParameter(["height_75_zone2", "height_100_zone2", "height_125_exact"], default="height_100_zone2", space="sell", optimize=True, load=True)
    exit_plan.batch_tags = ("family:exits", "mode:sieve3_exit")
    ACTIVE_SELL_PARAMS = ("exit_plan",)
    TARGET_PLANS = {"height_75_zone2": (0.75, 0.02), "height_100_zone2": (1.0, 0.02), "height_125_exact": (1.25, 0.0)}

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float: return 1.0
    def informative_pairs(self) -> list[tuple[str, str]]: return []
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = add_pattern_geometry_v2(dataframe, timeframe=self.timeframe, output_slots=1, include_triangle_patterns=False, include_wedge_patterns=True, include_compression_patterns=False, include_rectangle_patterns=False, include_ascending_channel_patterns=False, include_descending_channel_patterns=False, min_pattern_bars=int(12), max_pattern_bars=int(72), compression_max_width_atr=float(2.0), squeeze_active_width_atr=float(2.0), min_line_score=float(0.4), min_containment=float(0.88), output_prefix="pg2")
        return dataframe
    def _common_guards(self, f: DataFrame) -> Series:
        g = pd.Series(True, index=f.index, dtype="bool"); close = _num(f, "close")
        volume = _num(f, "volume").clip(lower=0.0); w = int(12); g &= volume.ge(volume.shift(1).rolling(w, min_periods=max(2, w // 3)).mean().replace(0.0, np.nan).mul(float(1.6)))
        open_ = _num(f, "open"); high = _num(f, "high"); low = _num(f, "low"); volume = _num(f, "volume").clip(lower=0.0).fillna(0.0); r = (high - low).replace(0.0, np.nan); p = ((((close-open_)/r).clip(-1,1).fillna(0)+(((close-low)/r)*2-1).clip(-1,1).fillna(0))/2).clip(-1,1); dv=(p*volume).fillna(0)
        w=int(24); base=volume.rolling(w,min_periods=max(2,w//3)).sum().replace(0,np.nan); g &= dv.rolling(w,min_periods=max(2,w//3)).sum().div(base).ge(float(0.1))
        return g.fillna(False)
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"]=0; dataframe["enter_short"]=0; dataframe["enter_tag"]=None
        c=_bool(dataframe,"pg2_wedge_pattern_present")&_bool(dataframe,"pg2_wedge_squeeze_active")&_num(dataframe,"pg2_wedge_indicator_score").ge(float(0.4))&_num(dataframe,"pg2_wedge_direction").ge(0)&_num(dataframe,"close").gt(_num(dataframe,"pg2_wedge_upper")); c &= self._common_guards(dataframe); c=pd.Series(c, index=dataframe.index).fillna(False).astype(bool).fillna(False); v=c.fillna(False)&dataframe["volume"].gt(0)&dataframe["close"].notna(); dataframe.loc[v,"enter_long"]=1; dataframe.loc[v,"enter_tag"]=ENTRY_TAG; return dataframe
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame: dataframe["exit_long"]=0; dataframe["exit_short"]=0; dataframe["exit_tag"]=None; return dataframe
    @staticmethod
    def _utc(value: Any) -> pd.Timestamp:
        stamp = pd.Timestamp(value)
        return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")
    def _entry_signal_row(self, pair: str, entry_order_time: datetime) -> Series:
        if self.dp is None:
            raise RuntimeError("wedge measured-move exit requires Freqtrade's data provider")
        f,_=self.dp.get_analyzed_dataframe(pair,self.timeframe)
        required={"date","pg2_slot_1_active","pg2_slot_1_family","pg2_slot_1_upper_start","pg2_slot_1_lower_start","pg2_wedge_upper"}; missing=sorted(required-set(f.columns))
        if missing: raise KeyError(f"wedge measured-move exit missing columns: {missing}")
        closes=pd.to_datetime(f["date"],utc=True,errors="raise")+pd.to_timedelta(timeframe_to_minutes(self.timeframe),unit="m")
        rows=f.loc[closes.le(self._utc(entry_order_time))].sort_values("date")
        if rows.empty: raise RuntimeError("wedge measured-move exit cannot locate the closed entry signal candle")
        return rows.iloc[-1]
    def _state(self, pair: str, trade: Any, entry_order_time: datetime | None = None) -> dict[str, Any]:
        state=trade.get_custom_data(key=STATE_KEY)
        if state is not None:
            try:
                state=dict(state); required={"plan","target","zone","signal_candle","entry_filled_at"}
                if required-set(state): raise ValueError
                plan=str(state["plan"]); target=float(state["target"]); zone=float(state["zone"])
                signal_candle=pd.Timestamp(state["signal_candle"]); filled_at=pd.Timestamp(state["entry_filled_at"])
                expected_zone=self.TARGET_PLANS[plan][1]
            except (TypeError,ValueError,KeyError) as exc:
                raise RuntimeError("restored wedge measured-move state is incompatible") from exc
            if not np.isfinite([target,zone]).all() or target<=float(trade.open_rate) or not np.isclose(zone,expected_zone) or pd.isna(signal_candle) or pd.isna(filled_at):
                raise RuntimeError("restored wedge measured-move state is incompatible")
            return state
        if entry_order_time is None: raise RuntimeError("wedge measured-move state is missing after entry restoration")
        row=self._entry_signal_row(pair,entry_order_time); multiplier,zone=self.TARGET_PLANS[str(self.exit_plan.value)]; upper=float(row["pg2_slot_1_upper_start"]); lower=float(row["pg2_slot_1_lower_start"]); rail=float(row["pg2_wedge_upper"]); height=upper-lower
        valid=bool(row["pg2_slot_1_active"]) and int(row["pg2_slot_1_family"])==2 and np.isfinite([upper,lower,rail,height]).all() and height>0
        target=rail+height*multiplier
        if not valid or not np.isfinite(target) or target<=float(trade.open_rate): raise ValueError("entry wedge geometry does not define a target above the long trade open rate")
        state={"plan":str(self.exit_plan.value),"target":target,"zone":zone,"signal_candle":self._utc(row["date"]).isoformat(),"entry_filled_at":self._utc(entry_order_time).isoformat()}; trade.set_custom_data(key=STATE_KEY,value=state); return state
    def order_filled(self, pair: str, trade: Any, order: Any, current_time: datetime, **kwargs: Any) -> None:
        if getattr(order,"ft_order_side",None)==getattr(trade,"entry_side",None) and trade.get_custom_data(key=STATE_KEY) is None:
            filled_at=getattr(order,"order_filled_utc",None) or getattr(trade,"date_entry_fill_utc",None)
            if filled_at is None: raise RuntimeError("wedge measured-move exit requires the actual entry fill timestamp")
            self._state(pair,trade,filled_at)
    def custom_exit(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, **kwargs: Any) -> str | None:
        s=self._state(pair,trade); target=s.get("target"); trigger=max(float(target)*(1-float(s["zone"])),float(trade.open_rate)) if target is not None else None
        if trigger is None: return None
        f,_=self.dp.get_analyzed_dataframe(pair,self.timeframe); dates=pd.to_datetime(f["date"],utc=True,errors="raise"); filled=self._utc(s["entry_filled_at"]); closed=f.loc[dates.ge(filled)&(dates+pd.to_timedelta(timeframe_to_minutes(self.timeframe),unit="m")<=self._utc(current_time))]
        return "wedge_measured_move_zone" if not closed.empty and bool(pd.to_numeric(closed["high"],errors="coerce").ge(trigger).any()) else None
    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime, current_rate: float, current_profit: float, after_fill: bool, **kwargs: Any) -> float:
        return stoploss_from_absolute(float(trade.open_rate)*0.98,current_rate=current_rate,is_short=False,leverage=float(getattr(trade,"leverage",1.0) or 1.0))
