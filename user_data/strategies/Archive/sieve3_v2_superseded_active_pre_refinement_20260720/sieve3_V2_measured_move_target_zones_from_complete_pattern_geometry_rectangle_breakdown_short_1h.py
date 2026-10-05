"""Rectangle-height measured-move target zones for the frozen 1h breakdown short."""
from __future__ import annotations
import math
from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd
from pandas import DataFrame,Series
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter, IStrategy
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2
ENTRY_MODE,ENTRY_TAG,SIDE,TIMEFRAME="entry_geometry_rectangle_breakdown_short_1h","geometry_rectangle_breakdown_short_1h","short","1h"
ENTRY_LOCK_STATUS="current executable defaults locked; promoted historical snapshots conflict"
TARGET_PLANS={"half_height_zone_0_25":(.50,.0025),"three_quarter_height_zone_0_25":(.75,.0025),"full_height_exact":(1.,0.),"full_height_zone_0_50":(1.,.005),"one_and_quarter_height_zone_0_50":(1.25,.005)}
def tagged_exit_parameter(p:Any)->Any:setattr(p,"batch_tags",("family:exits","mode:sieve3_exit"));return p
def _num(f: DataFrame, c: str) -> Series:
    return pd.to_numeric(f[c],errors="coerce").replace([np.inf,-np.inf],np.nan)

def _bool(f: DataFrame, c: str) -> Series:
    return pd.Series(f[c],index=f.index).astype("boolean").fillna(False).astype(bool)

def _cross_below(v:Series,l:Series)->Series:return v.lt(l)&v.shift(1).ge(l.shift(1))
SIEVE_STAGE = "sieve3"
ENTRY_SOURCE_STAGE = "sieve2"
RESEARCH_PATH = "sieve3_exit_measured_move_target_zones"
SOURCE_RESULT_BATCH = "historical_promotion_lineage_unknown"
SOURCE_STRATEGY = "sieve2_complete_pattern_geometry_rectangle_breakdown_short_1h"
EXIT_HYPOTHESIS = "Test the measured move target zones exit family while preserving the complete pattern geometry rectangle breakdown short 1h entry behavior."


class Sieve3V2MeasuredMoveTargetZonesFromCompletePatternGeometryRectangleBreakdownShort1H(IStrategy):
    INTERFACE_VERSION,timeframe,startup_candle_count,process_only_new_candles,can_short=3,TIMEFRAME,180,True,True
    minimal_roi,stoploss,use_exit_signal,use_custom_stoploss={"0":100.},-.03,True,False
    position_adjustment_enable,trailing_stop,exit_profit_only,ignore_roi_if_entry_signal=False,False,False,False
    
    
    # Inactive buy declaration: use_volume_guard = BooleanParameter(default=False, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: volume_guard_window = CategoricalParameter(default=48, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: volume_ratio_min = CategoricalParameter(default=1.0, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: use_pressure_guard = BooleanParameter(default=True, space="buy"); fixed-on entry gate makes its guard unconditional.
    # Inactive buy declaration: use_accumulation_guard = BooleanParameter(default=False, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: use_body_direction_guard = BooleanParameter(default=False, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: use_close_direction_guard = BooleanParameter(default=False, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    pressure_window = 24; pressure_min = 0.15
    min_pattern_bars = 12;max_pattern_bars = 72;compression_max_width_atr = 2.0;squeeze_active_width_atr = 2.0;local_narrowing_min_ratio = 0.1;min_line_score = 0.7;min_containment = 0.88;max_recent_touch_age_bars = 12;channel_min_pattern_bars = 18;channel_max_pattern_bars = 96;channel_min_quality = 0.82;channel_min_containment = 0.68;channel_near_boundary_atr_mult = 0.7;channel_breakout_atr_mult = 0.35;channel_lifecycle_confirm_break_bars = 2;score_min = 0.65;width_atr_max = 6.0;rail_buffer_pct = 0.004
    ACTIVE_SELL_PARAMS = ('target_plan',)

    target_plan=tagged_exit_parameter(CategoricalParameter(tuple(TARGET_PLANS),default="full_height_zone_0_50",space="sell",optimize=True,load=True))
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
        _ = (pair, current_time, current_rate, proposed_leverage, max_leverage, entry_tag, side, kwargs)
        return 1.0
    def informative_pairs(self)->list[tuple[str,str]]:return[]
    def populate_indicators(self,dataframe:DataFrame,metadata:dict)->DataFrame:
        dataframe=add_pattern_geometry_v2(dataframe,timeframe=self.timeframe,output_slots=1,include_triangle_patterns=False,include_wedge_patterns=False,include_compression_patterns=False,include_rectangle_patterns=True,include_ascending_channel_patterns=False,include_descending_channel_patterns=False,min_pattern_bars=int(self.min_pattern_bars),max_pattern_bars=int(self.max_pattern_bars),compression_max_width_atr=float(self.compression_max_width_atr),squeeze_active_width_atr=float(self.squeeze_active_width_atr),local_narrowing_min_ratio=float(self.local_narrowing_min_ratio),min_line_score=float(self.min_line_score),min_containment=float(self.min_containment),max_recent_touch_age_bars=int(self.max_recent_touch_age_bars),channel_min_pattern_bars=int(self.channel_min_pattern_bars),channel_max_pattern_bars=int(self.channel_max_pattern_bars),channel_min_quality=float(self.channel_min_quality),channel_min_containment=float(self.channel_min_containment),channel_near_boundary_atr_mult=float(self.channel_near_boundary_atr_mult),channel_breakout_atr_mult=float(self.channel_breakout_atr_mult),channel_lifecycle_confirm_break_bars=int(self.channel_lifecycle_confirm_break_bars),output_prefix="pg2");return dataframe
    def _common_guards(self,d:DataFrame)->Series:
        g=pd.Series(True,index=d.index,dtype="bool");c=_num(d,"close")
        
        o,h,l=_num(d,"open"),_num(d,"high"),_num(d,"low");v=_num(d,"volume").clip(lower=0.).fillna(0.);r=(h-l).replace(0.,np.nan);p=((c-o)/r).clip(-1.,1.).fillna(0.).add((((c-l)/r)*2.-1.).clip(-1.,1.).fillna(0.)).div(2.);dv=(p*v).fillna(0.)
        w=int(self.pressure_window);g&=(dv.rolling(w,min_periods=max(2,w//3)).sum()/v.rolling(w,min_periods=max(2,w//3)).sum().replace(0.,np.nan)).le(-float(self.pressure_min))
        
        
        
        return g.fillna(False)
    def populate_entry_trend(self,dataframe:DataFrame,metadata:dict)->DataFrame:
        dataframe["enter_long"],dataframe["enter_short"],dataframe["enter_tag"]=0,0,None;l=_num(dataframe,"pg2_rectangle_lower").mul(1.-float(self.rail_buffer_pct));c=_bool(dataframe,"pg2_rectangle_pattern_present")&_num(dataframe,"pg2_rectangle_indicator_score").ge(float(self.score_min))&_num(dataframe,"pg2_rectangle_width_atr").le(float(self.width_atr_max))&_cross_below(_num(dataframe,"close"),l);c&=self._common_guards(dataframe);v=c.fillna(False)&dataframe["volume"].gt(0.)&dataframe["close"].notna();dataframe.loc[v,"enter_short"],dataframe.loc[v,"enter_tag"]=1,ENTRY_TAG;return dataframe
    def populate_exit_trend(self,dataframe:DataFrame,metadata:dict)->DataFrame:dataframe["exit_long"],dataframe["exit_short"],dataframe["exit_tag"]=0,0,None;return dataframe
    @staticmethod
    def _utc(v:Any)->pd.Timestamp:
        t=pd.Timestamp(v);return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    def _closed(self,pair:str,when:Any)->DataFrame:
        if self.dp is None:raise RuntimeError("measured target requires data provider")
        f,_=self.dp.get_analyzed_dataframe(pair,self.timeframe);needed={"date","pg2_rectangle_upper","pg2_rectangle_lower"};missing=sorted(needed-set(f.columns))
        if missing:raise KeyError(f"measured target missing columns: {missing}")
        closes=pd.to_datetime(f["date"],utc=True,errors="raise")+pd.to_timedelta(timeframe_to_minutes(self.timeframe),unit="m");return f.loc[closes.le(self._utc(when))].sort_values("date")
    def _target(self,pair:str,trade:Any)->float:
        f=self._closed(pair,trade.open_date_utc)
        if f.empty:raise RuntimeError("no closed entry-signal candle")
        row=f.iloc[-1];upper,lower=float(row["pg2_rectangle_upper"]),float(row["pg2_rectangle_lower"])
        if not math.isfinite(upper+lower) or not 0.<lower<upper:raise ValueError("invalid entry rectangle rails")
        multiple,_=TARGET_PLANS[str(self.target_plan.value)];return lower-(upper-lower)*multiple
    def custom_exit(self,pair:str,trade:Any,current_time:datetime,current_rate:float,current_profit:float,**kwargs:Any)->str|None:
        target=self._target(pair,trade);_,band=TARGET_PLANS[str(self.target_plan.value)];return f"measured_target_{self.target_plan.value}" if current_rate<=target*(1.+band) else None
ACTIVE_SELL_PARAMS=("target_plan",)
