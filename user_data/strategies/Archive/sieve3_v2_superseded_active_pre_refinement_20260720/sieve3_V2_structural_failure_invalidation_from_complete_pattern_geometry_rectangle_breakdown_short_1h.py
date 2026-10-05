"""Rectangle-top or entry-signal-swing structural invalidation for the frozen 1h short."""
from __future__ import annotations
from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd
from pandas import DataFrame,Series
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter, IStrategy
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2

ENTRY_MODE,ENTRY_TAG,SIDE,TIMEFRAME="entry_geometry_rectangle_breakdown_short_1h","geometry_rectangle_breakdown_short_1h","short","1h"
SIEVE_STAGE="sieve3"
ENTRY_LOCK_STATUS="current executable defaults locked; promoted historical snapshots conflict"
STRUCTURE_PLANS=("one_close_above_rectangle_top","two_closes_above_rectangle_top","one_close_above_signal_high","two_closes_above_signal_high","wick_above_signal_high_bullish_close")
def tagged_exit_parameter(p:Any)->Any:setattr(p,"batch_tags",("family:exits","mode:sieve3_exit"));return p
def _num(f: DataFrame, c: str, _default: float | Series | object = ...) -> Series:
    if c not in f.columns:
        if _default is ...:
            raise KeyError(f"required numeric column is missing: {c!r}")
        if isinstance(_default, Series):
            return pd.to_numeric(_default, errors="coerce").reindex(f.index)
        return pd.Series(float(_default), index=f.index, dtype="float64")
    return pd.to_numeric(f[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
def _bool(f: DataFrame, c: str) -> Series:
    return pd.Series(f[c], index=f.index).astype("boolean").fillna(False).astype(bool)
def _cross_below(v:Series,l:Series)->Series:return v.lt(l)&v.shift(1).ge(l.shift(1))
EXIT_FAMILY = 'structural_failure_invalidation'


ENTRY_SOURCE_STAGE = 'sieve2'
RESEARCH_PATH = 'sieve3_exit_structural_failure_invalidation'
SOURCE_RESULT_BATCH = 'historical_promotion_lineage_unknown'
SOURCE_STRATEGY = 'sieve2_complete_pattern_geometry_rectangle_breakdown_short_1h'
EXIT_HYPOTHESIS = 'A confirmed close or bullish wick above the entry-frozen rectangle top or entry-signal swing high invalidates the rectangle-breakdown short.'


class Sieve3V2StructuralFailureInvalidationFromCompletePatternGeometryRectangleBreakdownShort1H(IStrategy):
    ACTIVE_SELL_PARAMS = ('structure_plan',)
    INTERFACE_VERSION,timeframe,startup_candle_count,process_only_new_candles,can_short=3,TIMEFRAME,180,True,True
    minimal_roi,stoploss,use_exit_signal,use_custom_stoploss={"0":100.},-.03,True,False
    position_adjustment_enable,trailing_stop,exit_profit_only,ignore_roi_if_entry_signal=False,False,False,False
    
    
    use_volume_guard=False;volume_guard_window=48;volume_ratio_min=1.0;use_pressure_guard=True;pressure_window=24;pressure_min=0.15;use_accumulation_guard=False;use_body_direction_guard=False;use_close_direction_guard=False
    min_pattern_bars=12;max_pattern_bars=72;compression_max_width_atr=2.0;squeeze_active_width_atr=2.0;local_narrowing_min_ratio=0.1;min_line_score=0.7;min_containment=0.88;max_recent_touch_age_bars=12;channel_min_pattern_bars=18;channel_max_pattern_bars=96;channel_min_quality=0.82;channel_min_containment=0.68;channel_near_boundary_atr_mult=0.7;channel_breakout_atr_mult=0.35;channel_lifecycle_confirm_break_bars=2;score_min=0.65;width_atr_max=6.0;rail_buffer_pct=0.004
    structure_plan=tagged_exit_parameter(CategoricalParameter(STRUCTURE_PLANS,default="one_close_above_rectangle_top",space="sell",optimize=True,load=True))
    def leverage(self,pair:str,current_time:datetime,current_rate:float,proposed_leverage:float,max_leverage:float,entry_tag:str|None,side:str,**kwargs:Any)->float:return 1.
    def informative_pairs(self)->list[tuple[str,str]]:return[]
    def populate_indicators(self,dataframe:DataFrame,metadata:dict)->DataFrame:
        dataframe=add_pattern_geometry_v2(dataframe,timeframe=self.timeframe,output_slots=1,include_triangle_patterns=False,include_wedge_patterns=False,include_compression_patterns=False,include_rectangle_patterns=True,include_ascending_channel_patterns=False,include_descending_channel_patterns=False,min_pattern_bars=int(self.min_pattern_bars),max_pattern_bars=int(self.max_pattern_bars),compression_max_width_atr=float(self.compression_max_width_atr),squeeze_active_width_atr=float(self.squeeze_active_width_atr),local_narrowing_min_ratio=float(self.local_narrowing_min_ratio),min_line_score=float(self.min_line_score),min_containment=float(self.min_containment),max_recent_touch_age_bars=int(self.max_recent_touch_age_bars),channel_min_pattern_bars=int(self.channel_min_pattern_bars),channel_max_pattern_bars=int(self.channel_max_pattern_bars),channel_min_quality=float(self.channel_min_quality),channel_min_containment=float(self.channel_min_containment),channel_near_boundary_atr_mult=float(self.channel_near_boundary_atr_mult),channel_breakout_atr_mult=float(self.channel_breakout_atr_mult),channel_lifecycle_confirm_break_bars=int(self.channel_lifecycle_confirm_break_bars),output_prefix="pg2");return dataframe
    def _common_guards(self,d:DataFrame)->Series:
        g=pd.Series(True,index=d.index,dtype="bool");c=_num(d,"close")
        if bool(self.use_volume_guard):v=_num(d,"volume").clip(lower=0.);w=int(self.volume_guard_window);g&=v.ge(v.shift(1).rolling(w,min_periods=max(2,w//3)).mean().replace(0.,np.nan).mul(float(self.volume_ratio_min)))
        if bool(self.use_pressure_guard) or bool(self.use_accumulation_guard):o,h,l=_num(d,"open"),_num(d,"high"),_num(d,"low");v=_num(d,"volume").clip(lower=0.).fillna(0.);r=(h-l).replace(0.,np.nan);p=((c-o)/r).clip(-1.,1.).fillna(0.).add((((c-l)/r)*2.-1.).clip(-1.,1.).fillna(0.)).div(2.);dv=(p*v).fillna(0.)
        if bool(self.use_pressure_guard):w=int(self.pressure_window);g&=(dv.rolling(w,min_periods=max(2,w//3)).sum()/v.rolling(w,min_periods=max(2,w//3)).sum().replace(0.,np.nan)).le(-float(self.pressure_min))
        if bool(self.use_accumulation_guard):g&=dv.rolling(int(self.pressure_window),min_periods=max(2,int(self.pressure_window)//3)).sum().le(0.)
        if bool(self.use_body_direction_guard):g&=c.lt(_num(d,"open"))
        if bool(self.use_close_direction_guard):g&=c.lt(c.shift(1))
        return g.fillna(False)
    def populate_entry_trend(self,dataframe:DataFrame,metadata:dict)->DataFrame:
        dataframe["enter_long"],dataframe["enter_short"],dataframe["enter_tag"]=0,0,None;l=_num(dataframe,"pg2_rectangle_lower").mul(1.-float(self.rail_buffer_pct));c=_bool(dataframe,"pg2_rectangle_pattern_present")&_num(dataframe,"pg2_rectangle_indicator_score").ge(float(self.score_min))&_num(dataframe,"pg2_rectangle_width_atr").le(float(self.width_atr_max))&_cross_below(_num(dataframe,"close"),l);c&=self._common_guards(dataframe);c=pd.Series(c, index=dataframe.index).fillna(False).astype(bool).fillna(False);v=c.fillna(False)&dataframe["volume"].gt(0.)&dataframe["close"].notna();dataframe.loc[v,"enter_short"],dataframe.loc[v,"enter_tag"]=1,ENTRY_TAG;return dataframe
    def populate_exit_trend(self,dataframe:DataFrame,metadata:dict)->DataFrame:dataframe["exit_long"],dataframe["exit_short"],dataframe["exit_tag"]=0,0,None;return dataframe
    @staticmethod
    def _utc(v:Any)->pd.Timestamp:t=pd.Timestamp(v);return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    def _closed(self,pair:str,when:Any)->DataFrame:
        if self.dp is None:raise RuntimeError("structural invalidation requires data provider")
        f,_=self.dp.get_analyzed_dataframe(pair,self.timeframe);needed={"date","open","high","close","pg2_rectangle_upper"};missing=sorted(needed-set(f.columns))
        if missing:raise KeyError(f"structural invalidation missing columns: {missing}")
        closes=pd.to_datetime(f["date"],utc=True,errors="raise")+pd.to_timedelta(timeframe_to_minutes(self.timeframe),unit="m");return f.loc[closes.le(self._utc(when))].sort_values("date")
    def _entry_levels(self,pair:str,trade:Any)->tuple[float,float]:
        f=self._closed(pair,trade.open_date_utc)
        if f.empty:raise RuntimeError("no closed entry-signal candle")
        r=f.iloc[-1];return float(r["pg2_rectangle_upper"]),float(r["high"])
    def custom_exit(self,pair:str,trade:Any,current_time:datetime,current_rate:float,current_profit:float,**kwargs:Any)->str|None:
        f=self._closed(pair,current_time);plan=str(self.structure_plan.value)
        if f.empty:return None
        top,swing=self._entry_levels(pair,trade);level=top if "rectangle_top" in plan else swing;latest=f.iloc[-1];one=float(latest["close"])>level;two=len(f)>=2 and bool(_num(f.tail(2),"close").gt(level).all());wick=float(latest["high"])>level and float(latest["close"])>float(latest["open"]);failed=two if plan.startswith("two_") else wick if plan.startswith("wick_") else one
        return f"structural_failure_{plan}" if failed else None
ACTIVE_SELL_PARAMS=("structure_plan",)
