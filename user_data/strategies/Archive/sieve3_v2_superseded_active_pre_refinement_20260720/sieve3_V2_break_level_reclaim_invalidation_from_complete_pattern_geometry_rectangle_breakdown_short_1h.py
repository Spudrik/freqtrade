"""Break-level reclaim invalidation for the frozen 1h rectangle-breakdown short."""
from __future__ import annotations
from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd
from pandas import DataFrame,Series
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter,IStrategy
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2
ENTRY_MODE,ENTRY_TAG,SIDE,TIMEFRAME="entry_geometry_rectangle_breakdown_short_1h","geometry_rectangle_breakdown_short_1h","short","1h"
ENTRY_LOCK_STATUS="current executable defaults locked; promoted historical snapshots conflict"
RECLAIM_PLANS=("one_close_above_break_level","one_close_above_break_level_0_25","two_closes_above_break_level","bullish_close_above_break_level")
def tagged_exit_parameter(p:Any)->Any:setattr(p,"batch_tags",("family:exits","mode:sieve3_exit"));return p
def _num(f: DataFrame, c: str, _default: float | Series=...) -> Series:
    if c in f.columns:
        value = f[c]
    elif _default is not ...:
        value = pd.Series(_default, index=f.index)
    else:
        raise KeyError(c)
    return pd.to_numeric(value, errors='coerce').replace([np.inf, -np.inf], np.nan)
def _bool(f:DataFrame,c:str)->Series:return pd.Series(f[c],index=f.index).astype("boolean").fillna(False).astype(bool)
def _cross_below(v:Series,l:Series)->Series:return v.lt(l)&v.shift(1).ge(l.shift(1))
SIEVE_STAGE = 'sieve3'
ENTRY_SOURCE_STAGE = 'sieve2'
RESEARCH_PATH = 'sieve3_exit_break_level_reclaim_invalidation'
EXIT_HYPOTHESIS = 'Exit the short when price reclaims the broken rectangle level, comparing confirmation plans.'
class Sieve3V2BreakLevelReclaimInvalidationFromCompletePatternGeometryRectangleBreakdownShort1H(IStrategy):
    SIEVE_STAGE = 'sieve3'
    ENTRY_SOURCE_STAGE = 'sieve2'
    RESEARCH_PATH = 'sieve3_exit_break_level_reclaim_invalidation'
    EXIT_HYPOTHESIS = 'Exit the short when price reclaims the broken rectangle level, comparing confirmation plans.'
    ACTIVE_SELL_PARAMS = ('reclaim_plan',)
    INTERFACE_VERSION,timeframe,startup_candle_count,process_only_new_candles,can_short=3,TIMEFRAME,180,True,True
    minimal_roi,stoploss,use_exit_signal,use_custom_stoploss={"0":100.},-.03,True,False
    position_adjustment_enable,trailing_stop,exit_profit_only,ignore_roi_if_entry_signal=False,False,False,False
    
    
    # use_volume_guard = False (locked off; gated entry branch removed).
    # volume_guard_window = 48 (inactive after its locked entry gate/mode was removed).
    # volume_ratio_min = 1.0 (inactive after its locked entry gate/mode was removed).
    use_pressure_guard = True
    pressure_window = 24
    pressure_min = 0.15
    # use_accumulation_guard = False (locked off; gated entry branch removed).
    # use_body_direction_guard = False (locked off; gated entry branch removed).
    # use_close_direction_guard = False (locked off; gated entry branch removed).
    min_pattern_bars = 12;max_pattern_bars = 72;compression_max_width_atr = 2.;squeeze_active_width_atr = 2.;local_narrowing_min_ratio = .1;min_line_score = .7;min_containment = .88;max_recent_touch_age_bars = 12;channel_min_pattern_bars = 18;channel_max_pattern_bars = 96;channel_min_quality = .82;channel_min_containment = .68;channel_near_boundary_atr_mult = .7;channel_breakout_atr_mult = .35;channel_lifecycle_confirm_break_bars = 2;score_min = .65;width_atr_max = 6.;rail_buffer_pct = .004
    reclaim_plan=tagged_exit_parameter(CategoricalParameter(RECLAIM_PLANS,default="one_close_above_break_level",space="sell",optimize=True,load=True))
    def leverage(self,pair:str,current_time:datetime,current_rate:float,proposed_leverage:float,max_leverage:float,entry_tag:str|None,side:str,**kwargs:Any)->float:return 1.
    def informative_pairs(self)->list[tuple[str,str]]:return[]
    def populate_indicators(self,dataframe:DataFrame,metadata:dict)->DataFrame:
        dataframe=add_pattern_geometry_v2(dataframe,timeframe=self.timeframe,output_slots=1,include_triangle_patterns=False,include_wedge_patterns=False,include_compression_patterns=False,include_rectangle_patterns=True,include_ascending_channel_patterns=False,include_descending_channel_patterns=False,min_pattern_bars=int(self.min_pattern_bars),max_pattern_bars=int(self.max_pattern_bars),compression_max_width_atr=float(self.compression_max_width_atr),squeeze_active_width_atr=float(self.squeeze_active_width_atr),local_narrowing_min_ratio=float(self.local_narrowing_min_ratio),min_line_score=float(self.min_line_score),min_containment=float(self.min_containment),max_recent_touch_age_bars=int(self.max_recent_touch_age_bars),channel_min_pattern_bars=int(self.channel_min_pattern_bars),channel_max_pattern_bars=int(self.channel_max_pattern_bars),channel_min_quality=float(self.channel_min_quality),channel_min_containment=float(self.channel_min_containment),channel_near_boundary_atr_mult=float(self.channel_near_boundary_atr_mult),channel_breakout_atr_mult=float(self.channel_breakout_atr_mult),channel_lifecycle_confirm_break_bars=int(self.channel_lifecycle_confirm_break_bars),output_prefix="pg2");return dataframe
    def _common_guards(self,d:DataFrame)->Series:
        g=pd.Series(True,index=d.index,dtype="bool");c=_num(d,"close")
        if bool(self.use_pressure_guard):o,h,l=_num(d,"open"),_num(d,"high"),_num(d,"low");v=_num(d,"volume").clip(lower=0.).fillna(0.);r=(h-l).replace(0.,np.nan);p=((c-o)/r).clip(-1.,1.).fillna(0.).add((((c-l)/r)*2.-1.).clip(-1.,1.).fillna(0.)).div(2.);dv=(p*v).fillna(0.)
        if bool(self.use_pressure_guard):w=int(self.pressure_window);g&=(dv.rolling(w,min_periods=max(2,w//3)).sum()/v.rolling(w,min_periods=max(2,w//3)).sum().replace(0.,np.nan)).le(-float(self.pressure_min))
        return g.fillna(False)
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        missing = sorted(set(('pg2_rectangle_lower', 'pg2_rectangle_width_atr')).difference(dataframe.columns))
        if missing:
            raise KeyError(f'{type(self).__name__} entry dataframe is missing required source columns: {missing}')
        dataframe['enter_long'], dataframe['enter_short'], dataframe['enter_tag'] = (0, 0, None)
        l = _num(dataframe, 'pg2_rectangle_lower', np.nan).mul(1.0 - float(self.rail_buffer_pct))
        c = _bool(dataframe, 'pg2_rectangle_pattern_present') & _num(dataframe, 'pg2_rectangle_indicator_score').ge(float(self.score_min)) & _num(dataframe, 'pg2_rectangle_width_atr', np.nan).le(float(self.width_atr_max)) & _cross_below(_num(dataframe, 'close'), l)
        c &= self._common_guards(dataframe)
        v = c.fillna(False) & dataframe['volume'].gt(0.0) & dataframe['close'].notna()
        dataframe.loc[v, 'enter_short'], dataframe.loc[v, 'enter_tag'] = (1, ENTRY_TAG)
        return dataframe
    def populate_exit_trend(self,dataframe:DataFrame,metadata:dict)->DataFrame:dataframe["exit_long"],dataframe["exit_short"],dataframe["exit_tag"]=0,0,None;return dataframe
    @staticmethod
    def _utc(v:Any)->pd.Timestamp:t=pd.Timestamp(v);return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    def _closed(self,pair:str,when:Any)->DataFrame:
        if self.dp is None:raise RuntimeError("break reclaim requires data provider")
        f,_=self.dp.get_analyzed_dataframe(pair,self.timeframe);needed={"date","open","close","pg2_rectangle_lower"};missing=sorted(needed-set(f.columns))
        if missing:raise KeyError(f"break reclaim missing columns: {missing}")
        closes=pd.to_datetime(f["date"],utc=True,errors="raise")+pd.to_timedelta(timeframe_to_minutes(self.timeframe),unit="m");return f.loc[closes.le(self._utc(when))].sort_values("date")
    def _break_level(self,pair:str,trade:Any)->float:
        f=self._closed(pair,trade.open_date_utc)
        if f.empty:raise RuntimeError("no closed entry-signal candle")
        return float(f.iloc[-1]["pg2_rectangle_lower"])*(1.-float(self.rail_buffer_pct))
    def custom_exit(self,pair:str,trade:Any,current_time:datetime,current_rate:float,current_profit:float,**kwargs:Any)->str|None:
        f=self._closed(pair,current_time);plan=str(self.reclaim_plan.value)
        if f.empty:return None
        level=self._break_level(pair,trade)*(1.0025 if plan.endswith("_0_25") else 1.);latest=f.iloc[-1];one=float(latest["close"])>level;two=len(f)>=2 and bool(_num(f.tail(2),"close").gt(level).all());bullish=one and float(latest["close"])>float(latest["open"]);failed=two if plan.startswith("two_") else bullish if plan.startswith("bullish_") else one
        return f"break_level_reclaim_{plan}" if failed else None
ACTIVE_SELL_PARAMS = ('reclaim_plan',)
