"""Profit-armed bullish weakening/reversal exits for the frozen 1h rectangle short."""
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
ENTRY_GEOMETRY_STATE_KEY="s3v2_rectangle_short_entry_geometry"
ENTRY_LOCK_STATUS="current executable defaults locked; promoted historical snapshots conflict"
REVERSAL_PLANS=("profit_1_bullish_close_above_prior_high","profit_2_two_bullish_closes","profit_2_break_level_reclaim","profit_3_bullish_engulfing")
def tagged_exit_parameter(p:Any)->Any:setattr(p,"batch_tags",("family:exits","mode:sieve3_exit"));return p
def _num(f: DataFrame, c: str, d: float | Series=...) -> Series:
    if c not in f.columns:
        if d is ...:
            raise KeyError(f'missing required column: {c!r}')
        return pd.to_numeric(
            pd.Series(d, index=f.index), errors='coerce'
        ).replace([np.inf, -np.inf], np.nan)
    return pd.to_numeric(f[c], errors='coerce').replace([np.inf, -np.inf], np.nan)
def _bool(f: DataFrame, c: str) -> Series:
    if c not in f.columns:
        raise KeyError(f'missing required column: {c!r}')
    return pd.Series(f[c], index=f.index).astype('boolean').fillna(False).astype(bool)
def _cross_below(v:Series,l:Series)->Series:return v.lt(l)&v.shift(1).ge(l.shift(1))
EXIT_FAMILY = 'profit_armed_bullish_reversal'
PRIMARY_TRIGGER = '1h close cross below the rectangle lower rail with locked score and width qualification'
PRIMARY_GUARD = 'locked bearish pressure guard'
PRIMARY_TARGET = 'selected open-profit gate'
PRIMARY_INVALIDATION = 'bullish close, two-close, engulfing, prior-high, or broken-level reclaim evidence'
TARGET_PROVIDER = PRIMARY_TARGET
INVALIDATION_PROVIDER = PRIMARY_INVALIDATION
ENTRY_SOURCE_STAGE = 'sieve2'
SOURCE_STRATEGY = 'filename-derived source foundation: complete_pattern_geometry_rectangle_breakdown_short_1h'
SOURCE_RESULT_BATCH = 'historical_promotion_lineage_unknown'
RESEARCH_PATH = 'sieve3_exit_profit_armed_bullish_reversal'
EXIT_HYPOTHESIS = 'After profit is armed, a selected bullish reversal or rectangle-break reclaim should close the 1h rectangle short.'
ACTIVE_SELL_PARAMS = ('reversal_plan',)
SIEVE_STAGE = 'sieve3'
SIDE = 'short'
ENTRY_TAG = 'geometry_rectangle_breakdown_short_1h'


class Sieve3V2ProfitArmedBullishReversalFromCompletePatternGeometryRectangleBreakdownShort1H(IStrategy):
    INTERFACE_VERSION,timeframe,startup_candle_count,process_only_new_candles,can_short=3,TIMEFRAME,180,True,True
    minimal_roi,stoploss,use_exit_signal,use_custom_stoploss={"0":100.},-.03,True,False
    position_adjustment_enable,trailing_stop,exit_profit_only,ignore_roi_if_entry_signal=False,False,False,False
    
    
    # Legacy fixed-off entry parameter: use_volume_guard=False (BooleanParameter, buy space).
    volume_guard_window = 48
    volume_ratio_min = 1.0
    use_pressure_guard = True
    pressure_window = 24
    pressure_min = 0.15
    # Legacy fixed-off entry parameter: use_accumulation_guard=False (BooleanParameter, buy space).
    # Legacy fixed-off entry parameter: use_body_direction_guard=False (BooleanParameter, buy space).
    # Legacy fixed-off entry parameter: use_close_direction_guard=False (BooleanParameter, buy space).
    min_pattern_bars = 12
    max_pattern_bars = 72
    compression_max_width_atr = 2.0
    squeeze_active_width_atr = 2.0
    local_narrowing_min_ratio = 0.1
    min_line_score = 0.7
    min_containment = 0.88
    max_recent_touch_age_bars = 12
    channel_min_pattern_bars = 18
    channel_max_pattern_bars = 96
    channel_min_quality = 0.82
    channel_min_containment = 0.68
    channel_near_boundary_atr_mult = 0.7
    channel_breakout_atr_mult = 0.35
    channel_lifecycle_confirm_break_bars = 2
    score_min = 0.65
    width_atr_max = 6.0
    rail_buffer_pct = 0.004
    reversal_plan=tagged_exit_parameter(CategoricalParameter(REVERSAL_PLANS,default="profit_2_break_level_reclaim",space="sell",optimize=True,load=True))
    ACTIVE_SELL_PARAMS = ('reversal_plan',)
    def leverage(self,pair:str,current_time:datetime,current_rate:float,proposed_leverage:float,max_leverage:float,entry_tag:str|None,side:str,**kwargs:Any)->float:return 1.
    def informative_pairs(self)->list[tuple[str,str]]:return[]
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe = add_pattern_geometry_v2(dataframe, timeframe=self.timeframe, output_slots=1, include_triangle_patterns=False, include_wedge_patterns=False, include_compression_patterns=False, include_rectangle_patterns=True, include_ascending_channel_patterns=False, include_descending_channel_patterns=False, min_pattern_bars=int(self.min_pattern_bars), max_pattern_bars=int(self.max_pattern_bars), compression_max_width_atr=float(self.compression_max_width_atr), squeeze_active_width_atr=float(self.squeeze_active_width_atr), local_narrowing_min_ratio=float(self.local_narrowing_min_ratio), min_line_score=float(self.min_line_score), min_containment=float(self.min_containment), max_recent_touch_age_bars=int(self.max_recent_touch_age_bars), channel_min_pattern_bars=int(self.channel_min_pattern_bars), channel_max_pattern_bars=int(self.channel_max_pattern_bars), channel_min_quality=float(self.channel_min_quality), channel_min_containment=float(self.channel_min_containment), channel_near_boundary_atr_mult=float(self.channel_near_boundary_atr_mult), channel_breakout_atr_mult=float(self.channel_breakout_atr_mult), channel_lifecycle_confirm_break_bars=int(self.channel_lifecycle_confirm_break_bars), output_prefix='pg2')
        return dataframe
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
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'], dataframe['exit_short'], dataframe['exit_tag'] = (0, 0, None)
        return dataframe
    @staticmethod
    def _utc(v:Any)->pd.Timestamp:t=pd.Timestamp(v);return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    def _closed(self,pair:str,when:Any)->DataFrame:
        if self.dp is None:raise RuntimeError("bullish reversal exit requires data provider")
        f,_=self.dp.get_analyzed_dataframe(pair,self.timeframe);needed={"date","open","high","close","pg2_rectangle_lower"};missing=sorted(needed-set(f.columns))
        if missing:raise KeyError(f"bullish reversal missing columns: {missing}")
        closes=pd.to_datetime(f["date"],utc=True,errors="raise")+pd.to_timedelta(timeframe_to_minutes(self.timeframe),unit="m");return f.loc[closes.le(self._utc(when))].sort_values("date")
    def _freeze_break_level(self,pair:str,trade:Any,cutoff:Any)->float:
        f=self._closed(pair,cutoff)
        if f.empty:raise RuntimeError("no closed entry-signal candle")
        lower=float(f.iloc[-1]["pg2_rectangle_lower"])
        if not np.isfinite(lower) or lower<=0.0:raise RuntimeError("entry signal has no usable rectangle lower rail")
        break_level=lower*(1.-float(self.rail_buffer_pct));trade.set_custom_data(key=ENTRY_GEOMETRY_STATE_KEY,value={"lower":lower,"break_level":break_level});return break_level
    def _break_level(self,pair:str,trade:Any)->float:
        state=trade.get_custom_data(key=ENTRY_GEOMETRY_STATE_KEY)
        if state is None:return self._freeze_break_level(pair,trade,getattr(trade,"date_entry_fill_utc",None) or trade.open_date_utc)
        if not isinstance(state,dict):raise RuntimeError("rectangle entry geometry snapshot is invalid")
        return float(state["break_level"])
    def order_filled(self,pair:str,trade:Any,order:Any,current_time:datetime,**kwargs:Any)->None:
        if getattr(order,"ft_order_side",None)==getattr(trade,"entry_side",None) and trade.get_custom_data(key=ENTRY_GEOMETRY_STATE_KEY) is None:self._freeze_break_level(pair,trade,current_time)
    def custom_exit(self,pair:str,trade:Any,current_time:datetime,current_rate:float,current_profit:float,**kwargs:Any)->str|None:
        plan=str(self.reversal_plan.value);gate={"profit_1":.01,"profit_2":.02,"profit_3":.03}[plan[:8]]
        if current_profit<gate:return None
        f=self._closed(pair,current_time)
        dates=pd.to_datetime(f["date"],utc=True,errors="raise");entered=self._utc(getattr(trade,"date_entry_fill_utc",None) or trade.open_date_utc);f=f.loc[dates.ge(entered)]
        if len(f)<2:return None
        prev,latest=f.iloc[-2],f.iloc[-1];bullish=float(latest["close"])>float(latest["open"]);above_high=bullish and float(latest["close"])>float(prev["high"]);two_bullish=bullish and float(prev["close"])>float(prev["open"]);reclaim=float(latest["close"])>self._break_level(pair,trade);engulf=bullish and float(latest["open"])<=float(prev["close"]) and float(latest["close"])>=float(prev["open"]);signal=above_high if "prior_high" in plan else two_bullish if "two_bullish" in plan else reclaim if "reclaim" in plan else engulf
        return f"profit_armed_reversal_{plan}" if signal else None
