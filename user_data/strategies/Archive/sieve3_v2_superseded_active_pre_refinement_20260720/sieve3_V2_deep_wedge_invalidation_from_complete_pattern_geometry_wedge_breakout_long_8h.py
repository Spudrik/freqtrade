"""Deep structural invalidations at the lower wedge rail or signal-candle swing low."""
from __future__ import annotations
from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter, IStrategy, stoploss_from_absolute
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2
ENTRY_TAG="geometry_wedge_breakout_long_8h";STATE_KEY="s3v2_deep_wedge_invalidation";SIEVE_STAGE = 'sieve3';ENTRY_SOURCE_STAGE = 'sieve2';SOURCE_STRATEGY="Sieve2GeometryWedgeBreakoutLong8H tested snapshot 2026-05-23";SOURCE_RESULT_BATCH="20260521T012740_entry_all_resume line 246";RESEARCH_PATH = 'sieve3_exit_deep_wedge_invalidation';EXIT_HYPOTHESIS = 'A lower-rail or signal-swing failure breaks deeper wedge structure and is distinct from a simple lost breakout.'
LOCKED_BUY_PARAMS={'use_volume_guard': True, 'volume_guard_window': 12, 'volume_ratio_min': 1.6, 'use_pressure_guard': True, 'pressure_window': 24, 'pressure_min': 0.1, 'use_accumulation_guard': False, 'use_body_direction_guard': False, 'use_close_direction_guard': False, 'min_pattern_bars': 12, 'max_pattern_bars': 72, 'compression_max_width_atr': 2.0, 'squeeze_active_width_atr': 2.0, 'min_line_score': 0.4, 'min_containment': 0.88}
def _num(f: DataFrame, c: str, _default: float | Series=...) -> Series:
    if c in f.columns:
        value = f[c]
    elif _default is not ...:
        value = pd.Series(_default, index=f.index)
    else:
        raise KeyError(c)
    return pd.to_numeric(value, errors='coerce').replace([np.inf, -np.inf], np.nan)
def _bool(f:DataFrame,c:str)->Series:
    return pd.Series(f[c],index=f.index).astype("boolean").fillna(False).astype(bool)
ACTIVE_SELL_PARAMS = ('exit_plan',)
class Sieve3V2DeepWedgeInvalidationFromCompletePatternGeometryWedgeBreakoutLong8H(IStrategy):
    SIEVE_STAGE = 'sieve3'
    ENTRY_SOURCE_STAGE = 'sieve2'
    RESEARCH_PATH = 'sieve3_exit_deep_wedge_invalidation'
    EXIT_HYPOTHESIS = 'A lower-rail or signal-swing failure breaks deeper wedge structure and is distinct from a simple lost breakout.'
    ACTIVE_SELL_PARAMS = ('exit_plan',)
    INTERFACE_VERSION=3;timeframe="8h";startup_candle_count=180;process_only_new_candles=True;can_short=False;minimal_roi={"0":100.0};stoploss=-0.99;use_exit_signal=True;use_custom_stoploss=True;trailing_stop=False;position_adjustment_enable=False
    
    
    # Inactive buy declaration: use_volume_guard = BooleanParameter(default=True, space="buy"); fixed-on entry gate makes its guard unconditional.
    # Inactive buy declaration: use_pressure_guard = BooleanParameter(default=True, space="buy"); fixed-on entry gate makes its guard unconditional.
    # Inactive buy declaration: use_accumulation_guard = BooleanParameter(default=False, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: use_body_direction_guard = BooleanParameter(default=False, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: use_close_direction_guard = BooleanParameter(default=False, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    volume_guard_window = 12; volume_ratio_min = 1.6; pressure_window = 24; pressure_min = 0.1
    min_pattern_bars = 12; max_pattern_bars = 72; compression_max_width_atr = 2.0; squeeze_active_width_atr = 2.0; min_line_score = 0.4; min_containment = 0.88
    exit_plan=CategoricalParameter(["lower_rail_close","signal_swing_close","signal_swing_low"],default="signal_swing_close",space="sell",optimize=True,load=True)
    exit_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    def leverage(self,pair:str,current_time:datetime,current_rate:float,proposed_leverage:float,max_leverage:float,entry_tag:str|None,side:str,**kwargs:Any)->float:return 1.0
    def informative_pairs(self)->list[tuple[str,str]]:return []
    def populate_indicators(self,dataframe:DataFrame,metadata:dict)->DataFrame:
        dataframe=add_pattern_geometry_v2(dataframe,timeframe=self.timeframe,output_slots=1,include_triangle_patterns=False,include_wedge_patterns=True,include_compression_patterns=False,include_rectangle_patterns=False,include_ascending_channel_patterns=False,include_descending_channel_patterns=False,min_pattern_bars=int(self.min_pattern_bars),max_pattern_bars=int(self.max_pattern_bars),compression_max_width_atr=float(self.compression_max_width_atr),squeeze_active_width_atr=float(self.squeeze_active_width_atr),min_line_score=float(self.min_line_score),min_containment=float(self.min_containment),output_prefix="pg2");return dataframe
    def _common_guards(self,f:DataFrame)->Series:
        g=pd.Series(True,index=f.index,dtype="bool");close=_num(f,"close")
        volume=_num(f,"volume").clip(lower=0);w=int(self.volume_guard_window);g&=volume.ge(volume.shift(1).rolling(w,min_periods=max(2,w//3)).mean().replace(0,np.nan).mul(float(self.volume_ratio_min)))
        open_=_num(f,"open");high=_num(f,"high");low=_num(f,"low");volume=_num(f,"volume").clip(lower=0).fillna(0);r=(high-low).replace(0,np.nan);p=((((close-open_)/r).clip(-1,1).fillna(0)+(((close-low)/r)*2-1).clip(-1,1).fillna(0))/2).clip(-1,1);dv=(p*volume).fillna(0)
        w=int(self.pressure_window);base=volume.rolling(w,min_periods=max(2,w//3)).sum().replace(0,np.nan);g&=dv.rolling(w,min_periods=max(2,w//3)).sum().div(base).ge(float(self.pressure_min))
        
        
        
        return g.fillna(False)
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        missing = sorted(set(('pg2_wedge_upper', 'pg2_wedge_lower')).difference(dataframe.columns))
        if missing:
            raise KeyError(f'{type(self).__name__} entry dataframe is missing required source columns: {missing}')
        dataframe['enter_long'] = 0
        dataframe['enter_short'] = 0
        dataframe['enter_tag'] = None
        lower = _num(dataframe, 'pg2_wedge_lower', np.nan)
        c = _bool(dataframe, 'pg2_wedge_pattern_present') & _bool(dataframe, 'pg2_wedge_squeeze_active') & _num(dataframe, 'pg2_wedge_indicator_score').ge(float(self.min_line_score)) & _num(dataframe, 'pg2_wedge_direction').ge(0) & _num(dataframe, 'close').gt(_num(dataframe, 'pg2_wedge_upper', np.nan))
        c &= lower.notna() & lower.gt(0.0)
        c &= self._common_guards(dataframe)
        v = c.fillna(False) & dataframe['volume'].gt(0) & dataframe['close'].notna()
        dataframe.loc[v, 'enter_long'] = 1
        dataframe.loc[v, 'enter_tag'] = ENTRY_TAG
        return dataframe
    def populate_exit_trend(self,dataframe:DataFrame,metadata:dict)->DataFrame:dataframe["exit_long"]=0;dataframe["exit_short"]=0;dataframe["exit_tag"]=None;return dataframe
    def _closed(self,pair:str,now:datetime)->DataFrame:f,_=self.dp.get_analyzed_dataframe(pair,self.timeframe);d=pd.to_datetime(f["date"],utc=True,errors="raise");return f.loc[d+pd.to_timedelta(timeframe_to_minutes(self.timeframe),unit="m")<=pd.Timestamp(now)].sort_values("date")
    def _new_state(self,pair:str,trade:Any,order:Any)->dict[str,Any]:
        placed_at=getattr(order,"order_date_utc",None)
        if placed_at is None:
            raise RuntimeError("deep wedge entry snapshot requires the entry order placement time")
        closed=self._closed(pair,placed_at)
        if closed.empty:
            raise RuntimeError("deep wedge entry snapshot requires a fully closed pre-order candle")
        missing=sorted({"low","pg2_wedge_lower"}.difference(closed.columns))
        if missing:
            raise KeyError(f"deep wedge entry snapshot is missing required columns: {missing}")
        row=closed.iloc[-1];lower=float(row["pg2_wedge_lower"]);swing=float(row["low"])
        if not np.isfinite(lower) or lower<=0:
            raise ValueError("deep wedge entry snapshot requires a finite positive lower rail")
        if not np.isfinite(swing) or swing<=0:
            raise ValueError("deep wedge entry snapshot requires a finite positive signal-candle low")
        state={"version":1,"plan":str(self.exit_plan.value),"lower":lower,"signal_swing":swing,"evidence_candle":pd.Timestamp(row["date"]).isoformat(),"order_placed_at":pd.Timestamp(placed_at).isoformat()}
        trade.set_custom_data(key=STATE_KEY,value=state)
        return state
    def _state(self,trade:Any)->dict[str,Any]:
        state=trade.get_custom_data(key=STATE_KEY)
        if state is None:
            raise RuntimeError("deep wedge trade is missing its required entry geometry snapshot; post-entry reconstruction is unsafe")
        if not isinstance(state,dict) or state.get("version")!=1:
            raise ValueError("deep wedge trade geometry snapshot is invalid")
        if state.get("plan") not in {"lower_rail_close","signal_swing_close","signal_swing_low"}:
            raise ValueError("deep wedge trade geometry snapshot has an unknown exit plan")
        for key in ("lower","signal_swing"):
            value=float(state.get(key,0.0))
            if not np.isfinite(value) or value<=0:
                raise ValueError(f"deep wedge trade geometry snapshot has invalid {key}")
        return dict(state)
    def order_filled(self,pair:str,trade:Any,order:Any,current_time:datetime,**kwargs:Any)->None:
        if getattr(order,"ft_order_side",None)==getattr(trade,"entry_side",None) and trade.get_custom_data(key=STATE_KEY) is None:self._new_state(pair,trade,order)
    def custom_exit(self,pair:str,trade:Any,current_time:datetime,current_rate:float,current_profit:float,**kwargs:Any)->str|None:
        s=self._state(trade);closed=self._closed(pair,current_time)
        if closed.empty:raise RuntimeError("deep wedge exit evaluation requires a fully closed candle")
        row=closed.iloc[-1];plan=str(s["plan"]);level=s["lower"] if plan=="lower_rail_close" else s["signal_swing"]
        breached=float(row["low"])<float(level) if plan=="signal_swing_low" else float(row["close"])<float(level)
        return "deep_wedge_invalidation" if breached else None
    def custom_stoploss(self,pair:str,trade:Any,current_time:datetime,current_rate:float,current_profit:float,after_fill:bool,**kwargs:Any)->float:return stoploss_from_absolute(float(trade.open_rate)*0.98,current_rate=current_rate,is_short=False,leverage=float(getattr(trade,"leverage",1.0) or 1.0))
