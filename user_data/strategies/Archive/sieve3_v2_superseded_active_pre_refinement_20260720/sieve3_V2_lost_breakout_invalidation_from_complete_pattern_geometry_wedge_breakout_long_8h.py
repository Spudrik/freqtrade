"""Lost-breakout invalidations against the entry-frozen upper wedge rail."""
from __future__ import annotations
from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
from freqtrade.strategy import CategoricalParameter, IStrategy, stoploss_from_absolute
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2
ENTRY_TAG="geometry_wedge_breakout_long_8h"; STATE_KEY="s3v2_lost_wedge_breakout"
SIEVE_STAGE="sieve3"; ENTRY_SOURCE_STAGE="sieve2"; SOURCE_STRATEGY="Sieve2GeometryWedgeBreakoutLong8H tested snapshot 2026-05-23"; SOURCE_RESULT_BATCH="20260521T012740_entry_all_resume line 246"; RESEARCH_PATH="sieve3_exit_lost_breakout_invalidation"; EXIT_HYPOTHESIS="A close back under the entry-frozen upper rail means the bullish wedge breakout did not hold."
LOCKED_BUY_PARAMS={"use_sieve2_vp_guard":False,"sieve2_vp_guard_mode":"score_or_context","sieve2_vp_window":96,"sieve2_vp_bins":36,"sieve2_vp_score_min":0.25,"sieve2_vp_context_min":0.28,"use_sieve2_market_guard":False,"sieve2_market_guard_mode":"pressure_or_trend","sieve2_market_window":24,"sieve2_market_pressure_min":0.07,"sieve2_market_trend_min":0.25,"sieve2_rs_benchmark_pair":"BTC/USDT:USDT","sieve2_rs_score_min":0.45,"use_volume_guard":True,"volume_guard_window":12,"volume_ratio_min":1.6,"use_pressure_guard":True,"pressure_window":24,"pressure_min":0.1,"use_accumulation_guard":False,"use_body_direction_guard":False,"use_close_direction_guard":False,"min_pattern_bars":12,"max_pattern_bars":72,"compression_max_width_atr":2.0,"squeeze_active_width_atr":2.0,"min_line_score":0.4,"min_containment":0.88}
def _num(f: DataFrame, c: str) -> Series:
    return pd.to_numeric(f[c],errors="coerce").replace([np.inf,-np.inf],np.nan)

def _bool(f: DataFrame, c: str) -> Series:
    return pd.Series(f[c],index=f.index).astype("boolean").fillna(False).astype(bool)

ACTIVE_SELL_PARAMS = ("exit_plan",)

class Sieve3V2LostBreakoutInvalidationFromCompletePatternGeometryWedgeBreakoutLong8H(IStrategy):
    INTERFACE_VERSION=3; timeframe="8h"; startup_candle_count=180; process_only_new_candles=True; can_short=False; minimal_roi={"0":100.0}; stoploss=-0.99; use_exit_signal=True; use_custom_stoploss=True; trailing_stop=False; position_adjustment_enable=False
    ACTIVE_SELL_PARAMS = ("exit_plan",)
    
    
    # Inactive buy declaration: use_volume_guard = BooleanParameter(default=True, space="buy"); fixed-on entry gate makes its guard unconditional.
    # Inactive buy declaration: use_pressure_guard = BooleanParameter(default=True, space="buy"); fixed-on entry gate makes its guard unconditional.
    # Inactive buy declaration: use_accumulation_guard = BooleanParameter(default=False, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: use_body_direction_guard = BooleanParameter(default=False, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: use_close_direction_guard = BooleanParameter(default=False, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    volume_guard_window = 12; volume_ratio_min = 1.6; pressure_window = 24; pressure_min = 0.1
    min_pattern_bars = 12; max_pattern_bars = 72; compression_max_width_atr = 2.0; squeeze_active_width_atr = 2.0; min_line_score = 0.4; min_containment = 0.88
    exit_plan=CategoricalParameter(["one_close_below_upper","two_closes_below_upper","one_close_below_upper_band_0_5"],default="two_closes_below_upper",space="sell",optimize=True,load=True)
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
    def populate_entry_trend(self,dataframe:DataFrame,metadata:dict)->DataFrame:
        dataframe["enter_long"]=0;dataframe["enter_short"]=0;dataframe["enter_tag"]=None;c=_bool(dataframe,"pg2_wedge_pattern_present")&_bool(dataframe,"pg2_wedge_squeeze_active")&_num(dataframe,"pg2_wedge_indicator_score").ge(float(self.min_line_score))&_num(dataframe,"pg2_wedge_direction").ge(0)&_num(dataframe,"close").gt(_num(dataframe,"pg2_wedge_upper"));c&=self._common_guards(dataframe);v=c.fillna(False)&dataframe["volume"].gt(0)&dataframe["close"].notna();dataframe.loc[v,"enter_long"]=1;dataframe.loc[v,"enter_tag"]=ENTRY_TAG;return dataframe
    def populate_exit_trend(self,dataframe:DataFrame,metadata:dict)->DataFrame:dataframe["exit_long"]=0;dataframe["exit_short"]=0;dataframe["exit_tag"]=None;return dataframe
    @staticmethod
    def _utc(value:Any)->pd.Timestamp:
        timestamp=pd.Timestamp(value);return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")
    def _closed(self,pair:str,now:datetime)->DataFrame:
        f,_=self.dp.get_analyzed_dataframe(pair,self.timeframe);d=pd.to_datetime(f["date"],utc=True,errors="raise");closes_at=d+pd.to_timedelta(timeframe_to_minutes(self.timeframe),unit="m");return f.loc[closes_at.le(self._utc(now))].sort_values("date")
    def _post_entry_closed(self,pair:str,trade:Any,now:datetime)->DataFrame:
        f=self._closed(pair,now)
        filled_at=self._utc(getattr(trade,"date_entry_fill_utc",None) or trade.open_date_utc)
        candle_open=self._utc(timeframe_to_prev_date(self.timeframe,filled_at.to_pydatetime()))
        dates=pd.to_datetime(f["date"],utc=True,errors="raise")
        eligible=dates.ge(candle_open) if filled_at==candle_open else dates.gt(candle_open)
        return f.loc[eligible]
    def _upper(self,pair:str,trade:Any,now:datetime)->float|None:
        value=trade.get_custom_data(key=STATE_KEY)
        if value is not None:return float(value)
        filled_at=getattr(trade,"date_entry_fill_utc",None) or trade.open_date_utc
        signal=self._closed(pair,filled_at)
        if signal.empty:raise RuntimeError("no fully closed pre-entry signal candle is available")
        upper=float(signal.iloc[-1]["pg2_wedge_upper"])
        if not np.isfinite(upper) or upper<=0:raise ValueError("entry-frozen wedge upper boundary must be finite and positive")
        trade.set_custom_data(key=STATE_KEY,value=upper)
        return upper
    def order_filled(self,pair:str,trade:Any,order:Any,current_time:datetime,**kwargs:Any)->None:
        if getattr(order,"ft_order_side",None)==getattr(trade,"entry_side",None) and trade.get_custom_data(key=STATE_KEY) is None:self._upper(pair,trade,current_time)
    def custom_exit(self,pair:str,trade:Any,current_time:datetime,current_rate:float,current_profit:float,**kwargs:Any)->str|None:
        upper=self._upper(pair,trade,current_time);f=self._post_entry_closed(pair,trade,current_time)
        if upper is None or f.empty:return None
        closes=pd.to_numeric(f["close"],errors="coerce");plan=str(self.exit_plan.value)
        failed=bool(closes.iloc[-1]<upper*0.995) if plan=="one_close_below_upper_band_0_5" else bool(closes.iloc[-1]<upper) if plan=="one_close_below_upper" else len(closes)>=2 and bool(closes.iloc[-2:].lt(upper).all())
        return "lost_wedge_breakout" if failed else None
    def custom_stoploss(self,pair:str,trade:Any,current_time:datetime,current_rate:float,current_profit:float,after_fill:bool,**kwargs:Any)->float:return stoploss_from_absolute(float(trade.open_rate)*0.97,current_rate=current_rate,is_short=False,leverage=float(getattr(trade,"leverage",1.0) or 1.0))
