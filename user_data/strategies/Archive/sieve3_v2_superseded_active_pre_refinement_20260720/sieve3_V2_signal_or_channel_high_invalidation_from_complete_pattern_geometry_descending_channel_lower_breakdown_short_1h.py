from __future__ import annotations
from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from freqtrade.exchange import timeframe_to_minutes  # noqa: E402
from freqtrade.strategy import CategoricalParameter, IStrategy, stoploss_from_absolute
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2  # noqa: E402
  # noqa: E402
ENTRY_MODE="entry_geometry_descending_channel_lower_breakdown_short_1h"; ENTRY_TAG="geometry_descending_channel_lower_breakdown_short_1h"; SIDE="short"; TIMEFRAME="1h"
def _num(frame: DataFrame, column: str, default: float | Series=...) -> Series:
    if column not in frame.columns:
        if default is ...:
            raise KeyError(f'missing required column: {column!r}')
        return pd.to_numeric(
            pd.Series(default, index=frame.index), errors='coerce'
        ).replace([np.inf, -np.inf], np.nan)
    return pd.to_numeric(frame[column], errors='coerce').replace([np.inf, -np.inf], np.nan)
def _bool(frame: DataFrame, column: str) -> Series:
    if column not in frame.columns:
        raise KeyError(f'missing required column: {column!r}')
    return pd.Series(frame[column], index=frame.index).astype('boolean').fillna(False).astype(bool)
def _cross_below(value:Series,level:Series)->Series: return value.lt(level)&value.shift(1).ge(level.shift(1))

EXIT_FAMILY = 'signal_or_channel_high_invalidation'
PRIMARY_TRIGGER = '1h close cross below the descending-channel lower rail with locked score and width qualification'
PRIMARY_GUARD = 'locked volume and bearish-pressure guards'
PRIMARY_TARGET = 'none'
PRIMARY_INVALIDATION = 'entry signal high or entry-frozen descending-channel upper rail'
TARGET_PROVIDER = PRIMARY_TARGET
INVALIDATION_PROVIDER = PRIMARY_INVALIDATION
ENTRY_SOURCE_STAGE = 'sieve2'
SOURCE_STRATEGY = 'sieve3_exit_fixed_tp_sl_from_complete_pattern_geometry_descending_channel_lower_breakdown_short_1h.py:Sieve3ExitFixedTpSlFromCompletePatternGeometryDescendingChannelLowerBreakdownShort1H'
SOURCE_RESULT_BATCH = 'current_executable_defaults_historical_promoted_snapshots_conflict'
RESEARCH_PATH = 'sieve3_exit_signal_or_channel_high_invalidation'
EXIT_HYPOTHESIS = 'Compare two materially higher structural invalidations: the signal candle high and the channel upper rail.'
ACTIVE_SELL_PARAMS = ('high_invalidation_plan',)
SIEVE_STAGE = 'sieve3'
SIDE = 'short'
ENTRY_TAG = 'geometry_descending_channel_lower_breakdown_short_1h'

class Sieve3V2SignalOrChannelHighInvalidationFromCompletePatternGeometryDescendingChannelLowerBreakdownShort1H(IStrategy):
    """Protect the short at the entry signal high or entry-frozen channel upper rail."""
    INTERFACE_VERSION=3; timeframe=TIMEFRAME; startup_candle_count=180; process_only_new_candles=True; can_short=True; max_entry_position_adjustment=0
    SIEVE_STAGE="sieve3"; SOURCE_STRATEGY="sieve3_exit_fixed_tp_sl_from_complete_pattern_geometry_descending_channel_lower_breakdown_short_1h.py:Sieve3ExitFixedTpSlFromCompletePatternGeometryDescendingChannelLowerBreakdownShort1H"; SOURCE_RESULT_BATCH="current_executable_defaults_historical_promoted_snapshots_conflict"; RESEARCH_PATH='sieve3_exit_signal_or_channel_high_invalidation'; ENTRY_SOURCE_STAGE='sieve2'
    EXIT_HYPOTHESIS="Compare two materially higher structural invalidations: the signal candle high and the channel upper rail."; TARGET_PROVIDER="none"; INVALIDATION_PROVIDER="entry signal high or entry-frozen pg2 descending-channel upper"; ACTIVE_SELL_PARAMS=("high_invalidation_plan",)
    
    
    use_volume_guard = True
    volume_guard_window = 48
    volume_ratio_min = 1.0
    use_pressure_guard = True
    pressure_window = 24
    pressure_min = 0.2
    # Legacy fixed-off entry parameter: use_accumulation_guard=False (BooleanParameter, buy space).
    # Legacy fixed-off entry parameter: use_body_direction_guard=False (BooleanParameter, buy space).
    # Legacy fixed-off entry parameter: use_close_direction_guard=False (BooleanParameter, buy space).
    min_pattern_bars = 12
    max_pattern_bars = 72
    compression_max_width_atr = 2.0
    squeeze_active_width_atr = 2.0
    local_narrowing_min_ratio = 0.1
    min_line_score = 0.5
    min_containment = 0.7
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
    high_invalidation_plan=CategoricalParameter(["signal_candle_high","channel_upper_rail"],default="signal_candle_high",space="sell",optimize=True,load=True); _STATE_KEY="sieve3_v2_desc_channel_high"; minimal_roi={"0":100.0}; stoploss=-0.06; use_custom_stoploss=True; use_exit_signal=True; position_adjustment_enable=False; trailing_stop=False
    high_invalidation_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    def leverage(self,pair:str,current_time:datetime,current_rate:float,proposed_leverage:float,max_leverage:float,entry_tag:str|None,side:str,**kwargs:Any)->float: return 1.0
    def informative_pairs(self)->list[tuple[str,str]]: return []
    def populate_indicators(self,dataframe:DataFrame,metadata:dict)->DataFrame:
        dataframe=add_pattern_geometry_v2(dataframe,timeframe=self.timeframe,output_slots=1,include_triangle_patterns=False,include_wedge_patterns=False,include_compression_patterns=False,include_rectangle_patterns=False,include_ascending_channel_patterns=False,include_descending_channel_patterns=True,min_pattern_bars=int(self.min_pattern_bars),max_pattern_bars=int(self.max_pattern_bars),compression_max_width_atr=float(self.compression_max_width_atr),squeeze_active_width_atr=float(self.squeeze_active_width_atr),local_narrowing_min_ratio=float(self.local_narrowing_min_ratio),min_line_score=float(self.min_line_score),min_containment=float(self.min_containment),max_recent_touch_age_bars=int(self.max_recent_touch_age_bars),channel_min_pattern_bars=int(self.channel_min_pattern_bars),channel_max_pattern_bars=int(self.channel_max_pattern_bars),channel_min_quality=float(self.channel_min_quality),channel_min_containment=float(self.channel_min_containment),channel_near_boundary_atr_mult=float(self.channel_near_boundary_atr_mult),channel_breakout_atr_mult=float(self.channel_breakout_atr_mult),channel_lifecycle_confirm_break_bars=int(self.channel_lifecycle_confirm_break_bars),output_prefix="pg2"); return dataframe
    def _common_guards(self,dataframe:DataFrame)->Series:
        guard=pd.Series(True,index=dataframe.index,dtype="bool"); close=_num(dataframe,"close")
        if bool(self.use_volume_guard): volume=_num(dataframe,"volume").clip(lower=0.0); window=int(self.volume_guard_window); baseline=volume.shift(1).rolling(window,min_periods=max(2,window//3)).mean().replace(0.0,np.nan); guard &= volume.ge(baseline.mul(float(self.volume_ratio_min)))
        if bool(self.use_pressure_guard): open_=_num(dataframe,"open"); high=_num(dataframe,"high"); low=_num(dataframe,"low"); volume=_num(dataframe,"volume").clip(lower=0.0).fillna(0.0); candle_range=(high-low).replace(0.0,np.nan); pressure=(( ((close-open_)/candle_range).clip(-1,1).fillna(0)+(((close-low)/candle_range)*2-1).clip(-1,1).fillna(0) )/2).clip(-1,1); directional_volume=(pressure*volume).fillna(0)
        if bool(self.use_pressure_guard): window=int(self.pressure_window); denominator=volume.rolling(window,min_periods=max(2,window//3)).sum().replace(0.0,np.nan); guard &= (directional_volume.rolling(window,min_periods=max(2,window//3)).sum()/denominator).le(-float(self.pressure_min))
        return guard.fillna(False)
    def populate_entry_trend(self,dataframe:DataFrame,metadata:dict)->DataFrame:
        missing = sorted(set(("pg2_descending_channel_lower", "pg2_descending_channel_width_atr")).difference(dataframe.columns))
        if missing:
            raise KeyError(f"{type(self).__name__} entry dataframe is missing required source columns: {missing}")
        dataframe["enter_long"]=0; dataframe["enter_short"]=0; dataframe["enter_tag"]=None; lower=_num(dataframe,"pg2_descending_channel_lower",np.nan).mul(1.0-float(self.rail_buffer_pct)); condition=_bool(dataframe,"pg2_descending_channel_pattern_present")&_num(dataframe,"pg2_descending_channel_indicator_score").ge(float(self.score_min))&_num(dataframe,"pg2_descending_channel_width_atr",np.nan).le(float(self.width_atr_max))&_cross_below(_num(dataframe,"close"),lower); condition &= self._common_guards(dataframe); valid=condition.fillna(False)&dataframe["volume"].gt(0.0)&dataframe["close"].notna(); dataframe.loc[valid,"enter_short"]=1; dataframe.loc[valid,"enter_tag"]=ENTRY_TAG; return dataframe
    def populate_exit_trend(self,dataframe:DataFrame,metadata:dict)->DataFrame: dataframe["exit_long"]=0; dataframe["exit_short"]=0; dataframe["exit_tag"]=None; return dataframe
    def _state(self,pair:str,trade:Any,current_time:datetime)->dict[str,float]:
        state=trade.get_custom_data(key=self._STATE_KEY)
        if state is not None: return dict(state)
        if self.dp is None: raise RuntimeError("data provider required for high invalidation snapshot")
        frame,_=self.dp.get_analyzed_dataframe(pair,self.timeframe); required={"date","high","pg2_descending_channel_upper"}; missing=sorted(required-set(frame.columns))
        if missing: raise KeyError(f"missing exact high invalidation columns: {missing}")
        dates=pd.to_datetime(frame["date"],utc=True,errors="raise")+pd.to_timedelta(timeframe_to_minutes(self.timeframe),unit="m"); freeze=pd.Timestamp(getattr(trade,"open_date_utc",None) or current_time); freeze=freeze.tz_localize("UTC") if freeze.tzinfo is None else freeze.tz_convert("UTC"); closed=frame.loc[dates.le(freeze)]
        if closed.empty: raise RuntimeError("no closed entry candle for high invalidation")
        row=closed.iloc[-1]; state={"signal_candle_high":float(row["high"]),"channel_upper_rail":float(row["pg2_descending_channel_upper"])}; trade.set_custom_data(key=self._STATE_KEY,value=state); return state
    def custom_exit(self,pair:str,trade:Any,current_time:datetime,current_rate:float,current_profit:float,**kwargs:Any)->str|None:
        _=current_profit,kwargs; level=self._state(pair,trade,current_time)[str(self.high_invalidation_plan.value)]; return f"high_invalidation_{self.high_invalidation_plan.value}" if current_rate>=level else None
    def custom_stoploss(self,pair:str,trade:Any,current_time:datetime,current_rate:float,current_profit:float,after_fill:bool,**kwargs:Any)->float|None:
        _=current_profit,after_fill,kwargs; level=self._state(pair,trade,current_time)[str(self.high_invalidation_plan.value)]
        if not np.isfinite(level) or level<=float(trade.open_rate): return None
        return stoploss_from_absolute(level,current_rate=current_rate,is_short=True,leverage=float(getattr(trade,"leverage",1.0) or 1.0))
