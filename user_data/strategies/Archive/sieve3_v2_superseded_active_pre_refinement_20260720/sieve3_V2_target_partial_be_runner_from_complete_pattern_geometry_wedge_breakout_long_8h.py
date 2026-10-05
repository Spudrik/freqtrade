"""Ordered measured target partial, exact-entry stop, then runner/reversal remainder."""
from __future__ import annotations
from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter, IStrategy, stoploss_from_absolute
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2

ENTRY_TAG="geometry_wedge_breakout_long_8h";PARTIAL_TAG="wedge_target_partial";STATE_KEY="s3v2_wedge_partial_be_runner";SIEVE_STAGE="sieve3";ENTRY_SOURCE_STAGE='sieve2';SOURCE_STRATEGY="Sieve2GeometryWedgeBreakoutLong8H tested snapshot 2026-05-23";SOURCE_RESULT_BATCH="20260521T012740_entry_all_resume line 246";RESEARCH_PATH='sieve3_exit_target_partial_be_runner';EXIT_HYPOTHESIS="At a valid measured target, take one ordered partial, move only the remainder stop to exact entry, then hold for a farther target or bearish reversal."
SIDE="long"
LOCKED_BUY_PARAMS={"use_sieve2_vp_guard":False,"sieve2_vp_guard_mode":"score_or_context","sieve2_vp_window":96,"sieve2_vp_bins":36,"sieve2_vp_score_min":0.25,"sieve2_vp_context_min":0.28,"use_sieve2_market_guard":False,"sieve2_market_guard_mode":"pressure_or_trend","sieve2_market_window":24,"sieve2_market_pressure_min":0.07,"sieve2_market_trend_min":0.25,"sieve2_rs_benchmark_pair":"BTC/USDT:USDT","sieve2_rs_score_min":0.45,"use_volume_guard":True,"volume_guard_window":12,"volume_ratio_min":1.6,"use_pressure_guard":True,"pressure_window":24,"pressure_min":0.1,"use_accumulation_guard":False,"use_body_direction_guard":False,"use_close_direction_guard":False,"min_pattern_bars":12,"max_pattern_bars":72,"compression_max_width_atr":2.0,"squeeze_active_width_atr":2.0,"min_line_score":0.4,"min_containment":0.88}
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
EXIT_FAMILY = 'target_partial_be_runner'
ACTIVE_SELL_PARAMS = ('exit_plan',)




class Sieve3V2TargetPartialBeRunnerFromCompletePatternGeometryWedgeBreakoutLong8H(IStrategy):
    ACTIVE_SELL_PARAMS = ('exit_plan',)
    INTERFACE_VERSION=3;timeframe="8h";startup_candle_count=180;process_only_new_candles=True;can_short=False;minimal_roi={"0":100.0};stoploss=-0.99;use_exit_signal=True;use_custom_stoploss=True;trailing_stop=False;position_adjustment_enable=True;max_entry_position_adjustment=0
    
    
    use_volume_guard=True;volume_guard_window=12;volume_ratio_min=1.6;use_pressure_guard=True;pressure_window=24;pressure_min=0.1;use_accumulation_guard=False;use_body_direction_guard=False;use_close_direction_guard=False
    min_pattern_bars=12;max_pattern_bars=72;compression_max_width_atr=2.0;squeeze_active_width_atr=2.0;min_line_score=0.4;min_containment=0.88
    exit_plan=CategoricalParameter(["p33_height75_be_two_lower","p50_height100_be_engulf","p50_height100_be_runner150_or_reversal"],default="p50_height100_be_runner150_or_reversal",space="sell",optimize=True,load=True)
    exit_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    PLANS={"p33_height75_be_two_lower":(0.33,0.75,None,"two_lower"),"p50_height100_be_engulf":(0.50,1.0,None,"engulf"),"p50_height100_be_runner150_or_reversal":(0.50,1.0,1.5,"bearish_lower")}
    def _partial_fill_state(self, trade: Any) -> dict[str, Any]:
        key = f"sieve3_v2_partial_fill:{self.__class__.__name__}"
        state = trade.get_custom_data(key=key)
        if state is None:
            state = {
                "status": "ready",
                "target_stake": None,
                "credited_stake": 0.0,
                "request_stake": None,
                "requested_at": None,
                "order_id": None,
                "resolved_order_ids": [],
            }
            trade.set_custom_data(key=key, value=state)
        return dict(state)

    def _partial_fill_save(self, trade: Any, state: dict[str, Any]) -> None:
        key = f"sieve3_v2_partial_fill:{self.__class__.__name__}"
        trade.set_custom_data(key=key, value=state)

    @staticmethod
    def _partial_fill_order_key(order: Any) -> str:
        order_key = str(getattr(order, "order_id", None) or getattr(order, "id", None) or "")
        if not order_key:
            raise ValueError("partial exit order requires a persistent identity")
        return order_key

    @staticmethod
    def _partial_fill_snapshot(order: Any) -> tuple[bool, float, bool]:
        status = str(getattr(order, "status", None) or "").casefold()
        is_open = bool(getattr(order, "ft_is_open", False)) and status not in {
            "canceled",
            "cancelled",
            "rejected",
            "expired",
            "closed",
        }
        filled_amount = float(getattr(order, "safe_filled", 0.0) or 0.0)
        requested_amount = float(getattr(order, "safe_amount", 0.0) or 0.0)
        filled_stake = max(0.0, float(getattr(order, "stake_amount_filled", 0.0) or 0.0))
        tolerance = max(1e-12, requested_amount * 1e-9)
        return is_open, filled_stake, requested_amount > 0.0 and filled_amount >= requested_amount - tolerance

    def _partial_fill_apply(
        self,
        trade: Any,
        state: dict[str, Any],
        order: Any,
        current_time: datetime,
    ) -> None:
        order_key = self._partial_fill_order_key(order)
        resolved = state.setdefault("resolved_order_ids", [])
        if order_key in {str(value) for value in resolved}:
            return
        is_open, filled_stake, fully_filled = self._partial_fill_snapshot(order)
        if is_open:
            state["status"] = "requested"
            state["order_id"] = order_key
            return
        target_stake = float(state.get("target_stake") or 0.0)
        if filled_stake > 0.0:
            credited = float(state.get("credited_stake") or 0.0) + filled_stake
            state["credited_stake"] = min(target_stake, credited) if target_stake > 0.0 else credited
        resolved.append(order_key)
        tolerance = max(1e-9, target_stake * 1e-9)
        target_reached = target_stake > 0.0 and float(state.get("credited_stake") or 0.0) >= target_stake - tolerance
        if fully_filled or target_reached:
            state["status"] = "filled"
            state["order_id"] = order_key
        else:
            state["status"] = "ready"
            state["order_id"] = None
        state["request_stake"] = None
        state["requested_at"] = None

    def _partial_fill_reconcile(self, trade: Any, current_time: datetime, tag: str) -> dict[str, Any]:
        state = self._partial_fill_state(trade)
        if state["status"] == "filled":
            return state
        resolved = {str(value) for value in state.setdefault("resolved_order_ids", [])}
        orders = [
            order
            for order in (getattr(trade, "orders", ()) or ())
            if getattr(order, "ft_order_side", None) == getattr(trade, "exit_side", None)
            and str(getattr(order, "ft_order_tag", None) or "") == tag
            and self._partial_fill_order_key(order) not in resolved
        ]
        terminal = [order for order in orders if not self._partial_fill_snapshot(order)[0]]
        for order in terminal:
            self._partial_fill_apply(trade, state, order, current_time)
            if state["status"] == "filled":
                break
        if state["status"] != "filled":
            open_order = next((order for order in reversed(orders) if self._partial_fill_snapshot(order)[0]), None)
            if open_order is not None:
                self._partial_fill_apply(trade, state, open_order, current_time)
            elif not terminal and state["status"] == "requested":
                requested_at = state.get("requested_at")
                if requested_at != pd.Timestamp(current_time).isoformat():
                    state["status"] = "ready"
                    state["request_stake"] = None
                    state["requested_at"] = None
                    state["order_id"] = None
        self._partial_fill_save(trade, state)
        return state

    def _partial_fill_complete(self, trade: Any, current_time: datetime, tag: str) -> bool:
        return self._partial_fill_reconcile(trade, current_time, tag)["status"] == "filled"

    def _partial_fill_request(
        self,
        trade: Any,
        current_time: datetime,
        fraction: float,
        min_stake: float | None,
        tag: str,
    ) -> float | None:
        state = self._partial_fill_reconcile(trade, current_time, tag)
        if state["status"] != "ready" or bool(getattr(trade, "has_open_orders", False)):
            return None
        current_stake = float(getattr(trade, "stake_amount", 0.0) or 0.0)
        target_stake = state.get("target_stake")
        if target_stake is None:
            target_stake = current_stake * float(fraction)
            if min_stake is not None and 0.0 < current_stake - target_stake < float(min_stake):
                target_stake = current_stake - float(min_stake)
            if target_stake <= 0.0 or target_stake >= current_stake:
                return None
            state["target_stake"] = target_stake
        target_stake = float(target_stake)
        credited_stake = max(0.0, float(state.get("credited_stake") or 0.0))
        remaining_stake = max(0.0, target_stake - credited_stake)
        tolerance = max(1e-9, target_stake * 1e-9)
        if remaining_stake <= tolerance:
            state["status"] = "filled"
            self._partial_fill_save(trade, state)
            return None
        request = min(current_stake, remaining_stake)
        if request <= 0.0 or request >= current_stake:
            return None
        state["status"] = "requested"
        state["request_stake"] = request
        state["requested_at"] = pd.Timestamp(current_time).isoformat()
        state["order_id"] = None
        self._partial_fill_save(trade, state)
        return request


    def leverage(self,pair:str,current_time:datetime,current_rate:float,proposed_leverage:float,max_leverage:float,entry_tag:str|None,side:str,**kwargs:Any)->float:return 1.0
    def informative_pairs(self)->list[tuple[str,str]]:return []
    def populate_indicators(self,dataframe:DataFrame,metadata:dict)->DataFrame:dataframe=add_pattern_geometry_v2(dataframe,timeframe=self.timeframe,output_slots=1,include_triangle_patterns=False,include_wedge_patterns=True,include_compression_patterns=False,include_rectangle_patterns=False,include_ascending_channel_patterns=False,include_descending_channel_patterns=False,min_pattern_bars=int(self.min_pattern_bars),max_pattern_bars=int(self.max_pattern_bars),compression_max_width_atr=float(self.compression_max_width_atr),squeeze_active_width_atr=float(self.squeeze_active_width_atr),min_line_score=float(self.min_line_score),min_containment=float(self.min_containment),output_prefix="pg2");return dataframe
    def _common_guards(self,f:DataFrame)->Series:
        g=pd.Series(True,index=f.index,dtype="bool");close=_num(f,"close")
        if bool(self.use_volume_guard):volume=_num(f,"volume").clip(lower=0);w=int(self.volume_guard_window);g&=volume.ge(volume.shift(1).rolling(w,min_periods=max(2,w//3)).mean().replace(0,np.nan).mul(float(self.volume_ratio_min)))
        if bool(self.use_pressure_guard) or bool(self.use_accumulation_guard):open_=_num(f,"open");high=_num(f,"high");low=_num(f,"low");volume=_num(f,"volume").clip(lower=0).fillna(0);r=(high-low).replace(0,np.nan);p=((((close-open_)/r).clip(-1,1).fillna(0)+(((close-low)/r)*2-1).clip(-1,1).fillna(0))/2).clip(-1,1);dv=(p*volume).fillna(0)
        if bool(self.use_pressure_guard):w=int(self.pressure_window);base=volume.rolling(w,min_periods=max(2,w//3)).sum().replace(0,np.nan);g&=dv.rolling(w,min_periods=max(2,w//3)).sum().div(base).ge(float(self.pressure_min))
        if bool(self.use_accumulation_guard):w=int(self.pressure_window);g&=dv.rolling(w,min_periods=max(2,w//3)).sum().ge(0)
        if bool(self.use_body_direction_guard):g&=close.gt(_num(f,"open"))
        if bool(self.use_close_direction_guard):g&=close.gt(close.shift(1))
        return g.fillna(False)
    def populate_entry_trend(self,dataframe:DataFrame,metadata:dict)->DataFrame:
        dataframe["enter_long"]=0;dataframe["enter_short"]=0;dataframe["enter_tag"]=None;c=_bool(dataframe,"pg2_wedge_pattern_present")&_bool(dataframe,"pg2_wedge_squeeze_active")&_num(dataframe,"pg2_wedge_indicator_score").ge(float(self.min_line_score))&_num(dataframe,"pg2_wedge_direction").ge(0)&_num(dataframe,"close").gt(_num(dataframe,"pg2_wedge_upper"));c&=self._common_guards(dataframe);c=pd.Series(c, index=dataframe.index).fillna(False).astype(bool).fillna(False);v=c.fillna(False)&dataframe["volume"].gt(0)&dataframe["close"].notna();dataframe.loc[v,"enter_long"]=1;dataframe.loc[v,"enter_tag"]=ENTRY_TAG;return dataframe
    def populate_exit_trend(self,dataframe:DataFrame,metadata:dict)->DataFrame:dataframe["exit_long"]=0;dataframe["exit_short"]=0;dataframe["exit_tag"]=None;return dataframe
    def _closed(self,pair:str,now:datetime)->DataFrame:
        if self.dp is None: raise RuntimeError("wedge exit requires Freqtrade's data provider")
        f,_=self.dp.get_analyzed_dataframe(pair,self.timeframe);required={"close","date","high","open","pg2_slot_1_active","pg2_slot_1_family","pg2_slot_1_lower_start","pg2_slot_1_upper_start","pg2_wedge_upper"};missing=sorted(required.difference(f.columns))
        if missing: raise KeyError(f"wedge exit dataframe is missing required columns: {missing}")
        d=pd.to_datetime(f["date"],utc=True,errors="raise");cutoff=pd.Timestamp(now);cutoff=cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC");return f.loc[d+pd.to_timedelta(timeframe_to_minutes(self.timeframe),unit="m")<=cutoff].sort_values("date")
    def _state(self,pair:str,trade:Any,now:datetime)->dict[str,Any]:
        s=trade.get_custom_data(key=STATE_KEY)
        if s is not None:return dict(s)
        entry_rows=self._closed(pair,trade.open_date_utc)
        if entry_rows.empty: raise RuntimeError("no fully closed entry candle is available for wedge target snapshot")
        row=entry_rows.iloc[-1];fraction,m1,m2,reversal=self.PLANS[str(self.exit_plan.value)];upper=float(row["pg2_slot_1_upper_start"]);lower=float(row["pg2_slot_1_lower_start"]);rail=float(row["pg2_wedge_upper"]);height=upper-lower;valid=bool(row["pg2_slot_1_active"]) and int(row["pg2_slot_1_family"])==2 and np.isfinite([upper,lower,rail,height]).all() and height>0;t1=rail+height*m1 if valid and rail+height*m1>float(trade.open_rate) else None;t2=rail+height*m2 if valid and m2 is not None and rail+height*m2>float(trade.open_rate) else None;s={"plan":str(self.exit_plan.value),"fraction":fraction,"target_1":t1,"target_2":t2,"reversal":reversal,"initial_stake":float(trade.stake_amount)};trade.set_custom_data(key=STATE_KEY,value=s);return s
    def order_filled(self,pair:str,trade:Any,order:Any,current_time:datetime,**kwargs:Any)->None:
        if getattr(order,"ft_order_side",None)==getattr(trade,"entry_side",None) and trade.get_custom_data(key=STATE_KEY) is None:self._state(pair,trade,current_time)
        if getattr(order,"ft_order_side",None)==getattr(trade,"exit_side",None) and str(getattr(order,"ft_order_tag",None) or "")==PARTIAL_TAG:
            state=self._partial_fill_state(trade);self._partial_fill_apply(trade,state,order,current_time);self._partial_fill_save(trade,state)
    def adjust_trade_position(self,trade:Any,current_time:datetime,current_rate:float,current_profit:float,min_stake:float|None,max_stake:float,current_entry_rate:float,current_exit_rate:float,current_entry_profit:float,current_exit_profit:float,**kwargs:Any)->float|tuple[float,str]|None:
        s=self._state(str(trade.pair),trade,current_time)
        frame=self._closed(str(trade.pair),current_time);opened=pd.Timestamp(trade.open_date_utc);opened=opened.tz_localize("UTC") if opened.tzinfo is None else opened.tz_convert("UTC");close_times=pd.to_datetime(frame["date"],utc=True,errors="raise")+pd.to_timedelta(timeframe_to_minutes(self.timeframe),unit="m");post_entry=frame.loc[close_times.gt(opened)]
        if s.get("target_1") is None or post_entry.empty or pd.to_numeric(post_entry["high"],errors="coerce").max()<float(s["target_1"])*0.98:return None
        request=self._partial_fill_request(trade,current_time,float(s["fraction"]),min_stake,PARTIAL_TAG)
        return (-request,PARTIAL_TAG) if request is not None else None
    def custom_exit(self,pair:str,trade:Any,current_time:datetime,current_rate:float,current_profit:float,**kwargs:Any)->str|None:
        if not self._partial_fill_complete(trade,current_time,PARTIAL_TAG) or bool(getattr(trade,"has_open_orders",False)):return None
        s=self._state(pair,trade,current_time)
        if s.get("target_2") is not None and current_rate>=float(s["target_2"]):return "wedge_runner_target"
        f=self._closed(pair,current_time)
        if len(f)<2:return None
        a,b=f.iloc[-2],f.iloc[-1];bearish=float(b["close"])<float(b["open"]);lower=float(b["close"])<float(a["close"]);engulf=bearish and float(a["close"])>float(a["open"]) and float(b["open"])>=float(a["close"]) and float(b["close"])<=float(a["open"]);mode=str(s["reversal"]);reversed_=engulf if mode=="engulf" else len(f)>=3 and lower and float(a["close"])<float(f.iloc[-3]["close"]) if mode=="two_lower" else bearish and lower
        return "post_partial_bearish_reversal" if reversed_ else None
    def custom_stoploss(self,pair:str,trade:Any,current_time:datetime,current_rate:float,current_profit:float,after_fill:bool,**kwargs:Any)->float:
        stop_price=float(trade.open_rate) if self._partial_fill_complete(trade,current_time,PARTIAL_TAG) else float(trade.open_rate)*0.98
        return stoploss_from_absolute(stop_price,current_rate=current_rate,is_short=False,leverage=float(getattr(trade,"leverage",1.0) or 1.0))
