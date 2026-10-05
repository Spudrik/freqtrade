"""Ordered measured-target partial, exact entry stop, then runner/reversal remainder."""
from __future__ import annotations
import math
from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd
from pandas import DataFrame,Series
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.strategy import CategoricalParameter, IStrategy, stoploss_from_absolute
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2
ENTRY_MODE,ENTRY_TAG,SIDE,TIMEFRAME="entry_geometry_rectangle_breakdown_short_1h","geometry_rectangle_breakdown_short_1h","short","1h"
ENTRY_LOCK_STATUS="current executable defaults locked; promoted historical snapshots conflict"
LAYERED_PLANS={"half_33_be_full_runner":(.5,.33,1.),"three_quarter_50_be_full_runner":(.75,.5,1.),"full_33_be_reversal_runner":(1.,.33,None),"full_50_be_reversal_runner":(1.,.5,None)}
def tagged_exit_parameter(p:Any)->Any:setattr(p,"batch_tags",("family:exits","mode:sieve3_exit"));return p
def _num(f: DataFrame, c: str) -> Series:
    return pd.to_numeric(f[c],errors="coerce").replace([np.inf,-np.inf],np.nan)

def _bool(f: DataFrame, c: str) -> Series:
    return pd.Series(f[c],index=f.index).astype("boolean").fillna(False).astype(bool)

def _cross_below(v:Series,l:Series)->Series:return v.lt(l)&v.shift(1).ge(l.shift(1))
SIEVE_STAGE = "sieve3"
ENTRY_SOURCE_STAGE = "sieve2"
RESEARCH_PATH = "sieve3_exit_measured_target_partial_breakeven_runner"
SOURCE_RESULT_BATCH = "historical_promotion_lineage_unknown"
SOURCE_STRATEGY = "sieve2_complete_pattern_geometry_rectangle_breakdown_short_1h"
EXIT_HYPOTHESIS = "Test the measured target partial breakeven runner exit family while preserving the complete pattern geometry rectangle breakdown short 1h entry behavior."


class Sieve3V2MeasuredTargetPartialBreakevenRunnerFromCompletePatternGeometryRectangleBreakdownShort1H(IStrategy):
    INTERFACE_VERSION,timeframe,startup_candle_count,process_only_new_candles,can_short=3,TIMEFRAME,180,True,True
    minimal_roi,stoploss,use_exit_signal,use_custom_stoploss={"0":100.},-.08,True,True
    position_adjustment_enable,max_entry_position_adjustment,trailing_stop=True,0,False
    exit_profit_only,ignore_roi_if_entry_signal=False,False
    
    
    # Inactive buy declaration: use_volume_guard = BooleanParameter(default=False, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: volume_guard_window = CategoricalParameter(default=48, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: volume_ratio_min = CategoricalParameter(default=1.0, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: use_pressure_guard = BooleanParameter(default=True, space="buy"); fixed-on entry gate makes its guard unconditional.
    # Inactive buy declaration: use_accumulation_guard = BooleanParameter(default=False, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: use_body_direction_guard = BooleanParameter(default=False, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    # Inactive buy declaration: use_close_direction_guard = BooleanParameter(default=False, space="buy"); fixed-off entry gate or fixed mode makes its branch unreachable.
    pressure_window = 24; pressure_min = 0.15
    min_pattern_bars = 12;max_pattern_bars = 72;compression_max_width_atr = 2.0;squeeze_active_width_atr = 2.0;local_narrowing_min_ratio = 0.1;min_line_score = 0.7;min_containment = 0.88;max_recent_touch_age_bars = 12;channel_min_pattern_bars = 18;channel_max_pattern_bars = 96;channel_min_quality = 0.82;channel_min_containment = 0.68;channel_near_boundary_atr_mult = 0.7;channel_breakout_atr_mult = 0.35;channel_lifecycle_confirm_break_bars = 2;score_min = 0.65;width_atr_max = 6.0;rail_buffer_pct = 0.004
    ACTIVE_SELL_PARAMS = ('layered_plan',)

    layered_plan=tagged_exit_parameter(CategoricalParameter(tuple(LAYERED_PLANS),default="half_33_be_full_runner",space="sell",optimize=True,load=True))
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
    def _utc(v:Any)->pd.Timestamp:t=pd.Timestamp(v);return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
    def _closed(self,pair:str,when:Any)->DataFrame:
        if self.dp is None:raise RuntimeError("layered measured exit requires data provider")
        f,_=self.dp.get_analyzed_dataframe(pair,self.timeframe);needed={"date","open","high","close","pg2_rectangle_upper","pg2_rectangle_lower"};missing=sorted(needed-set(f.columns))
        if missing:raise KeyError(f"layered measured exit missing columns: {missing}")
        closes=pd.to_datetime(f["date"],utc=True,errors="raise")+pd.to_timedelta(timeframe_to_minutes(self.timeframe),unit="m");return f.loc[closes.le(self._utc(when))].sort_values("date")
    def _levels(self,pair:str,trade:Any)->tuple[float,float,float]:
        f=self._closed(pair,trade.open_date_utc)
        if f.empty:raise RuntimeError("no closed entry-signal candle")
        r=f.iloc[-1];upper,lower=float(r["pg2_rectangle_upper"]),float(r["pg2_rectangle_lower"])
        if not math.isfinite(upper+lower) or not 0.<lower<upper:raise ValueError("invalid entry rectangle rails")
        return upper,lower,lower*(1.-float(self.rail_buffer_pct))
    PARTIAL_STAGE_TAG = "measured_target_partial"
    PARTIAL_STATE_KEY = "s3v2_partial_stage_state"

    @classmethod
    def _partial_state(cls, trade: Any) -> dict[str, Any]:
        raw = trade.get_custom_data(key=cls.PARTIAL_STATE_KEY)
        if isinstance(raw, dict):
            return dict(raw)
        return {
            "status": "ready",
            "target_stake": None,
            "credited_stake": 0.0,
            "order_credits": {},
            "requested_at": None,
            "filled_at": None,
        }

    @staticmethod
    def _partial_order_stake(order: Any, trade: Any, filled: bool) -> float:
        amount_name = "safe_filled" if filled else "safe_amount"
        amount = float(getattr(order, amount_name, 0.0) or 0.0)
        price = float(getattr(order, "safe_price", 0.0) or 0.0)
        leverage = float(getattr(trade, "leverage", 1.0) or 1.0)
        return amount * price / leverage

    def _reconcile_partial(self, trade: Any, current_time: Any, extra_order: Any | None = None) -> dict[str, Any]:
        state = self._partial_state(trade)
        before = repr(state)
        orders = list(getattr(trade, "orders", ()) or ())
        if extra_order is not None and all(order is not extra_order for order in orders):
            orders.append(extra_order)
        matching = [
            order
            for order in orders
            if getattr(order, "ft_order_side", None) == getattr(trade, "exit_side", None)
            and str(getattr(order, "ft_order_tag", None) or "") == self.PARTIAL_STAGE_TAG
        ]
        if state.get("target_stake") is None and matching:
            state["target_stake"] = self._partial_order_stake(matching[0], trade, filled=False)
        credits = dict(state.get("order_credits") or {})
        credited = float(state.get("credited_stake") or 0.0)
        has_open = False
        fully_filled = False
        for order in matching:
            order_id = str(getattr(order, "order_id", None) or "")
            if not order_id:
                raise ValueError("partial-stage reconciliation requires an order id")
            filled_stake = self._partial_order_stake(order, trade, filled=True)
            previous = float(credits.get(order_id) or 0.0)
            if filled_stake > previous:
                credited += filled_stake - previous
                credits[order_id] = filled_stake
            status = str(getattr(order, "status", None) or "").casefold()
            is_open = bool(getattr(order, "ft_is_open", False)) and status not in {
                "canceled", "cancelled", "closed", "expired", "failed", "rejected",
            }
            requested_amount = float(getattr(order, "safe_amount", 0.0) or 0.0)
            filled_amount = float(getattr(order, "safe_filled", 0.0) or 0.0)
            has_open |= is_open
            fully_filled |= requested_amount > 0.0 and filled_amount >= requested_amount - max(1e-12, requested_amount * 1e-9)
        state["order_credits"] = credits
        state["credited_stake"] = credited
        target = float(state.get("target_stake") or 0.0)
        tolerance = max(1e-8, target * 0.005)
        if fully_filled or (target > 0.0 and credited >= target - tolerance):
            state["status"] = "filled"
            state["requested_at"] = None
            if state.get("filled_at") is None:
                state["filled_at"] = pd.Timestamp(current_time).isoformat()
        elif has_open or (
            state.get("status") == "requested"
            and not matching
            and state.get("requested_at") == pd.Timestamp(current_time).isoformat()
        ):
            state["status"] = "requested"
        else:
            state["status"] = "ready"
            state["requested_at"] = None
        if repr(state) != before:
            trade.set_custom_data(key=self.PARTIAL_STATE_KEY, value=state)
        return state

    def _request_partial(self, trade: Any, current_time: Any, desired: float, min_stake: float | None) -> float | None:
        state = self._reconcile_partial(trade, current_time)
        if state["status"] != "ready":
            return None
        stake = float(getattr(trade, "stake_amount", 0.0) or 0.0)
        if stake <= 0.0:
            return None
        target = state.get("target_stake")
        request = min(stake, desired) if target is None else min(stake, max(0.0, float(target) - float(state["credited_stake"])))
        if min_stake is not None and 0.0 < stake - request < float(min_stake):
            request = stake - float(min_stake)
        if request <= 0.0 or request >= stake:
            return None
        if target is None:
            state["target_stake"] = request
        state["status"] = "requested"
        state["requested_at"] = pd.Timestamp(current_time).isoformat()
        trade.set_custom_data(key=self.PARTIAL_STATE_KEY, value=state)
        return request

    def _partial_complete(self, trade: Any, current_time: Any) -> bool:
        return self._reconcile_partial(trade, current_time)["status"] == "filled"


    def adjust_trade_position(self,trade:Any,current_time:datetime,current_rate:float,current_profit:float,min_stake:float|None,max_stake:float,current_entry_rate:float,current_exit_rate:float,current_entry_profit:float,current_exit_profit:float,**kwargs:Any)->tuple[float,str]|None:
        if bool(getattr(trade,"has_open_orders",False)):return None
        multiple,fraction,_=LAYERED_PLANS[str(self.layered_plan.value)];upper,lower,_=self._levels(trade.pair,trade);target=lower-(upper-lower)*multiple
        if current_rate>target:return None
        request=self._request_partial(trade,current_time,float(trade.stake_amount)*fraction,min_stake)
        return (-request,self.PARTIAL_STAGE_TAG) if request is not None else None
    def order_filled(self,pair:str,trade:Any,order:Any,current_time:datetime,**kwargs:Any)->None:
        if getattr(order,"ft_order_side",None)==getattr(trade,"exit_side",None) and str(getattr(order,"ft_order_tag",None) or "")==self.PARTIAL_STAGE_TAG:self._reconcile_partial(trade,current_time,order)
    def custom_stoploss(self,pair:str,trade:Any,current_time:datetime,current_rate:float,current_profit:float,after_fill:bool,**kwargs:Any)->float:
        stop=float(trade.open_rate) if self._partial_complete(trade,current_time) else float(trade.open_rate)*1.03
        return stoploss_from_absolute(stop,current_rate=current_rate,is_short=True,leverage=float(trade.leverage or 1.))
    def custom_exit(self,pair:str,trade:Any,current_time:datetime,current_rate:float,current_profit:float,**kwargs:Any)->str|None:
        if not self._partial_complete(trade,current_time):return None
        plan=str(self.layered_plan.value);_,_,runner_multiple=LAYERED_PLANS[plan];f=self._closed(pair,current_time)
        if len(f)<2:return None
        upper,lower,break_level=self._levels(pair,trade);latest,prev=f.iloc[-1],f.iloc[-2];reversal=float(latest["close"])>break_level or (float(latest["close"])>float(latest["open"]) and float(latest["close"])>float(prev["high"]));runner_hit=runner_multiple is not None and current_rate<=lower-(upper-lower)*runner_multiple
        return "post_partial_runner_target" if runner_hit else "post_partial_bullish_reversal" if reversal else None
ACTIVE_SELL_PARAMS=("layered_plan",)
