from __future__ import annotations
from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd
from pandas import DataFrame,Series
from freqtrade.exchange import timeframe_to_minutes  # noqa: E402
from freqtrade.strategy import CategoricalParameter, IStrategy, stoploss_from_absolute  # noqa: E402
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2  # noqa: E402
  # noqa: E402
ENTRY_MODE="entry_geometry_descending_channel_lower_breakdown_short_1h"; ENTRY_TAG="geometry_descending_channel_lower_breakdown_short_1h"; SIDE="short"; TIMEFRAME="1h"
def _num(frame: DataFrame, column: str, _default: float | Series | object = ...) -> Series:
    if column not in frame.columns:
        if _default is ...:
            raise KeyError(f"required numeric column is missing: {column!r}")
        if isinstance(_default, Series):
            return pd.to_numeric(_default, errors="coerce").reindex(frame.index)
        return pd.Series(float(_default), index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
def _bool(frame: DataFrame, column: str) -> Series:
    return pd.Series(frame[column], index=frame.index).astype("boolean").fillna(False).astype(bool)
def _cross_below(value:Series,level:Series)->Series: return value.lt(level)&value.shift(1).ge(level.shift(1))

EXIT_FAMILY = 'target_partial_entry_stop_runner'

ENTRY_SOURCE_STAGE = 'sieve2'
RESEARCH_PATH = 'sieve3_exit_target_partial_entry_stop_runner'
SOURCE_RESULT_BATCH = 'current_executable_defaults_historical_promoted_snapshots_conflict'
SOURCE_STRATEGY = 'sieve3_exit_fixed_tp_sl_from_complete_pattern_geometry_descending_channel_lower_breakdown_short_1h.py:Sieve3ExitFixedTpSlFromCompletePatternGeometryDescendingChannelLowerBreakdownShort1H'
EXIT_HYPOTHESIS = 'Take one partial at a channel-width target, move only the remainder stop exactly to trade.open_rate, then hold for a deeper extension unless bullish weakening appears.'


class Sieve3V2TargetPartialEntryStopRunnerFromCompletePatternGeometryDescendingChannelLowerBreakdownShort1H(IStrategy):
    """Ordered measured target partial, exact-entry stop, then runner target or bullish weakening."""
    INTERFACE_VERSION=3; timeframe=TIMEFRAME; startup_candle_count=180; process_only_new_candles=True; can_short=True; max_entry_position_adjustment=0
    SIEVE_STAGE="sieve3"; SOURCE_STRATEGY="sieve3_exit_fixed_tp_sl_from_complete_pattern_geometry_descending_channel_lower_breakdown_short_1h.py:Sieve3ExitFixedTpSlFromCompletePatternGeometryDescendingChannelLowerBreakdownShort1H"; SOURCE_RESULT_BATCH="current_executable_defaults_historical_promoted_snapshots_conflict"; RESEARCH_PATH='sieve3_exit_target_partial_entry_stop_runner'; ENTRY_SOURCE_STAGE='sieve2'
    EXIT_HYPOTHESIS="Take one partial at a channel-width target, move only the remainder stop exactly to trade.open_rate, then hold for a deeper extension unless bullish weakening appears."; TARGET_PROVIDER="entry-frozen descending-channel width extensions"; INVALIDATION_PROVIDER="post-partial exact entry stop plus bullish score/candle weakening"; ACTIVE_SELL_PARAMS=("ordered_plan","runner_weakening_plan")
    
    
    use_volume_guard=True; volume_guard_window=48; volume_ratio_min=1.0; use_pressure_guard=True; pressure_window=24; pressure_min=0.2; use_accumulation_guard=False; use_body_direction_guard=False; use_close_direction_guard=False
    min_pattern_bars=12; max_pattern_bars=72; compression_max_width_atr=2.0; squeeze_active_width_atr=2.0; local_narrowing_min_ratio=0.1; min_line_score=0.5; min_containment=0.7; max_recent_touch_age_bars=12
    channel_min_pattern_bars=18; channel_max_pattern_bars=96; channel_min_quality=0.82; channel_min_containment=0.68; channel_near_boundary_atr_mult=0.7; channel_breakout_atr_mult=0.35; channel_lifecycle_confirm_break_bars=2; score_min=0.65; width_atr_max=6.0; rail_buffer_pct=0.004
    ordered_plan=CategoricalParameter(["p33_halfwidth_runner_onewidth","p33_onewidth_runner_onehalf","p50_onewidth_runner_two_widths"],default="p33_onewidth_runner_onehalf",space="sell",optimize=True,load=True); runner_weakening_plan=CategoricalParameter(["bullish_score_below_locked_min","bullish_score_drop_50pct","target_only"],default="bullish_score_below_locked_min",space="sell",optimize=True,load=True)
    ordered_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    runner_weakening_plan.batch_tags = ('family:exits', 'mode:sieve3_exit')
    _PLANS={"p33_halfwidth_runner_onewidth":(0.33,0.5,1.0),"p33_onewidth_runner_onehalf":(0.33,1.0,1.5),"p50_onewidth_runner_two_widths":(0.50,1.0,2.0)}; _STATE_KEY="sieve3_v2_desc_channel_ordered"; minimal_roi={"0":100.0}; stoploss=-0.05; use_custom_stoploss=True; use_exit_signal=True; position_adjustment_enable=True; trailing_stop=False
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


    def leverage(self,pair:str,current_time:datetime,current_rate:float,proposed_leverage:float,max_leverage:float,entry_tag:str|None,side:str,**kwargs:Any)->float: return 1.0
    def informative_pairs(self)->list[tuple[str,str]]: return []
    def populate_indicators(self,dataframe:DataFrame,metadata:dict)->DataFrame:
        dataframe=add_pattern_geometry_v2(dataframe,timeframe=self.timeframe,output_slots=1,include_triangle_patterns=False,include_wedge_patterns=False,include_compression_patterns=False,include_rectangle_patterns=False,include_ascending_channel_patterns=False,include_descending_channel_patterns=True,min_pattern_bars=int(self.min_pattern_bars),max_pattern_bars=int(self.max_pattern_bars),compression_max_width_atr=float(self.compression_max_width_atr),squeeze_active_width_atr=float(self.squeeze_active_width_atr),local_narrowing_min_ratio=float(self.local_narrowing_min_ratio),min_line_score=float(self.min_line_score),min_containment=float(self.min_containment),max_recent_touch_age_bars=int(self.max_recent_touch_age_bars),channel_min_pattern_bars=int(self.channel_min_pattern_bars),channel_max_pattern_bars=int(self.channel_max_pattern_bars),channel_min_quality=float(self.channel_min_quality),channel_min_containment=float(self.channel_min_containment),channel_near_boundary_atr_mult=float(self.channel_near_boundary_atr_mult),channel_breakout_atr_mult=float(self.channel_breakout_atr_mult),channel_lifecycle_confirm_break_bars=int(self.channel_lifecycle_confirm_break_bars),output_prefix="pg2"); return dataframe
    def _common_guards(self,dataframe:DataFrame)->Series:
        guard=pd.Series(True,index=dataframe.index,dtype="bool"); close=_num(dataframe,"close")
        if bool(self.use_volume_guard): volume=_num(dataframe,"volume").clip(lower=0.0); window=int(self.volume_guard_window); baseline=volume.shift(1).rolling(window,min_periods=max(2,window//3)).mean().replace(0.0,np.nan); guard &= volume.ge(baseline.mul(float(self.volume_ratio_min)))
        if bool(self.use_pressure_guard) or bool(self.use_accumulation_guard): open_=_num(dataframe,"open"); high=_num(dataframe,"high"); low=_num(dataframe,"low"); volume=_num(dataframe,"volume").clip(lower=0.0).fillna(0.0); candle_range=(high-low).replace(0.0,np.nan); pressure=(( ((close-open_)/candle_range).clip(-1,1).fillna(0)+(((close-low)/candle_range)*2-1).clip(-1,1).fillna(0) )/2).clip(-1,1); directional_volume=(pressure*volume).fillna(0)
        if bool(self.use_pressure_guard): window=int(self.pressure_window); denominator=volume.rolling(window,min_periods=max(2,window//3)).sum().replace(0.0,np.nan); guard &= (directional_volume.rolling(window,min_periods=max(2,window//3)).sum()/denominator).le(-float(self.pressure_min))
        if bool(self.use_accumulation_guard): window=int(self.pressure_window); guard &= directional_volume.rolling(window,min_periods=max(2,window//3)).sum().le(0)
        if bool(self.use_body_direction_guard): guard &= close.lt(_num(dataframe,"open"))
        if bool(self.use_close_direction_guard): guard &= close.lt(close.shift(1))
        return guard.fillna(False)
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        missing = sorted(set(("pg2_descending_channel_lower", "pg2_descending_channel_width_atr")).difference(dataframe.columns))
        if missing:
            raise KeyError(f"{type(self).__name__} entry dataframe is missing required source columns: {missing}")
        _ = metadata
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None
        lower = _num(dataframe, "pg2_descending_channel_lower", np.nan).mul(1.0 - float(self.rail_buffer_pct))
        condition = _bool(dataframe, "pg2_descending_channel_pattern_present") & _num(dataframe, "pg2_descending_channel_indicator_score").ge(float(self.score_min)) & _num(dataframe, "pg2_descending_channel_width_atr", np.nan).le(float(self.width_atr_max)) & _cross_below(_num(dataframe, "close"), lower)
        valid = condition.fillna(False) & dataframe["volume"].gt(0.0) & dataframe["close"].notna()
        dataframe.loc[valid, "enter_short"] = 1
        dataframe.loc[valid, "enter_tag"] = ENTRY_TAG
        return dataframe
    def populate_exit_trend(self,dataframe:DataFrame,metadata:dict)->DataFrame: dataframe["exit_long"]=0; dataframe["exit_short"]=0; dataframe["exit_tag"]=None; return dataframe
    def _state(self,pair:str,trade:Any,current_time:datetime)->dict[str,float]:
        state=trade.get_custom_data(key=self._STATE_KEY)
        if state is not None: return dict(state)
        if self.dp is None: raise RuntimeError("data provider required for ordered target snapshot")
        frame,_=self.dp.get_analyzed_dataframe(pair,self.timeframe); required={"date","pg2_descending_channel_upper","pg2_descending_channel_lower","pg2_descending_channel_indicator_score"}; missing=sorted(required-set(frame.columns))
        if missing: raise KeyError(f"missing exact ordered target columns: {missing}")
        dates=pd.to_datetime(frame["date"],utc=True,errors="raise")+pd.to_timedelta(timeframe_to_minutes(self.timeframe),unit="m"); freeze=pd.Timestamp(getattr(trade,"open_date_utc",None) or current_time); freeze=freeze.tz_localize("UTC") if freeze.tzinfo is None else freeze.tz_convert("UTC"); closed=frame.loc[dates.le(freeze)]
        if closed.empty: raise RuntimeError("no closed entry candle for ordered target")
        row=closed.iloc[-1]; upper=float(row["pg2_descending_channel_upper"]); lower=float(row["pg2_descending_channel_lower"]); width=upper-lower
        if not np.isfinite([upper,lower,width]).all() or width<=0: raise ValueError("invalid entry-frozen descending-channel geometry")
        state={"lower":lower,"width":width}; trade.set_custom_data(key=self._STATE_KEY,value=state); return state
    def _targets(self,pair:str,trade:Any,current_time:datetime)->tuple[float,float,float]:
        fraction,first_mult,runner_mult=self._PLANS[str(self.ordered_plan.value)]; state=self._state(pair,trade,current_time); return fraction,state["lower"]-state["width"]*first_mult,state["lower"]-state["width"]*runner_mult
    def adjust_trade_position(self,trade:Any,current_time:datetime,current_rate:float,current_profit:float,min_stake:float|None,max_stake:float,current_entry_rate:float,current_exit_rate:float,current_entry_profit:float,current_exit_profit:float,**kwargs:Any)->tuple[float,str]|None:
        _=current_profit,min_stake,max_stake,current_entry_rate,current_exit_rate,current_entry_profit,current_exit_profit,kwargs
        if bool(getattr(trade,"has_open_orders",False)): return None
        fraction,first_target,_=self._targets(str(trade.pair),trade,current_time); stake=float(getattr(trade,"stake_amount",0.0) or 0.0)
        if stake <= 0 or first_target >= float(trade.open_rate) * 0.999 or current_rate > first_target: return None
        request=self._partial_fill_request(trade,current_time,fraction,min_stake,"geometry_first_target_partial")
        return (-request,"geometry_first_target_partial") if request is not None else None
    def custom_stoploss(self,pair:str,trade:Any,current_time:datetime,current_rate:float,current_profit:float,after_fill:bool=False,**kwargs:Any)->float|None:
        _=pair,current_time,current_profit,after_fill,kwargs
        if self._partial_fill_complete(trade,current_time,"geometry_first_target_partial"): return stoploss_from_absolute(float(trade.open_rate),current_rate=current_rate,is_short=True,leverage=float(getattr(trade,"leverage",1.0) or 1.0))
        return None
    def custom_exit(self,pair:str,trade:Any,current_time:datetime,current_rate:float,current_profit:float,**kwargs:Any)->str|None:
        _=current_profit,kwargs
        if not self._partial_fill_complete(trade,current_time,"geometry_first_target_partial"): return None
        _,_,runner_target=self._targets(pair,trade,current_time)
        if runner_target<float(trade.open_rate)*0.999 and current_rate<=runner_target: return "geometry_runner_target"
        plan=str(self.runner_weakening_plan.value)
        if plan=="target_only": return None
        frame,_=self.dp.get_analyzed_dataframe(pair,self.timeframe); required={"date","open","close","pg2_descending_channel_indicator_score"}; missing=sorted(required-set(frame.columns))
        if missing: raise KeyError(f"missing exact runner weakening columns: {missing}")
        now=pd.Timestamp(current_time); now=now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC"); opened=pd.Timestamp(trade.open_date_utc); opened=opened.tz_localize("UTC") if opened.tzinfo is None else opened.tz_convert("UTC"); close_times=pd.to_datetime(frame["date"],utc=True,errors="raise")+pd.to_timedelta(timeframe_to_minutes(self.timeframe),unit="m"); frame=frame.loc[close_times.le(now)&close_times.gt(opened)].sort_values("date")
        if len(frame)<2: return None
        previous,last=frame.iloc[-2],frame.iloc[-1]; bullish=float(last["close"])>float(last["open"]); score=float(last["pg2_descending_channel_indicator_score"]); prior_score=float(previous["pg2_descending_channel_indicator_score"]); weakened=score<float(self.score_min) if plan=="bullish_score_below_locked_min" else score<=prior_score*0.5
        return f"runner_weakening_{plan}" if bullish and weakened else None
