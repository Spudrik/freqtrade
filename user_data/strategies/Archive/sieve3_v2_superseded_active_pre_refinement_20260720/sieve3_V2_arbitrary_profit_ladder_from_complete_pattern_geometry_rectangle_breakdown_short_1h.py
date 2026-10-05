"""Arbitrary ordered profit ladders for the frozen 1h rectangle-breakdown short entry."""
from __future__ import annotations
from datetime import datetime
from typing import Any
import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from freqtrade.constants import NON_OPEN_EXCHANGE_STATES
from freqtrade.strategy import CategoricalParameter, IStrategy
from user_data.Indicators.pattern_geometry_v2 import add_pattern_geometry_v2

ENTRY_MODE, ENTRY_TAG, SIDE, TIMEFRAME = "entry_geometry_rectangle_breakdown_short_1h", "geometry_rectangle_breakdown_short_1h", "short", "1h"
ENTRY_LOCK_STATUS = "current executable defaults locked; promoted historical snapshots conflict"
LADDER_PLANS = {"early_1_2_4": (0.01, 0.02, 0.04, 0.33, 0.50), "balanced_2_4_6": (0.02, 0.04, 0.06, 0.33, 0.50), "wide_3_6_9": (0.03, 0.06, 0.09, 0.25, 0.50), "front_loaded_2_3_5": (0.02, 0.03, 0.05, 0.50, 0.50)}

def tagged_exit_parameter(p: Any) -> Any: setattr(p, "batch_tags", ("family:exits", "mode:sieve3_exit")); return p
def _num(f: DataFrame, c: str, _default: float | Series=...) -> Series:
    if c in f.columns:
        value = f[c]
    elif _default is not ...:
        value = pd.Series(_default, index=f.index)
    else:
        raise KeyError(c)
    return pd.to_numeric(value, errors='coerce').replace([np.inf, -np.inf], np.nan)
def _bool(f: DataFrame, c: str) -> Series:
    return pd.Series(f[c], index=f.index).astype("boolean").fillna(False).astype(bool)
def _cross_below(v: Series, level: Series) -> Series: return v.lt(level) & v.shift(1).ge(level.shift(1))

SIEVE_STAGE = 'sieve3'
ENTRY_SOURCE_STAGE = 'sieve2'
RESEARCH_PATH = 'sieve3_exit_arbitrary_profit_ladder'
EXIT_HYPOTHESIS = 'Compare ordered fixed-profit scale-out plans while preserving the locked entry.'
class Sieve3V2ArbitraryProfitLadderFromCompletePatternGeometryRectangleBreakdownShort1H(IStrategy):
    SIEVE_STAGE = 'sieve3'
    ENTRY_SOURCE_STAGE = 'sieve2'
    RESEARCH_PATH = 'sieve3_exit_arbitrary_profit_ladder'
    EXIT_HYPOTHESIS = 'Compare ordered fixed-profit scale-out plans while preserving the locked entry.'
    ACTIVE_SELL_PARAMS = ('ladder_plan',)
    INTERFACE_VERSION, timeframe, startup_candle_count, process_only_new_candles, can_short = 3, TIMEFRAME, 180, True, True
    minimal_roi, stoploss, use_exit_signal, use_custom_stoploss = {"0": 100.0}, -0.03, True, False
    position_adjustment_enable, max_entry_position_adjustment, trailing_stop = True, 0, False
    exit_profit_only, ignore_roi_if_entry_signal = False, False
    # use_volume_guard = False (locked off; gated entry branch removed).
    # volume_guard_window = 48 (inactive after its locked entry gate/mode was removed).
    # volume_ratio_min = 1. (inactive after its locked entry gate/mode was removed).
    use_pressure_guard = True
    pressure_window = 24
    pressure_min = .15
    # use_accumulation_guard = False (locked off; gated entry branch removed).
    # use_body_direction_guard = False (locked off; gated entry branch removed).
    # use_close_direction_guard = False (locked off; gated entry branch removed).
    min_pattern_bars = 12
    max_pattern_bars = 72
    compression_max_width_atr = 2.
    squeeze_active_width_atr = 2.
    local_narrowing_min_ratio = .1
    min_line_score = .7
    min_containment = .88
    max_recent_touch_age_bars = 12
    channel_min_pattern_bars = 18
    channel_max_pattern_bars = 96
    channel_min_quality = .82
    channel_min_containment = .68
    channel_near_boundary_atr_mult = .7
    channel_breakout_atr_mult = .35
    channel_lifecycle_confirm_break_bars = 2
    score_min = .65
    width_atr_max = 6.
    rail_buffer_pct = .004
    ladder_plan = tagged_exit_parameter(CategoricalParameter(tuple(LADDER_PLANS), default="balanced_2_4_6", space="sell", optimize=True, load=True))

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float, max_leverage: float, entry_tag: str | None, side: str, **kwargs: Any) -> float: return 1.0
    def informative_pairs(self) -> list[tuple[str,str]]: return []
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe=add_pattern_geometry_v2(dataframe,timeframe=self.timeframe,output_slots=1,include_triangle_patterns=False,include_wedge_patterns=False,include_compression_patterns=False,include_rectangle_patterns=True,include_ascending_channel_patterns=False,include_descending_channel_patterns=False,min_pattern_bars=int(self.min_pattern_bars),max_pattern_bars=int(self.max_pattern_bars),compression_max_width_atr=float(self.compression_max_width_atr),squeeze_active_width_atr=float(self.squeeze_active_width_atr),local_narrowing_min_ratio=float(self.local_narrowing_min_ratio),min_line_score=float(self.min_line_score),min_containment=float(self.min_containment),max_recent_touch_age_bars=int(self.max_recent_touch_age_bars),channel_min_pattern_bars=int(self.channel_min_pattern_bars),channel_max_pattern_bars=int(self.channel_max_pattern_bars),channel_min_quality=float(self.channel_min_quality),channel_min_containment=float(self.channel_min_containment),channel_near_boundary_atr_mult=float(self.channel_near_boundary_atr_mult),channel_breakout_atr_mult=float(self.channel_breakout_atr_mult),channel_lifecycle_confirm_break_bars=int(self.channel_lifecycle_confirm_break_bars),output_prefix="pg2")
        return dataframe
    def _common_guards(self,d:DataFrame)->Series:
        g=pd.Series(True,index=d.index,dtype="bool"); c=_num(d,"close")
        if bool(self.use_pressure_guard):
            o,h,l=_num(d,"open"),_num(d,"high"),_num(d,"low"); v=_num(d,"volume").clip(lower=0.).fillna(0.); r=(h-l).replace(0.,np.nan); p=((c-o)/r).clip(-1.,1.).fillna(0.).add((((c-l)/r)*2.-1.).clip(-1.,1.).fillna(0.)).div(2.); dv=(p*v).fillna(0.)
        if bool(self.use_pressure_guard):
            w=int(self.pressure_window); g&=(dv.rolling(w,min_periods=max(2,w//3)).sum()/v.rolling(w,min_periods=max(2,w//3)).sum().replace(0.,np.nan)).le(-float(self.pressure_min))
        return g.fillna(False)
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        missing = sorted(set(('pg2_rectangle_lower', 'pg2_rectangle_width_atr')).difference(dataframe.columns))
        if missing:
            raise KeyError(f'{type(self).__name__} entry dataframe is missing required source columns: {missing}')
        dataframe['enter_long'], dataframe['enter_short'], dataframe['enter_tag'] = (0, 0, None)
        lower = _num(dataframe, 'pg2_rectangle_lower', np.nan).mul(1.0 - float(self.rail_buffer_pct))
        c = _bool(dataframe, 'pg2_rectangle_pattern_present') & _num(dataframe, 'pg2_rectangle_indicator_score').ge(float(self.score_min)) & _num(dataframe, 'pg2_rectangle_width_atr', np.nan).le(float(self.width_atr_max)) & _cross_below(_num(dataframe, 'close'), lower)
        c &= self._common_guards(dataframe)
        v = c.fillna(False) & dataframe['volume'].gt(0.0) & dataframe['close'].notna()
        dataframe.loc[v, 'enter_short'], dataframe.loc[v, 'enter_tag'] = (1, ENTRY_TAG)
        return dataframe
    def populate_exit_trend(self,dataframe:DataFrame,metadata:dict)->DataFrame: dataframe["exit_long"],dataframe["exit_short"],dataframe["exit_tag"]=0,0,None; return dataframe
    def _partial_identity(self) -> tuple[str, tuple[str, ...]]:
        return str(self.ladder_plan.value), ("profit_ladder_partial_1", "profit_ladder_partial_2")

    @staticmethod
    def _partial_order_key(order: Any) -> str:
        identifier = getattr(order, "order_id", None) or getattr(order, "id", None)
        if identifier is not None:
            return str(identifier)
        return "|".join(
            (
                str(getattr(order, "ft_order_tag", None) or ""),
                str(getattr(order, "order_date_utc", None) or getattr(order, "order_date", None) or ""),
                str(getattr(order, "safe_amount", 0.0) or 0.0),
                str(getattr(order, "safe_price", 0.0) or 0.0),
            )
        )

    @staticmethod
    def _partial_time_key(value: Any) -> str:
        return value.isoformat() if hasattr(value, "isoformat") else str(value)

    def _partial_sync(
        self,
        trade: Any,
        current_time: Any,
        plan: str,
        tags: tuple[str, ...],
        order: Any | None = None,
    ) -> dict[str, Any]:
        key = f"sieve3_v2_partial:{type(self).__name__}"
        stored = trade.get_custom_data(key=key)
        if stored is None:
            state = {
                "version": 1,
                "plan": plan,
                "stages": [
                    {
                        "tag": tag,
                        "target_stake": None,
                        "credited_stake": 0.0,
                        "fills": {},
                        "complete": False,
                        "skipped": False,
                        "skip_reason": None,
                        "pending": False,
                        "requested_at": None,
                    }
                    for tag in tags
                ],
            }
        else:
            if not isinstance(stored, dict) or stored.get("version") != 1:
                raise ValueError("partial ladder state version mismatch")
            state = dict(stored)
            state["stages"] = [
                {**dict(stage), "fills": dict(stage.get("fills") or {})}
                for stage in state.get("stages", ())
            ]
            state_tags = tuple(str(stage.get("tag") or "") for stage in state["stages"])
            if state.get("plan") != plan or state_tags != tags:
                raise ValueError("partial ladder state does not match the selected plan")

        orders = list(getattr(trade, "orders", ()) or ())
        if order is not None:
            order_key = self._partial_order_key(order)
            if not any(self._partial_order_key(candidate) == order_key for candidate in orders):
                orders.append(order)
        exit_side = getattr(trade, "exit_side", None)
        leverage = float(getattr(trade, "leverage", 1.0) or 1.0)
        now_key = self._partial_time_key(current_time)

        for stage in state["stages"]:
            matching = [
                candidate
                for candidate in orders
                if getattr(candidate, "ft_order_side", None) == exit_side
                and str(getattr(candidate, "ft_order_tag", None) or "") == stage["tag"]
            ]
            fills = dict(stage.get("fills") or {})
            requested_stakes: list[float] = []
            open_order = False
            terminal_order = False
            fully_filled = False
            for candidate in matching:
                candidate_key = self._partial_order_key(candidate)
                amount = float(getattr(candidate, "safe_amount", 0.0) or 0.0)
                filled = float(getattr(candidate, "safe_filled", 0.0) or 0.0)
                price = float(getattr(candidate, "safe_price", 0.0) or 0.0)
                if amount > 0.0 and price > 0.0:
                    requested_stakes.append(amount * price / leverage)
                filled_stake = filled * price / leverage if filled > 0.0 and price > 0.0 else 0.0
                if filled_stake > float(fills.get(candidate_key, 0.0) or 0.0):
                    fills[candidate_key] = filled_stake
                status = str(getattr(candidate, "status", None) or "").casefold()
                is_open = bool(getattr(candidate, "ft_is_open", False)) and status not in NON_OPEN_EXCHANGE_STATES
                open_order = open_order or is_open
                if not is_open:
                    terminal_order = True
                    remaining = float(getattr(candidate, "safe_remaining", max(0.0, amount - filled)) or 0.0)
                    tolerance = max(1e-12, amount * 1e-6)
                    fully_filled = fully_filled or (
                        filled > 0.0
                        and (
                            (amount > 0.0 and filled >= amount - tolerance)
                            or remaining <= tolerance
                        )
                    )

            stage["fills"] = fills
            stage["credited_stake"] = sum(float(value) for value in fills.values())
            if stage.get("target_stake") is None and requested_stakes:
                stage["target_stake"] = requested_stakes[0]
            target = float(stage.get("target_stake") or 0.0)
            complete = bool(stage.get("complete")) or fully_filled
            if target > 0.0:
                complete = complete or stage["credited_stake"] >= target - max(1e-8, target * 1e-6)
            stage["complete"] = complete
            if complete:
                stage["pending"] = False
            elif open_order:
                stage["pending"] = True
            elif terminal_order or stage.get("requested_at") != now_key:
                stage["pending"] = False

        if stored != state:
            trade.set_custom_data(key=key, value=state)
        return state

    def _partial_request(
        self,
        trade: Any,
        current_time: Any,
        state: dict[str, Any],
        stage_index: int,
        current_stake: float,
        fraction: float,
        min_stake: float | None = None,
        enforce_minimum: bool = False,
        preserve_remainder_minimum: bool = False,
    ) -> float | None:
        stage = state["stages"][stage_index]
        if stage["complete"] or stage.get("skipped") or stage["pending"]:
            return None
        if stage.get("target_stake") is None:
            stage["target_stake"] = current_stake * fraction
        target = float(stage["target_stake"])
        remaining = max(0.0, target - float(stage.get("credited_stake") or 0.0))
        if remaining <= max(1e-8, target * 1e-6):
            stage["complete"] = True
            trade.set_custom_data(key=f"sieve3_v2_partial:{type(self).__name__}", value=state)
            return None
        request = min(current_stake, remaining)
        skip_reason = None
        if request <= 0.0:
            skip_reason = "nonpositive_reduction"
        elif request >= current_stake:
            skip_reason = "partial_would_close_position"
        elif enforce_minimum and min_stake is not None and request < float(min_stake):
            skip_reason = "reduction_below_min_stake"
        elif (
            preserve_remainder_minimum
            and min_stake is not None
            and 0.0 < current_stake - request < float(min_stake)
        ):
            skip_reason = "remainder_below_min_stake"
        if skip_reason is not None:
            stage["skipped"] = True
            stage["skip_reason"] = skip_reason
            stage["pending"] = False
            trade.set_custom_data(key=f"sieve3_v2_partial:{type(self).__name__}", value=state)
            return None
        stage["pending"] = True
        stage["requested_at"] = self._partial_time_key(current_time)
        trade.set_custom_data(key=f"sieve3_v2_partial:{type(self).__name__}", value=state)
        return request

    @staticmethod
    def _partials_complete(state: dict[str, Any]) -> bool:
        return bool(state["stages"]) and all(
            bool(stage["complete"]) or bool(stage.get("skipped"))
            for stage in state["stages"]
        )

    def order_filled(
        self,
        pair: str,
        trade: Any,
        order: Any,
        current_time: datetime,
        **kwargs: Any,
    ) -> None:
        _ = (pair, kwargs)
        if getattr(order, "ft_order_side", None) != getattr(trade, "exit_side", None):
            return None
        plan, tags = self._partial_identity()
        self._partial_sync(trade, current_time, plan, tags, order)
        return None


    def adjust_trade_position(
        self, trade: Any, current_time: datetime, current_rate: float, current_profit: float,
        min_stake: float | None, max_stake: float, current_entry_rate: float,
        current_exit_rate: float, current_entry_profit: float, current_exit_profit: float,
        **kwargs: Any,
    ) -> tuple[float, str] | None:
        _ = (current_rate, min_stake, max_stake, current_entry_rate, current_exit_rate, current_entry_profit, current_exit_profit, kwargs)
        plan, tags = self._partial_identity()
        state = self._partial_sync(trade, current_time, plan, tags)
        if bool(getattr(trade, "has_open_orders", False)):
            return None
        stage_index = next((index for index, stage in enumerate(state["stages"]) if not stage["complete"] and not stage.get("skipped")), None)
        if stage_index is None:
            return None
        first_trigger, second_trigger, _, first_fraction, second_fraction = LADDER_PLANS[plan]
        trigger, fraction = ((first_trigger, first_fraction) if stage_index == 0 else (second_trigger, second_fraction))
        if current_profit < trigger:
            return None
        stake = float(trade.stake_amount)
        request = self._partial_request(
            trade, current_time, state, stage_index, stake, fraction,
            min_stake, enforce_minimum=True, preserve_remainder_minimum=True,
        )
        return (-request, tags[stage_index]) if request is not None else None
    def custom_exit(
        self, pair: str, trade: Any, current_time: datetime,
        current_rate: float, current_profit: float, **kwargs: Any,
    ) -> str | None:
        _ = (pair, current_rate, kwargs)
        plan, tags = self._partial_identity()
        state = self._partial_sync(trade, current_time, plan, tags)
        final_target = LADDER_PLANS[plan][2]
        return "profit_ladder_final" if self._partials_complete(state) and current_profit >= final_target else None

ACTIVE_SELL_PARAMS = ('ladder_plan',)
