"""Four isolated PAPER adapters for reviewed Sieve3 entry/exit selections.

Entry indicators and exit plans remain inherited from their active Sieve3 sources.
The MTF/BOS and daily-support adapters restore only the tested exit methods
whose later active-source changes altered the validated exit behavior.
PAPER partial stages reconcile filled quantities at entry cost and exchange precision.
"""
from __future__ import annotations

from datetime import datetime
import math
from typing import Any, Mapping

from freqtrade.enums import RunMode
from freqtrade.exchange import amount_to_contract_precision
from freqtrade.strategy import stoploss_from_absolute

from user_data.strategies.sieve3_V2_target_partial_invalidation_remainder_from_pivot_midrange_reject_short_1h import (
    Sieve3V2TargetPartialInvalidationRemainderFromPivotMidrangeRejectShort1h as _PivotBase,
)
from user_data.strategies.sieve3_V2_profit_ladder_three_stage_ratchet_from_mtf_confluence_d1_vp_bos_4h_retest_short import (
    Sieve3V2ProfitLadderThreeStageRatchetFromMtfConfluenceD1VpBos4hRetestShort as _VpBosBase,
)
from user_data.strategies.sieve3_V2_target_partial_invalidation_remainder_from_novel_mtf_d1_support_hold_h4_break_1h_higher_low_break_long import (
    Sieve3V2TargetPartialInvalidationRemainderFromNovelMtfD1SupportHoldH4Break1HHigherLowBreakLong as _SupportBase,
)
from user_data.strategies.sieve3_V2_entry_target_full_or_zone_reversal_from_mtfx_h4_vp_lvn_traverse_long_1h_breakout import (
    Sieve3V2EntryTargetFullOrZoneReversalFromMtfxH4VpLvnTraverseLong1hBreakout as _LvnBase,
)


class _PaperSieveLockMixin:
    """Freeze selected source parameters after Freqtrade's actual load step."""
    MAX_STAKE_EQUITY_FRACTION = 0.02
    PAPER_PAIRS = frozenset({"BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"})
    PAPER_BOT_NAME: str

    def ft_load_hyper_params(self, hyperopt: bool = False) -> None:
        super().ft_load_hyper_params(hyperopt)
        self._assert_paper_config()
        self._assert_selected_parameters()
        if not math.isclose(float(self.stoploss), self.EMERGENCY_STOP_FLOOR, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"{self.__class__.__name__}: emergency stoploss differs from selected plan floor")

    def _assert_selected_parameters(self) -> None:
        for space, selected in (("buy", self.LOCKED_BUY_PARAMS), ("sell", self.LOCKED_SELL_PARAMS)):
            for name, expected in selected.items():
                parameter = getattr(self, name, None)
                if parameter is None:
                    inert = getattr(self, "INERT_BUY_PARAMS", {}) if space == "buy" else {}
                    if inert.get(name) == expected:
                        continue
                    raise RuntimeError(f"{self.__class__.__name__}: selected {space} parameter {name} is absent from its source")
                actual = getattr(parameter, "value", parameter)
                if actual != expected:
                    raise RuntimeError(
                        f"{self.__class__.__name__}: loaded {space} parameter {name}={actual!r}; "
                        f"selected frozen value is {expected!r}"
                    )

    def _assert_paper_config(self) -> None:
        config = self.config
        exchange = config.get("exchange", {})
        api = config.get("api_server", {})
        credential_keys = ("key", "secret", "password", "privateKey", "private_key")
        if (not self.PAPER_BOT_NAME
                or config.get("dry_run") is not True
                or config.get("runmode") != RunMode.DRY_RUN
                or config.get("trading_mode") != "futures"
                or config.get("margin_mode") != "isolated"
                or exchange.get("name") != "binance"
                or any(exchange.get(key) for key in credential_keys)
                or any(config.get(key) for key in ("private_key", "privateKey", "exchange_key", "exchange_secret"))
                or set(exchange.get("pair_whitelist", ())) != self.PAPER_PAIRS
                or config.get("max_open_trades") != 3
                or config.get("bot_name") != self.PAPER_BOT_NAME
                or config.get("strategy") != self.__class__.__name__
                or config.get("force_entry_enable") is not False
                or bool(api.get("enabled"))):
            raise RuntimeError(f"{self.__class__.__name__}: requires its exact isolated PAPER-only account config")

    def bot_start(self, **kwargs: Any) -> None:
        self._assert_paper_config()
        parent = getattr(super(), "bot_start", None)
        if callable(parent):
            parent(**kwargs)
        self._assert_paper_config()

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: str | None,
                 side: str, **kwargs: Any) -> float:
        _ = (pair, current_time, current_rate, proposed_leverage, entry_tag, side, kwargs)
        if float(max_leverage) < 1.0:
            raise RuntimeError("The selected paper cohort requires 1x leverage")
        return 1.0

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: float | None, max_stake: float,
                            leverage: float, entry_tag: str | None, side: str,
                            **kwargs: Any) -> float:
        _ = (pair, current_time, current_rate, proposed_stake, leverage, entry_tag, side, kwargs)
        equity = float(self.wallets.get_total_stake_amount())
        maximum = float(max_stake)
        if not math.isfinite(equity) or equity <= 0 or not math.isfinite(maximum) or maximum <= 0:
            raise RuntimeError("Paper Sieve sizing requires finite positive equity and max stake")
        stake = min(equity * self.MAX_STAKE_EQUITY_FRACTION, maximum)
        if min_stake is not None and stake < float(min_stake):
            return 0.0
        return stake


_PIVOT_BUY = {
    "pivot_strength": 3, "min_prominence_atr": 0.25, "min_pivot_spacing_bars": 8,
    "min_pivot_distance_atr": 0.42, "reclaim_buffer_pct": 0.001, "score_min": 0.24,
    "range_atr_mult": 3.37, "freshness_max_bars": 6,
}
_VP_BOS_BUY = {
    "use_sieve2_vp_guard": True, "sieve2_vp_guard_mode": "score_or_context",
    "sieve2_vp_window": 96, "sieve2_vp_bins": 36, "sieve2_vp_score_min": 0.15,
    "sieve2_vp_context_min": 0.15, "use_sieve2_market_guard": True,
    "sieve2_market_guard_mode": "pressure_or_trend", "sieve2_market_window": 24,
    "sieve2_market_pressure_min": 0.03, "sieve2_market_trend_min": 0.15,
    "sieve2_rs_benchmark_pair": "BTC/USDT:USDT", "sieve2_rs_score_min": 0.3,
    "min_htf_gates": 2, "min_ltf_gates": 2, "min_total_gates": 2,
    "htf_lookback": 24, "ltf_lookback": 24, "recent_htf_bars": 5,
    "level_buffer_pct": 0.0, "retest_buffer_pct": 0.012, "vp_score_min": 0.5,
    "vp_context_min": 0.5, "ltf_volume_ratio_min": 1.5, "ltf_pressure_min": 0.18,
    "use_ltf_volume_guard": False, "use_ltf_pressure_guard": True,
}
_SUPPORT_BUY = {
    "context_lookback": 22, "h4_lookback": 23, "exec_lookback": 4,
    "body_ratio_min": 0.47, "range_ratio_min": 1.96, "volume_ratio_min": 1.71,
    "retest_tolerance": 0.037, "close_follow_min": 0.29,
    "require_h4_confirm": True, "require_volume_confirm": False,
}
_LVN_BUY = {
    "use_sieve2_vp_guard": False, "sieve2_vp_guard_mode": "score_or_context",
    "sieve2_vp_window": 96, "sieve2_vp_bins": 36, "sieve2_vp_score_min": 0.25,
    "sieve2_vp_context_min": 0.28, "use_sieve2_market_guard": False,
    "sieve2_market_guard_mode": "pressure_or_trend", "sieve2_market_window": 24,
    "sieve2_market_pressure_min": 0.07, "sieve2_market_trend_min": 0.25,
    "sieve2_rs_benchmark_pair": "BTC/USDT:USDT", "sieve2_rs_score_min": 0.45,
    "score_min": 0.82, "htf_recent_bars": 3, "level_buffer_pct": 0.002,
    "local_window": 24, "use_ltf_volume_guard": True, "volume_ratio_min": 1.6,
    "use_ltf_pressure_guard": False, "pressure_min": 0.07,
}


class _PaperPartialFillMixin:
    """Reconcile named PAPER stages in the entry-cost units used by adjustments."""

    def _focused_ensure_state(self, pair: str, trade: Any, current_time: Any,
                              order: Any | None = None) -> dict[str, Any] | None:
        state = super()._focused_ensure_state(pair, trade, current_time, order)
        if state is not None:
            before = repr(state)
            self._focused_sync_partial_stages(trade, state, current_time)
            if repr(state) != before:
                self._focused_save_state(trade, state)
        return state

    def _paper_rounded_stake(self, trade: Any, state: Mapping[str, Any], stake: float) -> float:
        entry_unit_stake = float(state['entry_rate']) / float(trade.leverage)
        return amount_to_contract_precision(
            stake / entry_unit_stake, trade.amount_precision,
            trade.precision_mode, trade.contract_size,
        ) * entry_unit_stake

    def _focused_sync_partial_stages(self, trade: Any, state: dict[str, Any], current_time: Any) -> None:
        orders = tuple(trade.select_filled_or_open_orders())
        now = self._focused_utc(current_time)
        entry_unit_stake = float(state['entry_rate']) / float(trade.leverage)
        for name, stage in state.get('stages', {}).items():
            matching = [order for order in orders
                        if order.ft_order_side == trade.exit_side
                        and order.ft_order_tag == stage['tag']]
            # Never use exit proceeds or order amount to infer the filled quantity.
            credited = sum(max(0.0, float(order.safe_filled)) * entry_unit_stake
                           for order in matching)
            stage['credited_stake'] = credited
            target = float(stage.get('target_stake') or 0.0)
            rounded_target = self._paper_rounded_stake(trade, state, target)
            tolerance = max(1e-9, target * 1e-9)
            if any(order.ft_is_open for order in matching):
                stage['status'] = 'requested'
                stage['filled_at'] = None
                state['phase'] = f'{name}_pending'
            elif rounded_target > 0.0 and credited >= rounded_target - tolerance:
                stage['status'] = 'filled'
                times = [self._focused_utc(order.order_filled_utc) for order in matching]
                latest = max((value for value in times if value is not None), default=None)
                stage['filled_at'] = latest.isoformat() if latest is not None else stage.get('filled_at')
                stage['requested_at'] = None
                state['phase'] = 'remainder' if name == 'stage_1' else 'runner'
            elif stage['status'] == 'requested' and self._focused_utc(stage.get('requested_at')) == now:
                # The callback may precede insertion of its just-requested order.
                continue
            else:
                previous = stage['status']
                stage['status'] = 'ready'
                stage['requested_at'] = None
                stage['filled_at'] = None
                if matching or previous != 'ready':
                    state['phase'] = f'{name}_retry'

    def adjust_trade_position(self, trade: Any, current_time: datetime, current_rate: float,
                              current_profit: float, min_stake: float | None, max_stake: float,
                              current_entry_rate: float, current_exit_rate: float,
                              current_entry_profit: float, current_exit_profit: float,
                              **kwargs: Any) -> float | None | tuple[float | None, str | None]:
        _, state, _, decision = self._focused_context(
            str(trade.pair), trade, current_time, current_rate, current_profit)
        if not state or trade.has_open_orders or decision['action'] != 'partial' or not decision.get('stage'):
            return None
        stage_name = str(decision['stage'])
        stage = state['stages'][stage_name]
        if stage['status'] != 'ready':
            return None
        current_stake = self._focused_float(trade.stake_amount)
        initial_stake = self._focused_float(state.get('initial_stake'))
        if current_stake is None or initial_stake is None or current_stake <= 0.0:
            return None
        target = self._focused_float(stage.get('target_stake'))
        request = min(current_stake, initial_stake * float(decision['fraction'])
                      if target is None else max(0.0, target - float(stage['credited_stake'])))
        if min_stake is not None and 0.0 < current_stake - request < float(min_stake):
            request = current_stake - float(min_stake)
        if request <= 0.0 or request >= current_stake or self._paper_rounded_stake(trade, state, request) <= 0.0:
            return None
        if target is None:
            stage['target_stake'] = request
        stage['status'] = 'requested'
        stage['requested_at'] = self._focused_utc(current_time).isoformat()
        state['phase'] = f'{stage_name}_pending'
        self._focused_save_state(trade, state)
        return (-request, str(stage['tag']))

    def custom_stoploss(self, pair: str, trade: Any, current_time: datetime, current_rate: float,
                        current_profit: float, after_fill: bool, **kwargs: Any) -> float | None:
        plan, state, _, decision = self._focused_context(pair, trade, current_time, current_rate, current_profit)
        if not state or trade.has_open_orders or decision['action'] == 'full' or self._focused_requested_stage(state) is not None:
            return None
        stop_price = self._focused_stop_overlay(plan, state, current_profit)
        persisted = self._focused_float(state.get('stop_price'))
        # A repaired legacy stage must not loosen protection already installed.
        for existing in (persisted, self._focused_float(getattr(trade, 'stop_loss', None))):
            if existing is not None and existing > 0.0:
                stop_price = self._focused_tighter_stop(str(state['side']), existing, stop_price)
        if state['side'] == 'short' and stop_price <= current_rate or state['side'] == 'long' and stop_price >= current_rate:
            return None
        if persisted is None or not math.isclose(stop_price, persisted, rel_tol=0.0, abs_tol=1e-12):
            state['stop_price'] = stop_price
            self._focused_save_state(trade, state)
        return stoploss_from_absolute(stop_price, current_rate=current_rate,
                                      is_short=state['side'] == 'short', leverage=float(trade.leverage))


class PaperSievePivotPartial(_PaperPartialFillMixin, _PaperSieveLockMixin, _PivotBase):
    """1h pivot-midrange rejection short with selected partial invalidation exit."""
    PAPER_BOT_NAME = "paper_sieve_pivot_partial"
    EMERGENCY_STOP_FLOOR = -0.03
    stoploss = EMERGENCY_STOP_FLOOR
    buy_params = dict(_PIVOT_BUY)
    LOCKED_BUY_PARAMS = dict(_PIVOT_BUY)
    # Source documents these two retained Sieve2 fields as inert legacy metadata.
    INERT_BUY_PARAMS = {"score_min": 0.24, "freshness_max_bars": 6}
    sell_params = {
        "exit_plan": "touch_partial", "invalidation_band_quarter_percent": 0,
        "invalidation_confirmations": 1, "partial_five_percent_units": 1,
        "stop_after_partial": "unchanged", "target_band_quarter_percent": 11,
    }
    LOCKED_SELL_PARAMS = dict(sell_params)


class PaperSieveD1VpBosShort(_PaperPartialFillMixin, _PaperSieveLockMixin, _VpBosBase):
    """4h D1-volume-profile/BOS retest short with frozen tested exit-state behavior."""
    PAPER_BOT_NAME = "paper_sieve_d1_vp_bos_short"
    EMERGENCY_STOP_FLOOR = -0.02
    stoploss = EMERGENCY_STOP_FLOOR
    buy_params = dict(_VP_BOS_BUY)
    LOCKED_BUY_PARAMS = dict(_VP_BOS_BUY)
    INERT_BUY_PARAMS = {
        "use_sieve2_vp_guard": True, "sieve2_vp_guard_mode": "score_or_context",
        "sieve2_vp_window": 96, "sieve2_vp_bins": 36, "sieve2_vp_score_min": 0.15,
        "sieve2_vp_context_min": 0.15, "use_sieve2_market_guard": True,
        "sieve2_market_guard_mode": "pressure_or_trend", "sieve2_market_window": 24,
        "sieve2_market_pressure_min": 0.03, "sieve2_market_trend_min": 0.15,
        "sieve2_rs_benchmark_pair": "BTC/USDT:USDT", "sieve2_rs_score_min": 0.3,
        "ltf_volume_ratio_min": 1.5, "use_ltf_volume_guard": False,
        "use_ltf_pressure_guard": True,
    }
    sell_params = {
        "target_1_percent": 2, "target_gap_half_percent_units": 5,
        "partial_1_five_percent_units": 2, "partial_2_five_percent_units": 2,
        "hard_stop_percent": 2,
    }
    LOCKED_SELL_PARAMS = dict(sell_params)

    def _focused_state(self, trade: Any) -> dict[str, Any] | None:
        state = trade.get_custom_data(key=self.FOCUSED_STATE_KEY)
        if state is None:
            return None
        if not isinstance(state, dict):
            raise ValueError('focused exit trade state must be a mapping')
        if state.get('version') != self.FOCUSED_STATE_VERSION:
            raise ValueError('focused exit trade state version mismatch')
        if state.get('contract') != self.FOCUSED_EXIT_CONTRACT:
            raise ValueError('focused exit trade state contract mismatch')
        return dict(state)

    def _focused_save_state(self, trade: Any, state: Mapping[str, Any]) -> None:
        trade.set_custom_data(key=self.FOCUSED_STATE_KEY, value=dict(state))

    def _focused_new_state(self, pair: str, trade: Any, current_time: Any, order: Any | None=None) -> dict[str, Any] | None:
        side = self._focused_side(trade)
        order_rate = self._focused_float(getattr(order, 'safe_price', None)) if order else None
        entry_rate = order_rate or self._focused_float(getattr(trade, 'open_rate', None))
        if entry_rate is None or entry_rate <= 0.0:
            raise ValueError('focused exit runtime requires a positive entry rate')
        freeze_time = (getattr(order, 'order_date_utc', None) if order is not None else None) or (getattr(order, 'order_date', None) if order is not None else None) or getattr(trade, 'open_date_utc', None) or getattr(trade, 'date_entry_fill_utc', None) or current_time
        frame = self._focused_closed_frame(self._focused_analyzed_frame(pair, freeze_time), freeze_time)
        if frame.empty:
            return None
        self._focused_require_columns(frame)
        row = frame.iloc[-1]
        levels: dict[str, float | None] = {}
        target_specs = self.FOCUSED_SOURCE_PROFILE.get('targets', {})
        if isinstance(target_specs, Mapping):
            for slot in target_specs:
                levels[str(slot)] = self._focused_frozen_level(row, self._focused_side_binding('target', side, str(slot)), side, entry_rate, 'target')
        invalidation_binding = self._focused_side_binding('invalidation', side)
        levels['invalidation'] = self._focused_frozen_level(row, invalidation_binding, side, entry_rate, 'invalidation')
        invalidation_level = levels.get('invalidation')
        for slot in ('target_1', 'target_2'):
            target_level = levels.get(slot)
            if target_level is not None and invalidation_level is not None and math.isclose(target_level, invalidation_level, rel_tol=0.0, abs_tol=1e-12):
                raise ValueError(f'resolved {slot} equals the source invalidation level')
        first = levels.get('target_1')
        second = levels.get('target_2')
        if first is not None and second is not None:
            ordered = second < first if side == 'short' else second > first
            if not ordered:
                raise ValueError('entry-frozen target_2 must be beyond target_1')
        selected_plan = str(self.exit_plan)
        plan = self._focused_plan({'plan': selected_plan})
        fill_time = (getattr(order, 'order_filled_utc', None) if order is not None else None) or getattr(trade, 'date_entry_fill_utc', None) or getattr(trade, 'open_date_utc', None) or current_time
        fill_timestamp = self._focused_utc(fill_time)
        snapshot_date = self._focused_utc(row['date'])
        stages = {f'stage_{index + 1}': {'status': 'ready', 'fraction': float(fraction), 'tag': f'focused_partial_stage_{index + 1}', 'requested_at': None, 'target_stake': None, 'credited_stake': 0.0, 'filled_at': None} for index, fraction in enumerate(plan.get('partial_fractions', ()))}
        return {'version': self.FOCUSED_STATE_VERSION, 'contract': self.FOCUSED_EXIT_CONTRACT, 'plan': selected_plan, 'side': side, 'phase': 'pre_target', 'entry_rate': float(entry_rate), 'entry_filled_at': fill_timestamp.isoformat() if fill_timestamp is not None else None, 'entry_snapshot_candle': snapshot_date.isoformat() if snapshot_date is not None else None, 'levels': levels, 'target_states': {slot: {'status': 'available' if value is not None else 'unavailable', 'touched_at': None} for slot, value in levels.items() if slot.startswith('target_')}, 'invalidation_seen': False, 'guard_state': 'unknown', 'trigger_state': 'unknown', 'stages': stages, 'initial_stake': self._focused_float(getattr(trade, 'stake_amount', None)), 'favorable_rate': float(entry_rate), 'stop_price': None, 'terminal_pending': False, 'terminal_tag': None}

class PaperSieveD1SupportBreakLong(_PaperPartialFillMixin, _PaperSieveLockMixin, _SupportBase):
    """D1 support-hold/H4 higher-low 1h breakout with tested candle-state exit."""
    PAPER_BOT_NAME = "paper_sieve_d1_support_break_long"
    EMERGENCY_STOP_FLOOR = -0.03
    stoploss = EMERGENCY_STOP_FLOOR
    buy_params = dict(_SUPPORT_BUY)
    LOCKED_BUY_PARAMS = dict(_SUPPORT_BUY)
    INERT_BUY_PARAMS = {
        "retest_tolerance": 0.037, "require_h4_confirm": True,
        "require_volume_confirm": False,
    }
    sell_params = {
        "exit_plan": "touch_partial", "target_band_quarter_percent": 4,
        "partial_five_percent_units": 1, "stop_after_partial": "entry",
        "invalidation_band_quarter_percent": 0, "invalidation_confirmations": 1,
    }
    LOCKED_SELL_PARAMS = dict(sell_params)

    def _focused_state(self, trade: Any) -> dict[str, Any] | None:
        state = trade.get_custom_data(key=self.FOCUSED_STATE_KEY)
        if state is None:
            return None
        if not isinstance(state, dict):
            raise ValueError('focused exit trade state must be a mapping')
        if state.get('version') != self.FOCUSED_STATE_VERSION:
            raise ValueError('focused exit trade state version mismatch')
        if state.get('contract') != self.FOCUSED_EXIT_CONTRACT:
            raise ValueError('focused exit trade state contract mismatch')
        return dict(state)

    def _focused_save_state(self, trade: Any, state: Mapping[str, Any]) -> None:
        trade.set_custom_data(key=self.FOCUSED_STATE_KEY, value=dict(state))

    def _focused_context(self, pair: str, trade: Any, current_time: Any, current_rate: float,
                         current_profit: float) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        state = self._focused_ensure_state(pair, trade, current_time)
        if state is None:
            return (self._focused_plan(), {}, {}, self._focused_decision())
        before = repr(state)
        if state.get('stages'):
            self._focused_sync_partial_stages(trade, state, current_time)
        plan = self._focused_plan(state)
        frame = self._focused_analyzed_frame(pair, current_time)
        self._focused_require_columns(frame)
        post_entry = self._focused_post_entry(frame, state)
        if not post_entry.empty:
            self._focused_update_favorable(post_entry, state)
        events = self._focused_contract_events(post_entry, state, plan, current_profit)
        if not post_entry.empty:
            state['age_candles'] = int(state.get('age_candles') or 0) + len(post_entry)
            processed = self._focused_utc(post_entry['date'].iat[-1])
            state['last_processed_candle'] = processed.isoformat() if processed is not None else None
        events['age_candles'] = int(state.get('age_candles') or 0)
        events['favorable_move'] = self._focused_favorable_move(state)
        if self._focused_hard_stop_breached(plan, state, float(current_rate)):
            decision = self._focused_decision('full', 'focused_hard_stop')
        else:
            decision = self._focused_contract_decision(state, plan, events, current_profit)
        if decision['action'] == 'hold' and events['age_candles'] >= int(plan['max_hold_candles']):
            decision = self._focused_decision('full', 'focused_max_hold')
        if state.get('terminal_pending') and self._focused_requested_stage(state) is None:
            decision = self._focused_decision('full', str(state.get('terminal_tag') or 'focused_deferred_full'))
        if repr(state) != before:
            self._focused_save_state(trade, state)
        return (plan, state, events, decision)

class PaperSieveH4VpLvnLong(_PaperSieveLockMixin, _LvnBase):
    """H4 volume-profile LVN traverse 1h breakout with selected zone reversal."""
    PAPER_BOT_NAME = "paper_sieve_h4_vp_lvn_long"
    EMERGENCY_STOP_FLOOR = -0.03
    stoploss = EMERGENCY_STOP_FLOOR
    buy_params = dict(_LVN_BUY)
    LOCKED_BUY_PARAMS = dict(_LVN_BUY)
    sell_params = {
        "exit_plan": "zone_reversal", "target_band_quarter_percent": 9,
        "reversal_confirmations": 3, "invalidation_band_quarter_percent": 4,
        "invalidation_confirmations": 1,
    }
    LOCKED_SELL_PARAMS = dict(sell_params)
