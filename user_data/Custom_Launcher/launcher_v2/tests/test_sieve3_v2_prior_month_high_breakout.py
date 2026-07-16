from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd
import pytest

from freqtrade.enums import ExitType
from freqtrade.persistence import Trade
from freqtrade.persistence.usedb_context import FtNoDBContext
from freqtrade.resolvers import StrategyResolver
from freqtrade.strategy import IStrategy
from user_data.strategies.sieve3_exit_breakeven_from_prior_month_high_breakout_long import (
    Sieve3ExitBreakevenFromPriorMonthHighBreakoutLong,
)
from user_data.strategies.sieve3_V2_integrated_from_prior_month_high_breakout_long import (
    BACKTEST_ARCHIVE_SHA256,
    EXIT_PLANS,
    LOCKED_BUY_PARAMS_SHA256,
    PARTIAL_TAGS,
    RESULT_ROW_SHA256,
    SNAPSHOT_PARAMS_SHA256,
    SNAPSHOT_STRATEGY_SHA256,
    STATE_KEY,
    STATE_VERSION,
    TARGET_PROVIDER_CHAINS,
    ExitDecision,
    Sieve3V2IntegratedFromPriorMonthHighBreakoutLong,
)


STRATEGY_PATH = Path(
    "user_data/strategies/sieve3_V2_integrated_from_prior_month_high_breakout_long.py"
)
DATA_PATH = Path("user_data/data/binance/futures/BTC_USDT_USDT-1h-futures.feather")

LOCKED_BUY_DEFAULTS: dict[str, Any] = {
    "use_sieve2_vp_guard": False,
    "sieve2_vp_guard_mode": "score_or_context",
    "sieve2_vp_window": 96,
    "sieve2_vp_bins": 36,
    "sieve2_vp_score_min": 0.25,
    "sieve2_vp_context_min": 0.28,
    "use_sieve2_market_guard": False,
    "sieve2_market_guard_mode": "pressure_or_trend",
    "sieve2_market_window": 24,
    "sieve2_market_pressure_min": 0.07,
    "sieve2_market_trend_min": 0.25,
    "sieve2_rs_benchmark_pair": "BTC/USDT:USDT",
    "sieve2_rs_score_min": 0.45,
    "breakout_buffer_pct": 0.003,
    "reclaim_buffer_pct": 0.006,
    "sweep_buffer_pct": 0.006,
    "zone_near_pct": 0.01,
    "rolling_level_lookback": 72,
    "equal_level_lookback": 72,
    "equal_level_tolerance_pct": 0.006,
    "equal_level_min_touches": 2,
    "confluence_period": "day",
    "avwap_anchor_lookback": 120,
    "avwap_band_mult": 1.25,
    "zone_impulse_window": 36,
    "zone_impulse_atr_min": 0.8,
    "zone_body_fraction_min": 0.55,
    "zone_volume_ratio_min": 1.1,
    "zone_max_age_bars": 72,
    "use_volume_guard": True,
    "volume_window": 24,
    "volume_ratio_min": 1.3,
    "pressure_min": 0.35,
    "use_close_direction_guard": False,
    "vp_window": 96,
    "vp_bins": 48,
    "vp_value_area_pct": 0.7,
    "vp_price_source": "hlc3",
    "vp_smooth_bins": 3,
    "vp_hvn_threshold": 0.7,
    "vp_lvn_threshold": 0.35,
    "vp_pressure_delta_min": 0.05,
    "vp_node_near_pct": 0.01,
    "vp_volume_percentile_min": 0.55,
    "vp_score_window": 48,
    "vp_fast_traverse_atr_mult": 1.2,
    "vp_entry_score_margin": 0.02,
    "use_vp_1h_guard": False,
    "vp_guard_mode": "score_or_context",
    "vp_score_min": 0.25,
    "vp_context_min": 0.28,
    "use_vp_4h_guard": False,
    "vp_4h_window": 48,
    "vp_4h_bins": 36,
    "use_vp_1d_guard": False,
    "vp_1d_window": 30,
    "vp_1d_bins": 36,
}

EXPECTED_PLAN_SIGNATURES = {
    "baseline_3_3": ("baseline", None, None, "touch", "hold", "none", "hold", 0.0, "hold", 0),
    "promotion_2_2": ("baseline", None, None, "touch", "hold", "none", "hold", 0.0, "hold", 0),
    "period_nearest_touch_full": (
        "target_full",
        "period_nearest",
        None,
        "touch",
        "full",
        "none",
        "hold",
        0.0,
        "hold",
        0,
    ),
    "prior_week_rev1_full": (
        "target_full",
        "prior_week",
        None,
        "reversal1",
        "full",
        "none",
        "hold",
        0.0,
        "hold",
        0,
    ),
    "swing48_touch_full": (
        "target_full",
        "swing48",
        None,
        "touch",
        "full",
        "none",
        "hold",
        0.0,
        "hold",
        0,
    ),
    "swing96_rev2of3_full": (
        "target_full",
        "swing96",
        None,
        "reversal2of3",
        "full",
        "none",
        "hold",
        0.0,
        "hold",
        0,
    ),
    "vp_hvn_rev1_full": (
        "target_full",
        "vp_hvn",
        None,
        "reversal1",
        "full",
        "none",
        "hold",
        0.0,
        "hold",
        0,
    ),
    "measured_half_rev1_full": (
        "target_full",
        "measured_half",
        None,
        "reversal1",
        "full",
        "none",
        "hold",
        0.0,
        "hold",
        0,
    ),
    "boundary1_full": (
        "invalidation_full",
        None,
        None,
        "touch",
        "hold",
        "boundary1",
        "full",
        0.0,
        "hold",
        0,
    ),
    "boundary2_full": (
        "invalidation_full",
        None,
        None,
        "touch",
        "hold",
        "boundary2",
        "full",
        0.0,
        "hold",
        0,
    ),
    "retest1_full": (
        "invalidation_full",
        None,
        None,
        "touch",
        "hold",
        "retest1",
        "full",
        0.0,
        "hold",
        0,
    ),
    "retest2_full": (
        "invalidation_full",
        None,
        None,
        "touch",
        "hold",
        "retest2",
        "full",
        0.0,
        "hold",
        0,
    ),
    "boundary1_tighten": (
        "defensive_tighten",
        None,
        None,
        "touch",
        "hold",
        "boundary1",
        "tighten",
        0.0,
        "hold",
        0,
    ),
    "retest1_tighten": (
        "defensive_tighten",
        None,
        None,
        "touch",
        "hold",
        "retest1",
        "tighten",
        0.0,
        "hold",
        0,
    ),
    "period_nearest_touch_p33_be_boundary2": (
        "target_partial",
        "period_nearest",
        None,
        "touch",
        "partial",
        "boundary2",
        "full",
        0.33,
        "breakeven",
        0,
    ),
    "period_nearest_rev1_p50_trail_retest1": (
        "target_partial",
        "period_nearest",
        None,
        "reversal1",
        "partial",
        "retest1",
        "full",
        0.5,
        "trail",
        0,
    ),
    "prior_week_rev1_p33_be_boundary2": (
        "target_partial",
        "prior_week",
        None,
        "reversal1",
        "partial",
        "boundary2",
        "full",
        0.33,
        "breakeven",
        0,
    ),
    "swing48_touch_p33_trail_boundary2": (
        "target_partial",
        "swing48",
        None,
        "touch",
        "partial",
        "boundary2",
        "full",
        0.33,
        "trail",
        0,
    ),
    "swing48_rev1_p50_be_retest1": (
        "target_partial",
        "swing48",
        None,
        "reversal1",
        "partial",
        "retest1",
        "full",
        0.5,
        "breakeven",
        0,
    ),
    "swing96_rev2of3_p33_trail_boundary2": (
        "target_partial",
        "swing96",
        None,
        "reversal2of3",
        "partial",
        "boundary2",
        "full",
        0.33,
        "trail",
        0,
    ),
    "vp_hvn_touch_p33_be_retest1": (
        "target_partial",
        "vp_hvn",
        None,
        "touch",
        "partial",
        "retest1",
        "full",
        0.33,
        "breakeven",
        0,
    ),
    "vp_hvn_rev1_p50_trail_boundary2": (
        "target_partial",
        "vp_hvn",
        None,
        "reversal1",
        "partial",
        "boundary2",
        "full",
        0.5,
        "trail",
        0,
    ),
    "vp_vah_rev1_p33_trail_retest2": (
        "target_partial",
        "vp_vah",
        None,
        "reversal1",
        "partial",
        "retest2",
        "full",
        0.33,
        "trail",
        0,
    ),
    "measured_half_touch_p33_be_boundary2": (
        "target_partial",
        "measured_half",
        None,
        "touch",
        "partial",
        "boundary2",
        "full",
        0.33,
        "breakeven",
        0,
    ),
    "measured_half_rev1_p50_trail_retest1": (
        "target_partial",
        "measured_half",
        None,
        "reversal1",
        "partial",
        "retest1",
        "full",
        0.5,
        "trail",
        0,
    ),
    "nearest_rev1_p33_trail_boundary2": (
        "target_partial",
        "nearest",
        None,
        "reversal1",
        "partial",
        "boundary2",
        "full",
        0.33,
        "trail",
        0,
    ),
    "boundary1_reduce33_period_nearest": (
        "invalidation_reduce",
        "period_nearest",
        None,
        "touch",
        "full",
        "boundary1",
        "partial",
        0.33,
        "target",
        0,
    ),
    "retest1_reduce50_vp_hvn": (
        "invalidation_reduce",
        "vp_hvn",
        None,
        "touch",
        "full",
        "retest1",
        "partial",
        0.5,
        "target",
        0,
    ),
    "boundary2_reduce33_measured_half": (
        "invalidation_reduce",
        "measured_half",
        None,
        "touch",
        "full",
        "boundary2",
        "partial",
        0.33,
        "target",
        0,
    ),
    "period_nearest_touch_tighten_boundary2": (
        "target_tighten",
        "period_nearest",
        None,
        "touch",
        "tighten",
        "boundary2",
        "full",
        0.0,
        "tighten",
        0,
    ),
    "vp_hvn_touch_tighten_retest1": (
        "target_tighten",
        "vp_hvn",
        None,
        "touch",
        "tighten",
        "retest1",
        "full",
        0.0,
        "tighten",
        0,
    ),
    "measured_half_touch_tighten_boundary2": (
        "target_tighten",
        "measured_half",
        None,
        "touch",
        "tighten",
        "boundary2",
        "full",
        0.0,
        "tighten",
        0,
    ),
    "period_nearest_then_swing96_runner": (
        "dual_target",
        "period_nearest",
        "swing96",
        "touch",
        "partial",
        "boundary2",
        "full",
        0.33,
        "trail",
        0,
    ),
    "vp_hvn_then_measured_runner": (
        "dual_target",
        "vp_hvn",
        "measured_full",
        "touch",
        "partial",
        "retest2",
        "full",
        0.5,
        "trail",
        0,
    ),
    "progress24_nearest": (
        "progress_failure",
        "nearest",
        None,
        "touch",
        "full",
        "boundary2",
        "full",
        0.0,
        "hold",
        24,
    ),
    "progress48_measured": (
        "progress_failure",
        "measured_full",
        None,
        "touch",
        "full",
        "retest2",
        "full",
        0.0,
        "hold",
        48,
    ),
}


@dataclass
class FakeOrder:
    ft_order_side: str
    ft_order_tag: str | None = None
    safe_price: float = 100.0
    safe_filled: float = 0.0
    status: str = "closed"
    order_date_utc: datetime = datetime(2024, 1, 1, 1, tzinfo=UTC)
    order_filled_utc: datetime | None = None
    ft_is_open: bool = False
    safe_amount: float = 0.0
    order_id: str = "fake-order"


class FakeTrade:
    def __init__(self) -> None:
        self.pair = "BTC/USDT:USDT"
        self.entry_side = "buy"
        self.exit_side = "sell"
        self.open_rate = 100.0
        self.open_date_utc = datetime(2024, 1, 1, 1, tzinfo=UTC)
        self.max_rate = 100.0
        self.stake_amount = 1000.0
        self.leverage = 1.0
        self.orders: list[FakeOrder] = []
        self._custom: dict[str, Any] = {}

    @property
    def open_orders(self) -> list[FakeOrder]:
        return [
            order
            for order in self.orders
            if order.ft_is_open and order.ft_order_side != "stoploss"
        ]

    def get_custom_data(self, key: str) -> Any:
        return self._custom.get(key)

    def set_custom_data(self, key: str, value: Any) -> None:
        self._custom[key] = value


class FakeDataProvider:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def get_analyzed_dataframe(self, pair: str, timeframe: str):
        _ = pair, timeframe
        return self.frame.copy(), None


def _strategy() -> Sieve3V2IntegratedFromPriorMonthHighBreakoutLong:
    return Sieve3V2IntegratedFromPriorMonthHighBreakoutLong({})


def _state(
    plan: str,
    *,
    targets: dict[str, float] | None = None,
    partial_filled: bool = False,
) -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "plan": plan,
        "phase": "REMAINDER" if partial_filled else "ENTRY",
        "entry_rate": 100.0,
        "entry_candle": "2024-01-01T00:00:00+00:00",
        "entry_filled_at": "2024-01-01T01:00:00+00:00",
        "breakout_boundary": 99.0,
        "period_floor": 90.0,
        "targets": targets or {},
        "target_providers": {},
        "target_1_touched_at": None,
        "target_2_touched_at": None,
        "retest_held_at": None,
        "partial_filled": partial_filled,
        "partial_filled_at": (
            "2024-01-01T10:00:00+00:00" if partial_filled else None
        ),
        "partial_tag": None,
        "partial_target_stake": None,
        "partial_realized_stake": 0.0,
        "partial_fill_order_ids": [],
        "terminal_exit_pending": False,
        "terminal_exit_tag": None,
        "stop_floor": None,
    }


def _plan_signature(plan: Any) -> tuple[Any, ...]:
    return (
        plan.role,
        plan.target_1,
        plan.target_2,
        plan.confirmation,
        plan.target_action,
        plan.invalidation,
        plan.invalidation_action,
        plan.partial_fraction,
        plan.remainder,
        plan.progress_candles,
    )


def test_standalone_contract_exact_lock_and_sell_only_surface() -> None:
    strategy = _strategy()
    assert strategy.__class__.__bases__ == (IStrategy,)
    source = STRATEGY_PATH.read_text(encoding="utf-8")
    assert "from user_data.strategies" not in source
    for evidence_hash in (
        RESULT_ROW_SHA256,
        BACKTEST_ARCHIVE_SHA256,
        SNAPSHOT_STRATEGY_SHA256,
        SNAPSHOT_PARAMS_SHA256,
        LOCKED_BUY_PARAMS_SHA256,
    ):
        assert evidence_hash in source

    assert strategy.use_exit_signal is True
    assert strategy.use_custom_stoploss is True
    assert strategy.use_custom_roi is True
    assert strategy.position_adjustment_enable is True
    assert strategy.max_entry_position_adjustment == 0
    assert strategy.trailing_stop is False
    assert strategy.minimal_roi == {}
    assert strategy.exit_profit_only is False

    actual_lock: dict[str, Any] = {}
    for name, expected in LOCKED_BUY_DEFAULTS.items():
        parameter = getattr(strategy, name)
        actual_lock[name] = parameter.value
        assert parameter.value == expected, name
        assert parameter.space == "buy", name
        assert parameter.optimize is False, name
        assert parameter.load is False, name
    assert len(actual_lock) == 57
    lock_payload = json.dumps(
        actual_lock,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert hashlib.sha256(lock_payload).hexdigest() == LOCKED_BUY_PARAMS_SHA256

    parameters = {
        name: getattr(strategy, name)
        for name in dir(strategy)
        if getattr(strategy, name, None).__class__.__name__.endswith("Parameter")
    }
    assert {name for name, value in parameters.items() if value.optimize} == {
        "exit_policy_plan"
    }
    assert strategy.exit_policy_plan.space == "sell"


def test_freqtrade_strategy_resolver_loads_v2_module() -> None:
    with TemporaryDirectory() as temp_dir:
        copy2(STRATEGY_PATH, Path(temp_dir) / STRATEGY_PATH.name)
        config = {
            "strategy": "Sieve3V2IntegratedFromPriorMonthHighBreakoutLong",
            "strategy_path": temp_dir,
            "recursive_strategy_search": False,
            "user_data_dir": Path("user_data"),
            "timeframe": "1h",
            "stake_currency": "USDT",
            "stake_amount": 100,
            "max_open_trades": 5,
            "dry_run": True,
            "trading_mode": "futures",
            "margin_mode": "isolated",
            "exchange": {
                "name": "binance",
                "pair_whitelist": [],
                "pair_blacklist": [],
            },
        }
        strategy = StrategyResolver.load_strategy(config)
    assert strategy.use_exit_signal is True
    assert strategy.use_custom_roi is True
    assert strategy.minimal_roi == {}


def test_all_36_plan_signatures_are_unique_reachable_and_name_aligned() -> None:
    assert len(EXIT_PLANS) == 36
    assert len(EXIT_PLANS) == len(set(EXIT_PLANS.values()))
    assert set(EXIT_PLANS) == set(EXPECTED_PLAN_SIGNATURES)
    assert {
        name: _plan_signature(plan) for name, plan in EXIT_PLANS.items()
    } == EXPECTED_PLAN_SIGNATURES
    assert {plan.role for plan in EXIT_PLANS.values()} == {
        "baseline",
        "target_full",
        "invalidation_full",
        "defensive_tighten",
        "target_partial",
        "invalidation_reduce",
        "target_tighten",
        "dual_target",
        "progress_failure",
    }
    integrated = {
        "target_partial",
        "invalidation_reduce",
        "target_tighten",
        "dual_target",
        "progress_failure",
    }
    assert sum(plan.role in integrated for plan in EXIT_PLANS.values()) >= len(EXIT_PLANS) // 2

    strategy_class = Sieve3V2IntegratedFromPriorMonthHighBreakoutLong
    empty_state = {"partial_pending": False, "partial_filled": False}
    for name, plan in EXIT_PLANS.items():
        if plan.role == "baseline":
            assert plan.fixed_tp > 0.0, name
            continue
        if plan.target_action != "hold":
            decision = strategy_class.evaluate_policy(
                plan,
                empty_state,
                {"target_1": True, "profit_bucket": "profit"},
            )
            assert decision.action == plan.target_action, name
        if plan.invalidation != "none" and plan.invalidation_action != "hold":
            decision = strategy_class.evaluate_policy(
                plan,
                empty_state,
                {"invalidation": True, "profit_bucket": "profit"},
            )
            assert decision.action == plan.invalidation_action, name
        if plan.target_2:
            decision = strategy_class.evaluate_policy(
                plan,
                {"partial_pending": False, "partial_filled": True},
                {"target_2": True},
            )
            assert decision == ExitDecision("full", "s3v2_second_target"), name
        if plan.progress_candles:
            decision = strategy_class.evaluate_policy(
                plan,
                empty_state,
                {"time_failure": True},
            )
            assert decision == ExitDecision("full", "s3v2_progress_failure"), name


def test_long_target_hierarchy_and_invalidation_direction() -> None:
    strategy = _strategy()
    row = pd.Series(
        {
            "prior_month_high": 99.0,
            "prior_month_low": 90.0,
            "prior_day_high": 98.0,
            "prior_week_high": 106.0,
            "s3v2_swing_high_48": 103.0,
            "s3v2_swing_high_96": 107.0,
            "s3v2_swing_high_168": 112.0,
            "vp_hvn_above": 104.0,
            "vp_vah": 102.0,
        }
    )
    targets, providers, boundary, period_floor = strategy._frozen_targets(row, 100.0)
    assert boundary == 99.0
    assert period_floor == 90.0
    assert all(target is None or target > 100.2 for target in targets.values())
    assert targets["period_nearest"] == 106.0
    assert providers["period_nearest"] == "prior_week"
    for requested, provider in providers.items():
        assert provider is None or provider in TARGET_PROVIDER_CHAINS[requested]

    adverse_fill = row.copy()
    adverse_fill["prior_month_high"] = 101.0
    _, _, boundary, _ = strategy._frozen_targets(adverse_fill, 100.0)
    assert boundary == 101.0

    trade = FakeTrade()
    state = _state("boundary1_full")
    above = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01T01:00:00Z"], utc=True),
            "open": [100.0],
            "high": [101.0],
            "low": [99.0],
            "close": [99.5],
        }
    )
    below = above.assign(open=99.0, high=99.0, low=97.5, close=98.0)
    assert strategy._events(trade, state.copy(), above, datetime.now(UTC), 99.5, 0.0)[0][
        "invalidation"
    ] is False
    assert strategy._events(trade, state.copy(), below, datetime.now(UTC), 98.0, -0.02)[0][
        "invalidation"
    ] is True


def test_entry_freeze_uses_placement_candle_and_is_idempotent() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"],
                utc=True,
            ),
            "prior_month_high": [99.0, 500.0],
            "prior_month_low": [90.0, 400.0],
            "prior_day_high": [105.0, 510.0],
            "prior_week_high": [107.0, 520.0],
            "s3v2_swing_high_48": [106.0, 530.0],
            "s3v2_swing_high_96": [112.0, 540.0],
            "s3v2_swing_high_168": [115.0, 550.0],
            "vp_hvn_above": [108.0, 560.0],
            "vp_vah": [109.0, 570.0],
        }
    )
    strategy = _strategy()
    strategy.dp = FakeDataProvider(frame)
    trade = FakeTrade()
    order = FakeOrder(
        "buy",
        safe_price=100.0,
        order_filled_utc=datetime(2024, 1, 1, 5, tzinfo=UTC),
    )
    strategy.order_filled(
        trade.pair,
        trade,
        order,
        datetime(2024, 1, 1, 5, tzinfo=UTC),
    )
    state = trade.get_custom_data(STATE_KEY)
    assert state["breakout_boundary"] == 99.0
    assert state["period_floor"] == 90.0
    assert state["targets"]["period_nearest"] == 105.0
    assert state["entry_filled_at"] == datetime(2024, 1, 1, 5, tzinfo=UTC).isoformat()

    strategy.order_filled(
        trade.pair,
        trade,
        FakeOrder("buy", safe_price=200.0),
        datetime(2024, 1, 1, 6, tzinfo=UTC),
    )
    assert trade.get_custom_data(STATE_KEY) == state

    runner = _strategy()
    runner.exit_policy_plan.value = "period_nearest_then_swing96_runner"
    runner.dp = FakeDataProvider(frame)
    runner_trade = FakeTrade()
    runner.order_filled(
        runner_trade.pair,
        runner_trade,
        order,
        datetime(2024, 1, 1, 5, tzinfo=UTC),
    )
    runner_state = runner_trade.get_custom_data(STATE_KEY)
    assert runner_state["targets"]["period_nearest"] == 105.0
    assert runner_state["targets"]["swing96"] == 112.0


def test_adverse_fill_retains_boundary_and_invalidates_on_first_closed_loss() -> None:
    strategy = _strategy()
    strategy.exit_policy_plan.value = "boundary1_full"
    strategy.dp = FakeDataProvider(
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01T00:00:00Z"], utc=True),
                "prior_month_high": [101.0],
                "prior_month_low": [90.0],
            }
        )
    )
    trade = FakeTrade()
    fill = FakeOrder(
        "buy",
        safe_price=100.0,
        order_date_utc=datetime(2024, 1, 1, 1, tzinfo=UTC),
        order_filled_utc=datetime(2024, 1, 1, 1, tzinfo=UTC),
    )
    strategy.order_filled(
        trade.pair,
        trade,
        fill,
        datetime(2024, 1, 1, 1, tzinfo=UTC),
    )
    assert trade.get_custom_data(STATE_KEY)["breakout_boundary"] == 101.0

    strategy.dp = FakeDataProvider(
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01T01:00:00Z"], utc=True),
                "open": [100.5],
                "high": [100.8],
                "low": [99.5],
                "close": [100.0],
            }
        )
    )
    assert (
        strategy.custom_exit(
            trade.pair,
            trade,
            datetime(2024, 1, 1, 2, tzinfo=UTC),
            100.0,
            0.0,
        )
        == "s3v2_plan_invalidation"
    )


def test_delayed_fill_excludes_prefill_targets_and_progress() -> None:
    strategy = _strategy()
    trade = FakeTrade()
    trade.max_rate = 1000.0
    state = _state("progress24_nearest", targets={"nearest": 110.0})
    state["entry_filled_at"] = "2024-01-02T01:00:00+00:00"
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01T01:00:00Z", periods=30, freq="1h"),
            "open": 100.0,
            "high": [120.0] * 24 + [100.5] * 6,
            "low": 99.0,
            "close": 99.5,
        }
    )
    events, updated = strategy._events(
        trade,
        state,
        frame,
        datetime(2024, 1, 2, 7, tzinfo=UTC),
        99.5,
        -0.005,
    )
    assert events["target_1"] is False
    assert events["time_failure"] is False
    assert updated["target_1_touched_at"] is None


def test_target_confirmation_requires_postfill_closed_progress_and_reversal() -> None:
    strategy = _strategy()
    trade = FakeTrade()
    state = _state("swing48_touch_full", targets={"swing48": 110.0})
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"],
                utc=True,
            ),
            "open": [100.0, 105.0],
            "high": [120.0, 108.0],
            "low": [99.0, 104.0],
            "close": [115.0, 107.0],
        }
    )
    events, _ = strategy._events(
        trade,
        state,
        frame,
        datetime(2024, 1, 1, 2, tzinfo=UTC),
        107.0,
        0.07,
    )
    assert events["target_1"] is False

    dates = pd.date_range("2024-01-02", periods=6, freq="1h", tz="UTC")
    confirmation_frame = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0, 100.0, 100.0, 100.1, 100.2, 100.1],
            "high": [100.1, 100.1, 100.1, 100.25, 100.3, 100.2],
            "close": [99.9, 99.9, 99.9, 100.2, 100.1, 100.0],
        }
    )
    untouched_at, confirmed = strategy._confirmation(
        confirmation_frame.iloc[:3],
        100.2,
        "touch",
        None,
        0.03,
        100.0,
    )
    assert untouched_at is None
    assert confirmed is False
    touched_at, confirmed = strategy._confirmation(
        confirmation_frame.iloc[:5],
        100.2,
        "reversal2of3",
        None,
        0.03,
        100.0,
    )
    assert touched_at == dates[3].isoformat()
    assert confirmed is False
    same_touch_at, confirmed = strategy._confirmation(
        confirmation_frame,
        100.2,
        "reversal2of3",
        touched_at,
        0.03,
        100.0,
    )
    assert same_touch_at == touched_at
    assert confirmed is True


def test_failed_retest_and_confirmed_boundary_modes_are_distinct() -> None:
    strategy = _strategy()
    trade = FakeTrade()
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01T01:00:00Z", periods=3, freq="1h"),
            "open": [100.0, 99.3, 98.5],
            "high": [101.0, 99.5, 98.7],
            "low": [98.9, 97.8, 97.7],
            "close": [99.2, 98.0, 98.1],
        }
    )
    retest_state = _state("retest1_full")
    retest_events, updated = strategy._events(
        trade,
        retest_state,
        frame.iloc[:2],
        datetime.now(UTC),
        98.0,
        -0.02,
    )
    assert updated["retest_held_at"] == frame["date"].iloc[0].isoformat()
    assert retest_events["invalidation"] is True
    assert retest_events["hard_invalidation"] is False

    retest2_state = _state("retest2_full")
    retest2_events, _ = strategy._events(
        trade,
        retest2_state,
        frame,
        datetime.now(UTC),
        98.1,
        -0.019,
    )
    assert retest2_events["invalidation"] is True

    boundary2_state = _state("boundary2_full")
    boundary2_events, _ = strategy._events(
        trade,
        boundary2_state,
        frame.iloc[1:],
        datetime.now(UTC),
        98.1,
        -0.019,
    )
    assert boundary2_events["invalidation"] is True


def test_restart_reconstructs_failed_retest_from_persisted_hold_causally() -> None:
    strategy = _strategy()
    held_at = pd.Timestamp("2024-01-01T03:00:00Z")
    hold_frame = pd.DataFrame(
        {
            "date": pd.DatetimeIndex([held_at]),
            "open": [100.0],
            "high": [100.5],
            "low": [98.9],
            "close": [99.2],
        }
    )
    _, persisted = strategy._events(
        FakeTrade(),
        _state("retest1_full"),
        hold_frame,
        datetime(2024, 1, 1, 4, tzinfo=UTC),
        99.2,
        -0.008,
    )
    assert persisted["retest_held_at"] == held_at.isoformat()

    pre_hold_trade = FakeTrade()
    pre_hold_trade.set_custom_data(STATE_KEY, dict(persisted))
    pre_hold_restart = _strategy()
    pre_hold_restart.dp = FakeDataProvider(
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01T02:00:00Z"], utc=True),
                "open": [98.5],
                "high": [98.8],
                "low": [97.8],
                "close": [98.0],
            }
        )
    )
    assert (
        pre_hold_restart.custom_exit(
            pre_hold_trade.pair,
            pre_hold_trade,
            datetime(2024, 1, 1, 3, tzinfo=UTC),
            98.0,
            -0.02,
        )
        is None
    )

    restarted_trade = FakeTrade()
    restarted_trade.set_custom_data(STATE_KEY, dict(persisted))
    restarted = _strategy()
    restarted.dp = FakeDataProvider(
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01T04:00:00Z"], utc=True),
                "open": [98.5],
                "high": [98.8],
                "low": [97.8],
                "close": [98.0],
            }
        )
    )
    assert (
        restarted.custom_exit(
            restarted_trade.pair,
            restarted_trade,
            datetime(2024, 1, 1, 5, tzinfo=UTC),
            98.0,
            -0.02,
        )
        == "s3v2_plan_invalidation"
    )


def test_partial_state_is_positive_fill_idempotent() -> None:
    strategy = _strategy()
    trade = FakeTrade()
    state = _state(
        "period_nearest_touch_p33_be_boundary2",
        targets={"period_nearest": 110.0},
    )
    state["phase"] = "TARGET_ZONE"
    state["partial_target_stake"] = 330.0
    trade.set_custom_data(STATE_KEY, state)
    for tag, filled in (("wrong", 3.3), ("s3v2_partial_target", 0.0)):
        strategy.order_filled(
            trade.pair,
            trade,
            FakeOrder("sell", tag, safe_filled=filled),
            datetime.now(UTC),
        )
        assert trade.get_custom_data(STATE_KEY)["partial_filled"] is False

    filled_at = datetime(2024, 1, 2, 3, tzinfo=UTC)
    fill = FakeOrder(
        "sell",
        "s3v2_partial_target",
        safe_filled=3.3,
        order_filled_utc=filled_at,
        safe_amount=3.3,
        order_id="partial-complete",
    )
    strategy.order_filled(
        trade.pair,
        trade,
        fill,
        datetime(2024, 1, 2, 5, tzinfo=UTC),
    )
    completed = trade.get_custom_data(STATE_KEY)
    assert completed["partial_filled"] is True
    assert completed["phase"] == "REMAINDER"
    assert completed["partial_tag"] in PARTIAL_TAGS
    assert completed["partial_filled_at"] == filled_at.isoformat()

    strategy.order_filled(
        trade.pair,
        trade,
        fill,
        datetime(2024, 1, 2, 7, tzinfo=UTC),
    )
    assert trade.get_custom_data(STATE_KEY) == completed


def test_partial_fill_reconciles_quantity_and_preserves_minimum_remainder() -> None:
    strategy = _strategy()
    trade = FakeTrade()
    state = _state(
        "period_nearest_touch_p33_be_boundary2",
        targets={"period_nearest": 110.0},
    )
    state["partial_target_stake"] = 330.0
    trade.set_custom_data(STATE_KEY, state)
    strategy.order_filled(
        trade.pair,
        trade,
        FakeOrder(
            "sell",
            "s3v2_partial_target",
            safe_filled=1.65,
            safe_amount=3.3,
            safe_price=100.0,
            order_id="partial-half-1",
        ),
        datetime(2024, 1, 2, 3, tzinfo=UTC),
    )
    assert trade.get_custom_data(STATE_KEY)["partial_realized_stake"] == 165.0
    trade.stake_amount = 835.0

    def half_fill_context(*args: Any, **kwargs: Any):
        _ = args, kwargs
        return (
            EXIT_PLANS["period_nearest_touch_p33_be_boundary2"],
            trade.get_custom_data(STATE_KEY),
            {},
            ExitDecision("partial", "s3v2_partial_target", 0.33),
        )

    strategy._decision_context = half_fill_context  # type: ignore[method-assign]
    request = strategy.adjust_trade_position(
        trade,
        datetime.now(UTC),
        100.0,
        0.0,
        None,
        1000.0,
        100.0,
        100.0,
        0.0,
        0.0,
    )
    assert request == (-165.0, "s3v2_partial_target")

    safety_trade = FakeTrade()
    safety_trade.stake_amount = 100.0
    safety_state = _state(
        "retest1_reduce50_vp_hvn",
        targets={"vp_hvn": 110.0},
    )
    safety_state["partial_target_stake"] = 50.0
    safety_trade.set_custom_data(STATE_KEY, safety_state)

    def safety_context(*args: Any, **kwargs: Any):
        _ = args, kwargs
        return (
            EXIT_PLANS["retest1_reduce50_vp_hvn"],
            safety_trade.get_custom_data(STATE_KEY),
            {},
            ExitDecision("partial", "s3v2_partial_invalidation", 0.5),
        )

    strategy._decision_context = safety_context  # type: ignore[method-assign]
    safe_request = strategy.adjust_trade_position(
        safety_trade,
        datetime.now(UTC),
        100.0,
        0.0,
        80.0,
        1000.0,
        100.0,
        100.0,
        0.0,
        0.0,
    )
    assert safe_request == (-20.0, "s3v2_partial_invalidation")


def test_open_orders_block_duplicate_partial_submissions() -> None:
    strategy = _strategy()
    trade = FakeTrade()
    state = _state(
        "period_nearest_touch_p33_be_boundary2",
        targets={"period_nearest": 110.0},
    )
    state["partial_target_stake"] = 330.0
    trade.set_custom_data(STATE_KEY, state)
    trade.orders.append(
        FakeOrder(
            "sell",
            "s3v2_partial_target",
            status="open",
            ft_is_open=True,
            order_id="pending-partial",
        )
    )
    assert strategy._has_pending_partial(trade) is True
    assert (
        strategy.adjust_trade_position(
            trade,
            datetime.now(UTC),
            110.0,
            0.10,
            None,
            1000.0,
            110.0,
            110.0,
            0.10,
            0.10,
        )
        is None
    )

    stoploss_only = FakeTrade()
    stoploss_only.orders.append(FakeOrder("stoploss", status="open", ft_is_open=True))
    assert strategy._has_open_order(stoploss_only) is False


def test_transient_full_invalidation_waits_for_partial_and_keeps_exact_tag() -> None:
    strategy = _strategy()
    trade = FakeTrade()
    state = _state(
        "period_nearest_touch_p33_be_boundary2",
        targets={"period_nearest": 110.0},
    )
    state["partial_target_stake"] = 330.0
    trade.set_custom_data(STATE_KEY, state)
    trade.orders.append(
        FakeOrder(
            "sell",
            "s3v2_partial_target",
            status="open",
            ft_is_open=True,
            order_id="pending-partial",
        )
    )
    strategy.dp = FakeDataProvider(
        pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2024-01-01T01:00:00Z", "2024-01-01T02:00:00Z"],
                    utc=True,
                ),
                "open": [98.5, 98.4],
                "high": [98.8, 98.7],
                "low": [97.8, 97.7],
                "close": [98.0, 98.1],
            }
        )
    )
    assert (
        strategy.custom_exit(
            trade.pair,
            trade,
            datetime(2024, 1, 1, 3, tzinfo=UTC),
            98.1,
            -0.019,
        )
        is None
    )
    persisted = trade.get_custom_data(STATE_KEY)
    assert persisted["terminal_exit_pending"] is True
    assert persisted["terminal_exit_tag"] == "s3v2_plan_invalidation"
    assert (
        strategy.custom_exit(
            trade.pair,
            trade,
            datetime(2024, 1, 1, 3, tzinfo=UTC),
            98.1,
            -0.019,
        )
        is None
    )
    assert trade.get_custom_data(STATE_KEY)["terminal_exit_tag"] == persisted[
        "terminal_exit_tag"
    ]

    trade.orders.clear()
    strategy.dp = FakeDataProvider(
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01T03:00:00Z"], utc=True),
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
            }
        )
    )
    assert (
        strategy.custom_exit(
            trade.pair,
            trade,
            datetime(2024, 1, 1, 4, tzinfo=UTC),
            100.0,
            0.0,
        )
        == "s3v2_plan_invalidation"
    )
    assert (
        strategy.custom_exit(
            trade.pair,
            trade,
            datetime(2024, 1, 1, 4, tzinfo=UTC),
            100.0,
            0.0,
        )
        == "s3v2_plan_invalidation"
    )


def test_pending_hard_invalidation_keeps_legacy_compatibility() -> None:
    strategy = _strategy()
    trade = FakeTrade()
    state = _state(
        "period_nearest_touch_p33_be_boundary2",
        targets={"period_nearest": 110.0},
    )
    trade.set_custom_data(STATE_KEY, state)
    trade.orders.append(
        FakeOrder(
            "sell",
            "s3v2_partial_target",
            status="open",
            ft_is_open=True,
            order_id="pending-partial",
        )
    )
    strategy.dp = FakeDataProvider(
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01T01:00:00Z"], utc=True),
                "open": [98.0],
                "high": [98.0],
                "low": [94.0],
                "close": [95.0],
            }
        )
    )
    assert (
        strategy.custom_exit(
            trade.pair,
            trade,
            datetime(2024, 1, 1, 2, tzinfo=UTC),
            95.0,
            -0.05,
        )
        is None
    )
    persisted = trade.get_custom_data(STATE_KEY)
    assert persisted["terminal_exit_pending"] is True
    assert persisted["terminal_exit_tag"] == "s3v2_hard_invalidation"

    legacy_state = {"terminal_exit_pending": True, "partial_filled": False}
    assert strategy.evaluate_policy(
        EXIT_PLANS["period_nearest_touch_p33_be_boundary2"],
        legacy_state,
        {},
    ) == ExitDecision("full", "s3v2_hard_invalidation")


def test_all_plans_have_roi_fallback_bounded_hold_and_progress_failure() -> None:
    strategy = _strategy()
    trade = FakeTrade()
    for plan_name, expected in (("baseline_3_3", 0.03), ("promotion_2_2", 0.02)):
        trade.set_custom_data(STATE_KEY, {"version": STATE_VERSION, "plan": plan_name})
        assert (
            strategy.custom_roi(
                trade.pair,
                trade,
                datetime.now(UTC),
                10,
                None,
                "long",
            )
            == expected
        )
    trade.set_custom_data(
        STATE_KEY,
        {"version": STATE_VERSION, "plan": "swing48_touch_full"},
    )
    assert (
        strategy.custom_roi(
            trade.pair,
            trade,
            datetime.now(UTC),
            10,
            None,
            "long",
        )
        == 0.08
    )
    for name, plan in EXIT_PLANS.items():
        if plan.role == "baseline":
            assert plan.max_hold_candles == 0, name
        else:
            assert plan.fixed_tp == 0.08, name
            assert plan.max_hold_candles == 720, name
    assert Sieve3V2IntegratedFromPriorMonthHighBreakoutLong.evaluate_policy(
        EXIT_PLANS["swing48_touch_full"],
        {"partial_pending": False, "partial_filled": False},
        {"max_hold": True},
    ) == ExitDecision("full", "s3v2_max_hold")

    target_plan = EXIT_PLANS["period_nearest_touch_p33_be_boundary2"]
    assert Sieve3V2IntegratedFromPriorMonthHighBreakoutLong.evaluate_policy(
        target_plan,
        {"partial_pending": False, "partial_filled": False},
        {"target_1": True, "profit_bucket": "loss"},
    ) == ExitDecision("full", "s3v2_target_reversal_loss")
    reduce_plan = EXIT_PLANS["retest1_reduce50_vp_hvn"]
    assert Sieve3V2IntegratedFromPriorMonthHighBreakoutLong.evaluate_policy(
        reduce_plan,
        {"partial_pending": False, "partial_filled": False},
        {"invalidation": True, "profit_bucket": "loss"},
    ) == ExitDecision("full", "s3v2_invalidation_loss")

    progress_state = _state("progress24_nearest", targets={"nearest": 110.0})
    trade.max_rate = 1000.0
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01T01:00:00Z", periods=25, freq="1h"),
            "open": 99.5,
            "high": 100.5,
            "low": 98.9,
            "close": 99.5,
        }
    )
    events, _ = strategy._events(
        trade,
        progress_state,
        frame,
        datetime(2024, 1, 2, 2, tzinfo=UTC),
        99.5,
        -0.005,
    )
    assert events["time_failure"] is True


def test_custom_stoploss_is_monotonic_and_freqtrade_applies_it() -> None:
    strategy = _strategy()
    trade = FakeTrade()
    state = _state(
        "vp_hvn_rev1_p50_trail_boundary2",
        targets={"vp_hvn": 110.0},
        partial_filled=True,
    )
    trade.set_custom_data(STATE_KEY, state)
    strategy.custom_stoploss(
        trade.pair,
        trade,
        datetime.now(UTC),
        120.0,
        0.20,
        False,
    )
    first_floor = trade.get_custom_data(STATE_KEY)["stop_floor"]
    assert first_floor == pytest.approx(118.2)
    strategy.custom_stoploss(
        trade.pair,
        trade,
        datetime.now(UTC),
        110.0,
        0.10,
        True,
    )
    assert trade.get_custom_data(STATE_KEY)["stop_floor"] == first_floor

    with FtNoDBContext("1h"):
        real_trade = Trade(
            id=940001,
            pair="BTC/USDT:USDT",
            stake_amount=1000.0,
            amount=10.0,
            open_date=datetime(2024, 1, 1, tzinfo=UTC),
            fee_open=0.0,
            fee_close=0.0,
            exchange="binance",
            open_rate=100.0,
            price_precision=2,
            precision_mode=2,
            precision_mode_price=2,
            leverage=1.0,
            is_short=False,
        )
        real_trade.set_custom_data(
            STATE_KEY,
            _state(
                "vp_hvn_rev1_p50_trail_boundary2",
                targets={"vp_hvn": 110.0},
                partial_filled=True,
            ),
        )
        strategy.ft_stoploss_adjust(
            120.0,
            real_trade,
            datetime(2024, 1, 2, tzinfo=UTC),
            0.20,
            0,
        )
        assert real_trade.stop_loss == pytest.approx(118.2)


def test_tighten_floor_breach_is_explicit_without_mutating_floor() -> None:
    strategy = _strategy()
    trade = FakeTrade()
    trade.set_custom_data(
        STATE_KEY,
        _state(
            "period_nearest_touch_tighten_boundary2",
            targets={"period_nearest": 110.0},
        ),
    )
    strategy.dp = FakeDataProvider(
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01T01:00:00Z"], utc=True),
                "open": [109.0],
                "high": [111.0],
                "low": [104.0],
                "close": [105.0],
            }
        )
    )
    assert (
        strategy.custom_exit(
            trade.pair,
            trade,
            datetime(2024, 1, 1, 2, tzinfo=UTC),
            105.0,
            0.05,
        )
        == "s3v2_stop_floor_breached"
    )
    assert trade.get_custom_data(STATE_KEY)["stop_floor"] is None


def test_same_candle_callback_causality_excludes_unclosed_target_data() -> None:
    strategy = _strategy()
    trade = FakeTrade()
    trade.set_custom_data(
        STATE_KEY,
        _state("swing48_touch_full", targets={"swing48": 110.0}),
    )
    strategy.dp = FakeDataProvider(
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01T01:00:00Z"], utc=True),
                "open": [100.0],
                "high": [120.0],
                "low": [99.0],
                "close": [115.0],
            }
        )
    )
    assert (
        strategy.custom_exit(
            trade.pair,
            trade,
            datetime(2024, 1, 1, 1, 30, tzinfo=UTC),
            100.0,
            0.0,
        )
        is None
    )
    assert trade.get_custom_data(STATE_KEY)["target_1_touched_at"] is None
    assert (
        strategy.custom_exit(
            trade.pair,
            trade,
            datetime(2024, 1, 1, 2, tzinfo=UTC),
            115.0,
            0.15,
        )
        == "s3v2_target_full"
    )

    with FtNoDBContext("1h"):
        real_trade = Trade(
            id=940002,
            pair="BTC/USDT:USDT",
            stake_amount=1000.0,
            amount=10.0,
            open_date=datetime(2024, 1, 1, tzinfo=UTC),
            fee_open=0.0,
            fee_close=0.0,
            exchange="binance",
            open_rate=100.0,
            price_precision=2,
            precision_mode=2,
            precision_mode_price=2,
            leverage=1.0,
            is_short=False,
        )
        real_trade.set_custom_data(
            STATE_KEY,
            _state(
                "vp_hvn_rev1_p50_trail_boundary2",
                targets={"vp_hvn": 130.0},
                partial_filled=True,
            ),
        )
        exits = strategy.should_exit(
            real_trade,
            110.0,
            datetime(2024, 1, 2, tzinfo=UTC),
            enter=False,
            exit_=False,
            low=109.0,
            high=120.0,
        )
        assert all(check.exit_type != ExitType.CUSTOM_EXIT for check in exits)
        assert any(check.exit_type == ExitType.TRAILING_STOP_LOSS for check in exits)
        assert real_trade.get_custom_data(STATE_KEY)["stop_floor"] == pytest.approx(118.2)


def test_entry_driving_columns_and_signals_are_prefix_invariant() -> None:
    frame = pd.read_feather(DATA_PATH).iloc[5000:6000].reset_index(drop=True)
    short = frame.iloc[:700].copy()
    strategy = _strategy()
    metadata = {"pair": "BTC/USDT:USDT"}
    short_result = strategy.populate_entry_trend(
        strategy.populate_indicators(short.copy(), metadata),
        metadata,
    )
    full_result = strategy.populate_entry_trend(
        strategy.populate_indicators(frame.copy(), metadata),
        metadata,
    ).iloc[:700].reset_index(drop=True)
    for column in (
        "prior_month_high",
        "prior_month_low",
        "entry_volume_ratio",
        "entry_pressure",
        "s3v2_swing_high_48",
        "s3v2_swing_high_96",
        "s3v2_swing_high_168",
        "vp_hvn_above",
        "vp_vah",
    ):
        np.testing.assert_allclose(
            pd.to_numeric(short_result[column], errors="coerce"),
            pd.to_numeric(full_result[column], errors="coerce"),
            equal_nan=True,
        )
    pd.testing.assert_series_equal(short_result["enter_long"], full_result["enter_long"])
    pd.testing.assert_series_equal(
        short_result["enter_tag"].fillna(""),
        full_result["enter_tag"].fillna(""),
    )


def test_authoritative_entry_parity_against_archived_tested_control() -> None:
    frame = pd.read_feather(DATA_PATH)
    frame = frame.loc[
        frame["date"].ge(pd.Timestamp("2020-04-01", tz="UTC"))
        & frame["date"].lt(pd.Timestamp("2020-10-01", tz="UTC"))
    ].reset_index(drop=True)
    metadata = {"pair": "BTC/USDT:USDT"}
    control = Sieve3ExitBreakevenFromPriorMonthHighBreakoutLong({})
    candidate = _strategy()
    control_frame = control.populate_entry_trend(
        control.populate_indicators(frame.copy(), metadata),
        metadata,
    )
    candidate_frame = candidate.populate_entry_trend(
        candidate.populate_indicators(frame.copy(), metadata),
        metadata,
    )
    assert int(control_frame["enter_long"].sum()) == 9
    pd.testing.assert_series_equal(
        control_frame["enter_long"],
        candidate_frame["enter_long"],
    )
    pd.testing.assert_series_equal(
        control_frame["enter_tag"].fillna(""),
        candidate_frame["enter_tag"].fillna(""),
    )
