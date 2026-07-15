from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import pandas as pd

from freqtrade.resolvers import StrategyResolver
from freqtrade.strategy import IStrategy
from user_data.strategies.sieve3_exit_breakeven_from_multi2_tlv2_vp_res_break_vp_bullctx_long_1h import (
    Sieve3ExitBreakevenFromMulti2Tlv2VpResBreakVpBullctxLong1H,
)
from user_data.strategies.sieve3_V2_integrated_from_multi2_tlv2_vp_res_break_vp_bullctx_long_1h import (
    EXIT_PLANS,
    PARTIAL_TAGS,
    STATE_KEY,
    ExitDecision,
    Sieve3V2IntegratedFromMulti2Tlv2VpResBreakVpBullctxLong1H,
)


STRATEGY_PATH = Path(
    "user_data/strategies/sieve3_V2_integrated_from_multi2_tlv2_vp_res_break_vp_bullctx_long_1h.py"
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
    "use_volume_guard": True,
    "volume_guard_window": 24,
    "volume_ratio_min": 1.6,
    "use_pressure_guard": True,
    "pressure_window": 24,
    "pressure_min": 0.15,
    "use_accumulation_guard": False,
    "use_body_direction_guard": False,
    "use_close_direction_guard": False,
    "vp_window": 96,
    "vp_bins": 48,
    "vp_value_area_pct": 0.70,
    "vp_price_source": "hlc3",
    "vp_smooth_bins": 3,
    "vp_hvn_threshold": 0.70,
    "vp_lvn_threshold": 0.35,
    "vp_pressure_delta_min": 0.05,
    "vp_node_near_pct": 0.010,
    "vp_volume_percentile_min": 0.55,
    "vp_score_window": 48,
    "vp_fast_traverse_atr_mult": 1.2,
    "vp_entry_score_margin": 0.02,
    "vp_score_min": 0.4,
    "vp_context_min": 0.28,
    "vp_level_buffer_pct": 0.016,
    "pivot_strength": 2,
    "min_line_score": 0.6,
    "min_active_bars": 8,
    "max_distance_atr": 3.0,
    "proximity_rank_weight": 0.05,
    "line_buffer_pct": 0.0,
    "line_slope_min_pct": 0.0005,
}


@dataclass
class FakeOrder:
    ft_order_side: str
    ft_order_tag: str | None = None
    safe_price: float = 100.0
    safe_filled: float = 0.0
    status: str = "closed"
    order_date_utc: datetime = datetime(2024, 1, 1, 1, tzinfo=UTC)


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


def _strategy() -> Sieve3V2IntegratedFromMulti2Tlv2VpResBreakVpBullctxLong1H:
    return Sieve3V2IntegratedFromMulti2Tlv2VpResBreakVpBullctxLong1H({})


def test_standalone_contract_and_locked_entry_surface() -> None:
    strategy = _strategy()
    assert strategy.__class__.__bases__ == (IStrategy,)
    source = STRATEGY_PATH.read_text(encoding="utf-8")
    assert "user_data.strategies" not in source
    assert "tlv2_support_line_rank0" not in source

    assert strategy.use_exit_signal is True
    assert strategy.use_custom_stoploss is True
    assert strategy.use_custom_roi is False
    assert strategy.position_adjustment_enable is True
    assert strategy.max_entry_position_adjustment == 0
    assert strategy.trailing_stop is False
    assert strategy.minimal_roi == {}
    assert strategy.exit_profit_only is False

    for name, expected in LOCKED_BUY_DEFAULTS.items():
        parameter = getattr(strategy, name)
        assert parameter.value == expected, name
        assert parameter.space == "buy", name
        assert parameter.optimize is False, name
        assert parameter.load is False, name

    parameters = {
        name: getattr(strategy, name)
        for name in dir(strategy)
        if getattr(strategy, name, None).__class__.__name__.endswith("Parameter")
    }
    active = {name for name, value in parameters.items() if value.optimize}
    assert active == {"exit_policy_plan"}
    assert strategy.exit_policy_plan.space == "sell"


def test_freqtrade_strategy_resolver_loads_v2_module() -> None:
    with TemporaryDirectory() as temp_dir:
        copy2(STRATEGY_PATH, Path(temp_dir) / STRATEGY_PATH.name)
        config = {
            "strategy": "Sieve3V2IntegratedFromMulti2Tlv2VpResBreakVpBullctxLong1H",
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
    assert strategy.use_custom_roi is False
    assert strategy.minimal_roi == {}


def test_exit_plans_are_compact_unique_whole_policies() -> None:
    assert 20 <= len(EXIT_PLANS) <= 50
    assert len(EXIT_PLANS) == len(set(EXIT_PLANS.values()))
    assert {plan.role for plan in EXIT_PLANS.values()} >= {
        "baseline",
        "target_full",
        "invalidation_full",
        "target_partial",
        "invalidation_reduce",
        "target_tighten",
        "dual_target",
        "progress_failure",
    }


def test_pure_evaluator_priority_and_actions() -> None:
    partial_plan = EXIT_PLANS["hvn_touch_p33_be_level2"]
    pending = {"partial_pending": True, "partial_filled": False}
    assert (
        Sieve3V2IntegratedFromMulti2Tlv2VpResBreakVpBullctxLong1H.evaluate_policy(
            partial_plan,
            pending,
            {"hard_invalidation": True, "target_1": True},
        ).tag
        == "s3v2_hard_invalidation"
    )
    assert (
        Sieve3V2IntegratedFromMulti2Tlv2VpResBreakVpBullctxLong1H.evaluate_policy(
            partial_plan,
            pending,
            {"invalidation": True, "target_1": True},
        ).tag
        == "s3v2_plan_invalidation"
    )
    assert (
        Sieve3V2IntegratedFromMulti2Tlv2VpResBreakVpBullctxLong1H.evaluate_policy(
            partial_plan,
            pending,
            {"target_1": True},
        ).action
        == "hold"
    )

    partial = Sieve3V2IntegratedFromMulti2Tlv2VpResBreakVpBullctxLong1H.evaluate_policy(
        partial_plan,
        {"partial_pending": False, "partial_filled": False},
        {"target_1": True},
    )
    assert partial == ExitDecision("partial", "s3v2_partial_target", 0.33)

    tighten = Sieve3V2IntegratedFromMulti2Tlv2VpResBreakVpBullctxLong1H.evaluate_policy(
        EXIT_PLANS["level2_tighten"],
        {"partial_pending": False, "partial_filled": False},
        {"invalidation": True},
    )
    assert tighten == ExitDecision("tighten", tighten="invalidation")
    assert (
        Sieve3V2IntegratedFromMulti2Tlv2VpResBreakVpBullctxLong1H.evaluate_policy(
            partial_plan,
            {"partial_pending": False, "partial_filled": False},
            {},
        ).action
        == "hold"
    )


def test_entry_freeze_uses_placement_candle_and_fill_is_idempotent() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"], utc=True),
            "tlv2_resistance_line_rank0": [99.0, 500.0],
            "vp_hvn_above": [110.0, 600.0],
            "vp_vah": [108.0, 590.0],
            "s3v2_prior_high_48": [107.0, 580.0],
            "s3v2_prior_high_96": [112.0, 570.0],
            "s3v2_prior_low_48": [95.0, 400.0],
        }
    )
    strategy = _strategy()
    strategy.dp = FakeDataProvider(frame)
    trade = FakeTrade()
    order = FakeOrder("buy", safe_price=100.0)
    delayed_fill = datetime(2024, 1, 1, 5, tzinfo=UTC)
    strategy.order_filled(trade.pair, trade, order, delayed_fill)
    state = trade.get_custom_data(STATE_KEY)
    assert state["broken_resistance"] == 99.0
    assert state["structural_support"] == 95.0
    assert state["targets"]["hvn"] == 110.0

    changed_order = FakeOrder("buy", safe_price=200.0)
    strategy.order_filled(trade.pair, trade, changed_order, delayed_fill)
    assert trade.get_custom_data(STATE_KEY) == state

    runner = _strategy()
    runner.exit_policy_plan.value = "hvn_then_measured_runner"
    runner.dp = FakeDataProvider(frame)
    runner_trade = FakeTrade()
    runner.order_filled(runner_trade.pair, runner_trade, order, delayed_fill)
    runner_state = runner_trade.get_custom_data(STATE_KEY)
    assert runner_state["targets"]["hvn"] == 110.0
    assert runner_state["targets"]["measured_full"] is None


def test_partial_state_changes_only_after_exact_positive_fill() -> None:
    strategy = _strategy()
    trade = FakeTrade()
    trade.set_custom_data(
        STATE_KEY,
        {
            "version": 1,
            "plan": "hvn_touch_p33_be_level2",
            "phase": "TARGET_ZONE",
            "partial_filled": False,
        },
    )
    for tag, filled in (("wrong", 1.0), ("s3v2_partial_target", 0.0)):
        strategy.order_filled(
            trade.pair,
            trade,
            FakeOrder("sell", tag, safe_filled=filled),
            datetime.now(UTC),
        )
        assert trade.get_custom_data(STATE_KEY)["partial_filled"] is False

    strategy.order_filled(
        trade.pair,
        trade,
        FakeOrder("sell", "s3v2_partial_target", safe_filled=1.0),
        datetime.now(UTC),
    )
    state = trade.get_custom_data(STATE_KEY)
    assert state["partial_filled"] is True
    assert state["phase"] == "REMAINDER"
    assert state["partial_tag"] in PARTIAL_TAGS


def test_open_order_blocks_duplicate_partial_request() -> None:
    strategy = _strategy()
    trade = FakeTrade()
    trade.orders.append(
        FakeOrder(
            "sell",
            "s3v2_partial_target",
            status="open",
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


def test_target_confirmation_cannot_reuse_signal_candle_touch() -> None:
    strategy = _strategy()
    trade = FakeTrade()
    state = {
        "version": 1,
        "plan": "hvn_touch_full",
        "phase": "ENTRY",
        "entry_rate": 100.0,
        "entry_candle": "2024-01-01T00:00:00+00:00",
        "broken_resistance": 99.0,
        "structural_support": 95.0,
        "targets": {"hvn": 110.0},
        "target_1_touched": False,
        "target_2_touched": False,
        "partial_filled": False,
    }
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"], utc=True),
            "open": [100.0, 105.0],
            "high": [120.0, 108.0],
            "low": [99.0, 104.0],
            "close": [115.0, 107.0],
            "vp_context_score_bull": [0.5, 0.5],
            "vp_context_score_bear": [0.1, 0.1],
            "vp_market_context": [0.5, 0.5],
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


def test_entry_driving_indicators_are_prefix_invariant() -> None:
    frame = pd.read_feather(DATA_PATH).iloc[5000:6000].reset_index(drop=True)
    strategy = _strategy()
    short = frame.iloc[:700].copy()

    short_tlv2 = strategy._add_tlv2(short.copy())
    full_tlv2 = strategy._add_tlv2(frame.copy()).iloc[:700].reset_index(drop=True)
    for column in (
        "tlv2_resistance_line_rank0",
        "tlv2_resistance_score_rank0",
        "tlv2_resistance_distance_atr_rank0",
    ):
        np.testing.assert_allclose(
            pd.to_numeric(short_tlv2[column], errors="coerce"),
            pd.to_numeric(full_tlv2[column], errors="coerce"),
            equal_nan=True,
        )

    short_vp = strategy._add_vp(short.copy())
    full_vp = strategy._add_vp(frame.copy()).iloc[:700].reset_index(drop=True)
    for column in (
        "vp_score_long",
        "vp_context_score_bull",
        "vp_context_score_bear",
        "vp_market_context",
        "vp_hvn_above",
        "vp_vah",
    ):
        np.testing.assert_allclose(
            pd.to_numeric(short_vp[column], errors="coerce"),
            pd.to_numeric(full_vp[column], errors="coerce"),
            equal_nan=True,
        )


def test_neutral_entry_parity_against_tested_control() -> None:
    frame = pd.read_feather(DATA_PATH)
    frame = frame.loc[
        frame["date"].ge(pd.Timestamp("2020-04-01", tz="UTC"))
        & frame["date"].lt(pd.Timestamp("2020-10-01", tz="UTC"))
    ].reset_index(drop=True)
    metadata = {"pair": "BTC/USDT:USDT"}

    control = Sieve3ExitBreakevenFromMulti2Tlv2VpResBreakVpBullctxLong1H({})
    candidate = _strategy()
    control_frame = control.populate_entry_trend(
        control.populate_indicators(frame.copy(), metadata), metadata
    )
    candidate_frame = candidate.populate_entry_trend(
        candidate.populate_indicators(frame.copy(), metadata), metadata
    )

    assert int(control_frame["enter_long"].sum()) == 5
    pd.testing.assert_series_equal(control_frame["enter_long"], candidate_frame["enter_long"])
    pd.testing.assert_series_equal(
        control_frame["enter_tag"].fillna(""),
        candidate_frame["enter_tag"].fillna(""),
    )
