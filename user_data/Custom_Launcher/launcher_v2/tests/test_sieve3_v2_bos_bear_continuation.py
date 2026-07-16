from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd

from freqtrade.persistence import Trade
from freqtrade.persistence.usedb_context import FtNoDBContext
from freqtrade.resolvers import StrategyResolver
from freqtrade.strategy import IStrategy
from user_data.strategies.sieve3_V2_integrated_from_bos_bear_continuation_short_1h import (
    EFFECTIVE_BUY_LOCK,
    ENTRY_TAG,
    EXIT_PLAN_SIGNATURES,
    EXIT_PLANS,
    PARTIAL_TAGS,
    SOURCE_BACKTEST_ARCHIVE_SHA256,
    SOURCE_BASELINE_ENTRY_SIGNATURE_SHA256,
    SOURCE_EFFECTIVE_BUY_LOCK_SHA256,
    SOURCE_RESULT_ROW_SHA256,
    SOURCE_SNAPSHOT_PARAMS_SHA256,
    SOURCE_SNAPSHOT_STRATEGY_SHA256,
    STATE_KEY,
    STATE_VERSION,
    ExitDecision,
    Sieve3V2IntegratedFromBosBearContinuationShort1H,
)


STRATEGY_PATH = Path(
    "user_data/strategies/sieve3_V2_integrated_from_bos_bear_continuation_short_1h.py"
)
DATA_PATH = Path("user_data/data/binance/futures/BTC_USDT_USDT-1h-futures.feather")
RESULT_PATH = Path(
    "user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve/results/"
    "20260521T012740_entry_all_resume.jsonl"
)
ARCHIVE_PATH = Path(
    "D:/FreqTradeStuffLargeData/sieve_runtime/entry_sieve/backtests/"
    "sieve2_bos_bear_continuation_short_1h__pattern_continuation_1h_2021_22/"
    "full_cycle_2020_2026/tp_2_sl_2/backtest-result-2026-05-21_18-42-10.zip"
)
SNAPSHOT_STRATEGY_MEMBER = (
    "backtest-result-2026-05-21_18-42-10_Sieve2BOSBearContinuationShort1H.py"
)
SNAPSHOT_PARAMS_MEMBER = (
    "backtest-result-2026-05-21_18-42-10_Sieve2BOSBearContinuationShort1H.json"
)
ARCHIVED_DEFAULTS_SHA256 = "6f74863399ab0c0e3d1350a8c5adda7eeddfb677ad90097dd2147c04e7c2a44b"
SELECTED_BUY_PARAMS_SHA256 = "cf93f3da9737a5dfe19ce4805080d072bcc52ac5280e8206683eff99c11d184f"

SELECTED_BUY_PARAMS: dict[str, Any] = {
    "breakout_buffer_atr": 0.3,
    "pressure_min": 0.2,
    "pressure_window": 48,
    "use_pressure_guard": True,
    "use_state_guard": False,
    "use_volume_guard": True,
    "volume_guard_window": 24,
    "volume_ratio_min": 1.6,
}
ARCHIVED_BUY_DEFAULTS: dict[str, Any] = {
    **EFFECTIVE_BUY_LOCK,
    "use_volume_guard": False,
    "volume_ratio_min": 1.0,
    "use_pressure_guard": False,
    "pressure_window": 24,
    "pressure_min": 0.15,
    "breakout_buffer_atr": 0.15,
}


def _json_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


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
        self.entry_side = "sell"
        self.exit_side = "buy"
        self.open_rate = 100.0
        self.open_date_utc = datetime(2024, 1, 1, 1, tzinfo=UTC)
        self.date_entry_fill_utc = datetime(2024, 1, 1, 1, tzinfo=UTC)
        self.min_rate = 100.0
        self.stake_amount = 1000.0
        self.amount = 10.0
        self.leverage = 1.0
        self.is_short = True
        self.is_open = True
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


def _strategy() -> Sieve3V2IntegratedFromBosBearContinuationShort1H:
    return Sieve3V2IntegratedFromBosBearContinuationShort1H({})


def _base_state(plan: str = "measured_half_touch_p33_be_reclaim2") -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "plan": plan,
        "phase": "ENTRY",
        "entry_rate": 100.0,
        "entry_candle": "2024-01-01T00:00:00+00:00",
        "entry_filled_at": "2024-01-01T01:00:00+00:00",
        "broken_support": 101.0,
        "structural_invalidation": 110.0,
        "targets": {
            "measured_half": 96.5,
            "measured_full": 92.0,
            "swing": 94.0,
            "prior96": 90.0,
            "vp_lvn": 95.0,
            "vp_poc": 97.0,
            "nearest": 97.0,
        },
        "target_providers": {},
        "target_1_touched_at": None,
        "target_2_touched_at": None,
        "closed_favorable_low": None,
        "partial_filled": False,
        "partial_filled_at": None,
        "partial_tag": None,
        "partial_target_stake": None,
        "partial_realized_stake": 0.0,
        "partial_fill_order_ids": [],
        "terminal_fill": False,
        "terminal_exit_pending": False,
        "terminal_exit_tag": None,
        "stop_ceiling": None,
    }


def _load_snapshot_module(source_bytes: bytes, directory: str) -> ModuleType:
    module_path = Path(directory) / "authoritative_bos_snapshot.py"
    module_path.write_bytes(source_bytes)
    spec = importlib.util.spec_from_file_location("authoritative_bos_snapshot", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_authoritative_promotion_hashes_and_effective_lock() -> None:
    with RESULT_PATH.open("r", encoding="utf-8-sig") as handle:
        raw_row = handle.readlines()[34]
    assert hashlib.sha256(raw_row.encode("utf-8")).hexdigest() == SOURCE_RESULT_ROW_SHA256
    row = json.loads(raw_row)
    assert row["status"] == "ok"
    assert row["strategy"] == "sieve2_bos_bear_continuation_short_1h"
    assert row["job_id"] == "20260521T012740_entry_all_resume"
    assert row["training_window"] == "pattern_continuation_1h_2021_22"

    assert hashlib.sha256(ARCHIVE_PATH.read_bytes()).hexdigest() == SOURCE_BACKTEST_ARCHIVE_SHA256
    with ZipFile(ARCHIVE_PATH) as archive:
        source_bytes = archive.read(SNAPSHOT_STRATEGY_MEMBER)
        params_bytes = archive.read(SNAPSHOT_PARAMS_MEMBER)
    assert hashlib.sha256(source_bytes).hexdigest() == SOURCE_SNAPSHOT_STRATEGY_SHA256
    assert hashlib.sha256(params_bytes).hexdigest() == SOURCE_SNAPSHOT_PARAMS_SHA256
    assert json.loads(params_bytes)["params"]["buy"] == SELECTED_BUY_PARAMS

    assert _json_hash(ARCHIVED_BUY_DEFAULTS) == ARCHIVED_DEFAULTS_SHA256
    assert _json_hash(SELECTED_BUY_PARAMS) == SELECTED_BUY_PARAMS_SHA256
    assert {**ARCHIVED_BUY_DEFAULTS, **SELECTED_BUY_PARAMS} == EFFECTIVE_BUY_LOCK
    assert _json_hash(EFFECTIVE_BUY_LOCK) == SOURCE_EFFECTIVE_BUY_LOCK_SHA256
    assert SOURCE_BASELINE_ENTRY_SIGNATURE_SHA256 == (
        "4cfa1adb917d25bc350f21803d65f3fb9d257fd68651ac17dd30516e82b96df4"
    )


def test_standalone_contract_exact_defaults_and_sell_only_surface() -> None:
    strategy = _strategy()
    assert strategy.__class__.__bases__ == (IStrategy,)
    source = STRATEGY_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    strategy_imports = [
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module
        and node.module.startswith("user_data.strategies")
    ]
    assert strategy_imports == []

    assert strategy.can_short is True
    assert strategy.use_exit_signal is True
    assert strategy.use_custom_stoploss is True
    assert strategy.use_custom_roi is True
    assert strategy.position_adjustment_enable is True
    assert strategy.max_entry_position_adjustment == 0
    assert strategy.trailing_stop is False
    assert strategy.minimal_roi == {}
    assert strategy.exit_profit_only is False

    for name, expected in EFFECTIVE_BUY_LOCK.items():
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
    active = {name for name, parameter in parameters.items() if parameter.optimize}
    assert active == {"exit_policy_plan"}
    assert strategy.exit_policy_plan.space == "sell"


def test_entry_parity_against_authoritative_archived_source() -> None:
    with ZipFile(ARCHIVE_PATH) as archive:
        source_bytes = archive.read(SNAPSHOT_STRATEGY_MEMBER)
    assert hashlib.sha256(source_bytes).hexdigest() == SOURCE_SNAPSHOT_STRATEGY_SHA256

    with TemporaryDirectory() as temp_dir:
        module = _load_snapshot_module(source_bytes, temp_dir)
        source = module.Sieve2BOSBearContinuationShort1H({})
        for name, value in EFFECTIVE_BUY_LOCK.items():
            getattr(source, name).value = value

        frame = pd.read_feather(DATA_PATH)
        frame = frame.loc[
            frame["date"].ge(pd.Timestamp("2024-01-01", tz="UTC"))
            & frame["date"].lt(pd.Timestamp("2025-04-01", tz="UTC"))
        ].reset_index(drop=True)
        metadata = {"pair": "BTC/USDT:USDT"}
        candidate = _strategy()
        source_frame = source.populate_entry_trend(
            source.populate_indicators(frame.copy(), metadata), metadata
        )
        candidate_frame = candidate.populate_entry_trend(
            candidate.populate_indicators(frame.copy(), metadata), metadata
        )

    assert int(source_frame["enter_short"].sum()) == 1
    pd.testing.assert_series_equal(source_frame["enter_short"], candidate_frame["enter_short"])
    pd.testing.assert_series_equal(
        source_frame["enter_tag"].fillna(""),
        candidate_frame["enter_tag"].fillna(""),
    )
    assert set(
        (
            "ms_break_level",
            "ms_invalidation_level",
            "ms_prev_swing_low",
            "s2vp_lvn_below",
            "s2vp_poc",
            "s2vp_val",
        )
    ).issubset(candidate_frame.columns)


def test_freqtrade_strategy_resolver_loads_v2_module() -> None:
    with TemporaryDirectory() as temp_dir:
        copy2(STRATEGY_PATH, Path(temp_dir) / STRATEGY_PATH.name)
        config = {
            "strategy": "Sieve3V2IntegratedFromBosBearContinuationShort1H",
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
    assert strategy.can_short is True
    assert strategy.use_custom_roi is True
    assert strategy.minimal_roi == {}


def test_plan_count_reachability_signatures_and_name_alignment() -> None:
    strategy_class = Sieve3V2IntegratedFromBosBearContinuationShort1H
    assert len(EXIT_PLANS) == 32
    assert len(EXIT_PLANS) == len(set(EXIT_PLANS.values()))
    assert len(EXIT_PLAN_SIGNATURES) == len(set(EXIT_PLAN_SIGNATURES.values()))
    assert {plan.role for plan in EXIT_PLANS.values()} >= {
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
    integrated_roles = {
        "target_partial",
        "invalidation_reduce",
        "target_tighten",
        "dual_target",
        "progress_failure",
    }
    assert sum(plan.role in integrated_roles for plan in EXIT_PLANS.values()) >= 16

    pure_targets = [plan for plan in EXIT_PLANS.values() if plan.role == "target_full"]
    assert pure_targets and all(
        plan.invalidation == "none" and plan.invalidation_action == "hold"
        for plan in pure_targets
    )
    for name, plan in EXIT_PLANS.items():
        assert EXIT_PLAN_SIGNATURES[name] == plan.signature()
        empty_state = {"partial_pending": False, "partial_filled": False}
        if plan.role == "baseline":
            assert plan.fixed_tp > 0.0, name
            continue
        if plan.target_1:
            assert plan.target_1 in name, name
        if plan.target_2:
            assert plan.target_2 in name, name
        if plan.confirmation != "touch":
            assert plan.confirmation in name, name
        if plan.partial_fraction == 0.33:
            assert "33" in name, name
        if plan.partial_fraction == 0.50:
            assert "50" in name, name
        if plan.remainder == "breakeven":
            assert "_be_" in name, name
        if plan.remainder == "trail":
            assert "trail" in name or "runner" in name, name
        if plan.progress_candles:
            assert f"progress{plan.progress_candles}" in name, name

        if plan.target_action != "hold":
            decision = strategy_class.evaluate_policy(
                plan, empty_state, {"target_1": True, "profit_bucket": "flat"}
            )
            assert decision.action == plan.target_action, name
        if plan.invalidation != "none" and plan.invalidation_action != "hold":
            decision = strategy_class.evaluate_policy(
                plan, empty_state, {"invalidation": True, "profit_bucket": "flat"}
            )
            assert decision.action == plan.invalidation_action, name
        if plan.target_2:
            decision = strategy_class.evaluate_policy(
                plan,
                {"partial_pending": False, "partial_filled": True},
                {"target_2": True},
            )
            assert decision.action == "full", name
        if plan.progress_candles:
            decision = strategy_class.evaluate_policy(
                plan, empty_state, {"time_failure": True}
            )
            assert decision.tag == "s3v2_progress_failure", name


def test_short_target_and_invalidation_direction_semantics() -> None:
    strategy = _strategy()
    row = pd.Series(
        {
            "ms_break_level": 101.0,
            "ms_invalidation_level": 110.0,
            "ms_last_swing_high": 109.0,
            "ms_prev_swing_low": 94.0,
            "s3v2_prior_low_48": 93.0,
            "s3v2_prior_low_96": 90.0,
            "s2vp_hvn_below": 95.0,
            "s2vp_lvn_below": 96.0,
            "s2vp_poc": 97.0,
            "s2vp_val": 92.0,
        }
    )
    targets, providers, broken, invalidation = strategy._frozen_targets(row, 100.0)
    assert broken == 101.0
    assert invalidation == 110.0
    assert targets["measured_half"] == 96.5
    assert targets["measured_full"] == 92.0
    assert targets["nearest"] == 97.0
    assert providers["nearest"] == "vp_poc"
    assert all(value is None or value < 100.0 for value in targets.values())
    assert strategy._directional_target(101.0, 100.0) is None
    assert strategy._directional_invalidation(99.0, 100.0) is None

    assert strategy._invalidation_event(
        "combined1", True, False, True, False, False, True
    )
    assert not strategy._invalidation_event(
        "reclaim2", True, False, True, False, False, True
    )
    stop = strategy._desired_stop_price(
        EXIT_PLANS["measured_half_touch_p33_be_reclaim2"],
        _base_state(),
        ExitDecision("hold"),
        100.0,
    )
    assert stop > 100.0


def test_placement_freeze_and_delayed_fill_anchor_are_idempotent() -> None:
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"], utc=True
            ),
            "ms_break_level": [101.0, 500.0],
            "ms_invalidation_level": [110.0, 600.0],
            "ms_last_swing_high": [109.0, 590.0],
            "ms_prev_swing_low": [94.0, 400.0],
            "s3v2_prior_low_48": [93.0, 390.0],
            "s3v2_prior_low_96": [90.0, 380.0],
            "s2vp_hvn_below": [95.0, 370.0],
            "s2vp_lvn_below": [96.0, 360.0],
            "s2vp_poc": [97.0, 350.0],
            "s2vp_val": [92.0, 340.0],
        }
    )
    strategy = _strategy()
    strategy.exit_policy_plan.value = "progress48_measured_half"
    strategy.dp = FakeDataProvider(frame)
    trade = FakeTrade()
    delayed_fill = datetime(2024, 1, 1, 5, tzinfo=UTC)
    order = FakeOrder(
        "sell",
        safe_price=100.0,
        order_date_utc=datetime(2024, 1, 1, 1, tzinfo=UTC),
        order_filled_utc=delayed_fill,
    )
    strategy.order_filled(trade.pair, trade, order, delayed_fill)
    state = trade.get_custom_data(STATE_KEY)
    assert state["broken_support"] == 101.0
    assert state["structural_invalidation"] == 110.0
    assert state["targets"]["measured_half"] == 96.5
    assert state["entry_filled_at"] == delayed_fill.isoformat()
    assert state["terminal_exit_tag"] is None

    changed_order = FakeOrder("sell", safe_price=200.0, order_id="changed")
    strategy.order_filled(trade.pair, trade, changed_order, delayed_fill)
    assert trade.get_custom_data(STATE_KEY) == state

    event_frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01T01:00:00Z", periods=8, freq="1h"),
            "open": 100.0,
            "high": 101.0,
            "low": [80.0, 80.0, 80.0, 80.0, 99.0, 99.0, 99.0, 99.0],
            "close": 100.0,
            "ms_choch_to_bull": False,
            "ms_bos_to_bull": False,
            "ms_state": -1.0,
            "s2vp_context_score_bull": 0.1,
            "s2vp_context_score_bear": 0.5,
            "s2vp_score_long": 0.1,
            "s2vp_score_short": 0.5,
            "s2vp_market_context": -0.5,
        }
    )
    events, updated = strategy._events(state, event_frame, 0.0)
    assert events["target_1"] is False
    assert events["time_failure"] is False
    assert updated["target_1_touched_at"] is None
    assert updated["closed_favorable_low"] == 99.0


def test_partial_fill_idempotency_reconciliation_and_terminal_quantity_safety() -> None:
    strategy = _strategy()
    trade = FakeTrade()
    state = _base_state()
    state["phase"] = "TARGET_ZONE"
    state["partial_target_stake"] = 330.0
    trade.set_custom_data(STATE_KEY, state)

    for tag, filled in (("wrong", 1.0), ("s3v2_partial_target", 0.0)):
        strategy.order_filled(
            trade.pair,
            trade,
            FakeOrder("buy", tag, safe_filled=filled),
            datetime.now(UTC),
        )
        assert trade.get_custom_data(STATE_KEY)["partial_filled"] is False

    strategy.order_filled(
        trade.pair,
        trade,
        FakeOrder(
            "buy",
            "s3v2_partial_target",
            safe_filled=1.65,
            safe_amount=3.3,
            order_id="partial-half-1",
        ),
        datetime(2024, 1, 2, 3, tzinfo=UTC),
    )
    state = trade.get_custom_data(STATE_KEY)
    assert state["partial_filled"] is False
    assert state["partial_realized_stake"] == 165.0

    trade.stake_amount = 835.0
    strategy._decision_context = lambda *args, **kwargs: (  # type: ignore[method-assign]
        EXIT_PLANS["measured_half_touch_p33_be_reclaim2"],
        trade.get_custom_data(STATE_KEY),
        {},
        ExitDecision("partial", "s3v2_partial_target", 0.33),
    )
    request = strategy.adjust_trade_position(
        trade,
        datetime.now(UTC),
        95.0,
        0.05,
        None,
        1000.0,
        100.0,
        95.0,
        0.0,
        0.05,
    )
    assert request == (-165.0, "s3v2_partial_target")

    filled_at = datetime(2024, 1, 2, 4, tzinfo=UTC)
    completing_order = FakeOrder(
        "buy",
        "s3v2_partial_target",
        safe_filled=1.65,
        safe_amount=1.65,
        order_filled_utc=filled_at,
        order_id="partial-half-2",
    )
    strategy.order_filled(trade.pair, trade, completing_order, filled_at)
    completed = trade.get_custom_data(STATE_KEY)
    assert completed["partial_filled"] is True
    assert completed["phase"] == "REMAINDER"
    assert completed["partial_tag"] in PARTIAL_TAGS
    assert completed["partial_filled_at"] == filled_at.isoformat()

    strategy.order_filled(trade.pair, trade, completing_order, datetime(2024, 1, 2, 5, tzinfo=UTC))
    assert trade.get_custom_data(STATE_KEY)["partial_filled_at"] == filled_at.isoformat()

    terminal_trade = FakeTrade()
    terminal_state = _base_state()
    terminal_state["partial_target_stake"] = 500.0
    terminal_state["partial_realized_stake"] = 400.0
    terminal_trade.stake_amount = 80.0
    terminal_trade.amount = 0.8
    terminal_trade.set_custom_data(STATE_KEY, terminal_state)
    strategy._decision_context = lambda *args, **kwargs: (  # type: ignore[method-assign]
        EXIT_PLANS["measured_half_reversal1_p50_trail_combined1"],
        terminal_trade.get_custom_data(STATE_KEY),
        {},
        ExitDecision("partial", "s3v2_partial_target", 0.50),
    )
    terminal_request = strategy.adjust_trade_position(
        terminal_trade,
        datetime.now(UTC),
        90.0,
        0.10,
        None,
        1000.0,
        100.0,
        90.0,
        0.0,
        0.10,
    )
    assert terminal_request == (-80.0, "s3v2_partial_target")

    terminal_trade.stake_amount = 0.0
    terminal_trade.amount = 0.0
    terminal_trade.is_open = False
    strategy.order_filled(
        terminal_trade.pair,
        terminal_trade,
        FakeOrder(
            "buy",
            "s3v2_partial_target",
            safe_filled=0.8,
            safe_amount=0.8,
            order_id="terminal-partial",
        ),
        datetime(2024, 1, 2, 6, tzinfo=UTC),
    )
    terminal = terminal_trade.get_custom_data(STATE_KEY)
    assert terminal["phase"] == "TERMINAL"
    assert terminal["terminal_fill"] is True
    assert terminal["terminal_exit_pending"] is False


def test_pending_partial_defers_transient_plan_invalidation_until_order_clears() -> None:
    strategy = _strategy()
    trade = FakeTrade()
    state = _base_state("reclaim1_full")
    state["phase"] = "TARGET_ZONE"
    state["partial_target_stake"] = 330.0
    trade.set_custom_data(STATE_KEY, state)
    trade.orders.append(
        FakeOrder(
            "buy",
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
                "open": [101.0],
                "high": [103.0],
                "low": [100.0],
                "close": [102.0],
                "ms_choch_to_bull": [False],
                "ms_bos_to_bull": [False],
                "ms_state": [-1.0],
                "s2vp_context_score_bull": [0.1],
                "s2vp_context_score_bear": [0.5],
                "s2vp_score_long": [0.1],
                "s2vp_score_short": [0.5],
                "s2vp_market_context": [-0.5],
            }
        )
    )
    assert (
        strategy.custom_exit(
            trade.pair,
            trade,
            datetime(2024, 1, 1, 2, tzinfo=UTC),
            102.0,
            -0.02,
        )
        is None
    )
    pending_state = trade.get_custom_data(STATE_KEY)
    assert pending_state["terminal_exit_pending"] is True
    assert pending_state["terminal_exit_tag"] == "s3v2_plan_invalidation"
    assert (
        strategy.custom_exit(
            trade.pair,
            trade,
            datetime(2024, 1, 1, 2, tzinfo=UTC),
            102.0,
            -0.02,
        )
        is None
    )
    assert trade.get_custom_data(STATE_KEY)["terminal_exit_tag"] == pending_state[
        "terminal_exit_tag"
    ]

    trade.orders.clear()
    strategy.dp = FakeDataProvider(
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01T02:00:00Z"], utc=True),
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "ms_choch_to_bull": [False],
                "ms_bos_to_bull": [False],
                "ms_state": [-1.0],
                "s2vp_context_score_bull": [0.1],
                "s2vp_context_score_bear": [0.5],
                "s2vp_score_long": [0.1],
                "s2vp_score_short": [0.5],
                "s2vp_market_context": [-0.5],
            }
        )
    )
    assert strategy.custom_exit(
        trade.pair,
        trade,
        datetime(2024, 1, 1, 3, tzinfo=UTC),
        100.0,
        0.0,
    ) == "s3v2_plan_invalidation"


def test_pending_partial_defers_hard_invalidation_until_order_clears() -> None:
    strategy = _strategy()
    trade = FakeTrade()
    state = _base_state()
    state["phase"] = "TARGET_ZONE"
    state["partial_target_stake"] = 330.0
    trade.set_custom_data(STATE_KEY, state)
    trade.orders.append(
        FakeOrder(
            "buy",
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
                "open": [108.0],
                "high": [115.0],
                "low": [107.0],
                "close": [112.0],
                "ms_choch_to_bull": [True],
                "ms_bos_to_bull": [False],
                "ms_state": [1.0],
                "s2vp_context_score_bull": [0.5],
                "s2vp_context_score_bear": [0.1],
                "s2vp_score_long": [0.5],
                "s2vp_score_short": [0.1],
                "s2vp_market_context": [0.5],
            }
        )
    )
    assert (
        strategy.custom_exit(
            trade.pair,
            trade,
            datetime(2024, 1, 1, 2, tzinfo=UTC),
            112.0,
            -0.12,
        )
        is None
    )
    pending_state = trade.get_custom_data(STATE_KEY)
    assert pending_state["terminal_exit_pending"] is True
    assert pending_state["terminal_exit_tag"] == "s3v2_hard_invalidation"

    filled_at = datetime(2024, 1, 1, 2, 30, tzinfo=UTC)
    trade.stake_amount = 670.0
    trade.amount = 6.7
    strategy.order_filled(
        trade.pair,
        trade,
        FakeOrder(
            "buy",
            "s3v2_partial_target",
            safe_price=100.0,
            safe_filled=3.3,
            order_filled_utc=filled_at,
            order_id="pending-partial",
        ),
        filled_at,
    )
    trade.orders.clear()
    filled_state = trade.get_custom_data(STATE_KEY)
    assert filled_state["partial_filled"] is True
    assert filled_state["phase"] == "REMAINDER"
    assert filled_state["terminal_exit_tag"] == "s3v2_hard_invalidation"
    strategy.dp = FakeDataProvider(
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01T02:00:00Z"], utc=True),
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "ms_choch_to_bull": [False],
                "ms_bos_to_bull": [False],
                "ms_state": [-1.0],
                "s2vp_context_score_bull": [0.1],
                "s2vp_context_score_bear": [0.5],
                "s2vp_score_long": [0.1],
                "s2vp_score_short": [0.5],
                "s2vp_market_context": [-0.5],
            }
        )
    )
    assert strategy.custom_exit(
        trade.pair,
        trade,
        datetime(2024, 1, 1, 3, tzinfo=UTC),
        100.0,
        0.0,
    ) == "s3v2_hard_invalidation"


def test_legacy_pending_terminal_exit_defaults_to_hard_invalidation_tag() -> None:
    strategy = _strategy()
    trade = FakeTrade()
    legacy_state = _base_state()
    legacy_state.pop("terminal_exit_tag")
    legacy_state["terminal_exit_pending"] = True
    trade.set_custom_data(STATE_KEY, legacy_state)
    strategy.dp = FakeDataProvider(pd.DataFrame())

    assert strategy.custom_exit(
        trade.pair,
        trade,
        datetime(2024, 1, 1, 3, tzinfo=UTC),
        100.0,
        0.0,
    ) == "s3v2_hard_invalidation"
    migrated = trade.get_custom_data(STATE_KEY)
    assert migrated["terminal_exit_pending"] is True
    assert migrated["terminal_exit_tag"] == "s3v2_hard_invalidation"


def test_closed_candle_target_confirmation_and_callback_causality() -> None:
    strategy = _strategy()
    trade = FakeTrade()
    state = _base_state("measured_half_touch_full")
    state["targets"]["measured_half"] = 90.0
    trade.set_custom_data(STATE_KEY, state)
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2024-01-01T01:00:00Z", "2024-01-01T02:00:00Z"], utc=True
            ),
            "open": [100.0, 96.0],
            "high": [101.0, 97.0],
            "low": [95.0, 85.0],
            "close": [96.0, 90.0],
            "ms_choch_to_bull": False,
            "ms_bos_to_bull": False,
            "ms_state": -1.0,
            "s2vp_context_score_bull": 0.1,
            "s2vp_context_score_bear": 0.5,
            "s2vp_score_long": 0.1,
            "s2vp_score_short": 0.5,
            "s2vp_market_context": -0.5,
        }
    )
    strategy.dp = FakeDataProvider(frame)
    assert (
        strategy.custom_exit(
            trade.pair,
            trade,
            datetime(2024, 1, 1, 2, tzinfo=UTC),
            96.0,
            0.04,
        )
        is None
    )
    assert trade.get_custom_data(STATE_KEY)["target_1_touched_at"] is None
    assert strategy.custom_exit(
        trade.pair,
        trade,
        datetime(2024, 1, 1, 3, tzinfo=UTC),
        90.0,
        0.10,
    ) == "s3v2_target_full"

    dates = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
    reversal_frame = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0, 100.0, 100.0, 99.9, 99.8, 99.9],
            "low": [99.9, 99.9, 99.9, 99.75, 99.7, 99.8],
            "close": [100.1, 100.1, 100.1, 99.8, 99.9, 100.0],
        }
    )
    untouched_at, confirmed = strategy._confirmation(
        reversal_frame.iloc[:3], 99.8, "touch", None, 0.03, 100.0
    )
    assert untouched_at is None and confirmed is False
    touched_at, confirmed = strategy._confirmation(
        reversal_frame.iloc[:5], 99.8, "reversal2of3", None, 0.03, 100.0
    )
    assert touched_at == dates[3].isoformat()
    assert confirmed is False
    same_touch, confirmed = strategy._confirmation(
        reversal_frame, 99.8, "reversal2of3", touched_at, 0.03, 100.0
    )
    assert same_touch == touched_at and confirmed is True


def test_short_stop_ceiling_is_executable_monotonic_and_freqtrade_applied() -> None:
    strategy = _strategy()
    trade = FakeTrade()
    state = _base_state("measured_half_reversal1_p50_trail_combined1")
    state["phase"] = "REMAINDER"
    state["partial_filled"] = True
    state["partial_filled_at"] = "2024-01-01T10:00:00+00:00"
    state["closed_favorable_low"] = 80.0
    trade.set_custom_data(STATE_KEY, state)
    strategy.custom_stoploss(trade.pair, trade, datetime.now(UTC), 80.0, 0.20, False)
    first_ceiling = trade.get_custom_data(STATE_KEY)["stop_ceiling"]
    assert first_ceiling == 80.0 * 1.015
    assert first_ceiling > 80.0

    state = trade.get_custom_data(STATE_KEY)
    state["closed_favorable_low"] = 85.0
    trade.set_custom_data(STATE_KEY, state)
    strategy.custom_stoploss(trade.pair, trade, datetime.now(UTC), 75.0, 0.25, True)
    assert trade.get_custom_data(STATE_KEY)["stop_ceiling"] == first_ceiling
    assert strategy.custom_exit(
        trade.pair, trade, datetime.now(UTC), 82.0, 0.18
    ) == "s3v2_stop_ceiling_breached"

    with FtNoDBContext("1h"):
        ft_trade = Trade(
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
            is_short=True,
        )
        ft_state = _base_state("measured_half_reversal1_p50_trail_combined1")
        ft_state["phase"] = "REMAINDER"
        ft_state["partial_filled"] = True
        ft_state["partial_filled_at"] = "2024-01-01T10:00:00+00:00"
        ft_state["closed_favorable_low"] = 80.0
        ft_trade.set_custom_data(STATE_KEY, ft_state)
        strategy.ft_stoploss_adjust(
            80.0,
            ft_trade,
            datetime(2024, 1, 2, tzinfo=UTC),
            0.20,
            0,
        )
        assert ft_trade.stop_loss == 81.19


def test_unclosed_same_candle_low_cannot_tighten_callback_stop() -> None:
    strategy = _strategy()
    trade = FakeTrade()
    state = _base_state("measured_half_reversal1_p50_trail_combined1")
    state["phase"] = "REMAINDER"
    state["partial_filled"] = True
    state["partial_filled_at"] = "2024-01-01T01:00:00+00:00"
    trade.set_custom_data(STATE_KEY, state)
    strategy.dp = FakeDataProvider(
        pd.DataFrame(
            {
                "date": pd.to_datetime(
                    ["2024-01-01T01:00:00Z", "2024-01-01T02:00:00Z"], utc=True
                ),
                "open": [100.0, 90.0],
                "high": [101.0, 91.0],
                "low": [90.0, 70.0],
                "close": [92.0, 72.0],
                "ms_choch_to_bull": False,
                "ms_bos_to_bull": False,
                "ms_state": -1.0,
                "s2vp_context_score_bull": 0.1,
                "s2vp_context_score_bear": 0.5,
                "s2vp_score_long": 0.1,
                "s2vp_score_short": 0.5,
                "s2vp_market_context": -0.5,
            }
        )
    )
    strategy.custom_stoploss(
        trade.pair,
        trade,
        datetime(2024, 1, 1, 2, tzinfo=UTC),
        90.0,
        0.10,
        False,
    )
    assert trade.get_custom_data(STATE_KEY)["closed_favorable_low"] == 90.0
    assert trade.get_custom_data(STATE_KEY)["stop_ceiling"] == 90.0 * 1.015

    strategy.custom_stoploss(
        trade.pair,
        trade,
        datetime(2024, 1, 1, 3, tzinfo=UTC),
        70.0,
        0.30,
        False,
    )
    assert trade.get_custom_data(STATE_KEY)["closed_favorable_low"] == 70.0
    assert trade.get_custom_data(STATE_KEY)["stop_ceiling"] == 70.0 * 1.015


def test_entry_driving_and_target_indicators_are_prefix_invariant() -> None:
    frame = pd.read_feather(DATA_PATH).iloc[5000:6000].reset_index(drop=True)
    strategy = _strategy()
    metadata = {"pair": "BTC/USDT:USDT"}
    short = frame.iloc[:700].copy()
    short_out = strategy.populate_indicators(short, metadata)
    full_out = strategy.populate_indicators(frame.copy(), metadata).iloc[:700].reset_index(
        drop=True
    )
    for column in (
        "ms_bos_to_bear",
        "ms_choch_to_bull",
        "ms_state",
        "ms_break_level",
        "ms_invalidation_level",
        "ms_prev_swing_low",
        "s2m_pressure_ratio",
        "s2vp_score_short",
        "s2vp_context_score_bear",
        "s2vp_lvn_below",
        "s2vp_poc",
        "s2vp_val",
        "s3v2_prior_low_48",
        "s3v2_prior_low_96",
    ):
        if pd.api.types.is_bool_dtype(short_out[column]):
            pd.testing.assert_series_equal(
                short_out[column].reset_index(drop=True),
                full_out[column].reset_index(drop=True),
                check_names=False,
            )
        else:
            np.testing.assert_allclose(
                pd.to_numeric(short_out[column], errors="coerce"),
                pd.to_numeric(full_out[column], errors="coerce"),
                equal_nan=True,
            )


def test_all_plans_have_native_roi_fallback_and_bounded_hold() -> None:
    strategy = _strategy()
    trade = FakeTrade()
    for plan_name, expected in (("baseline_3_3", 0.03), ("promotion_2_2", 0.02)):
        trade.set_custom_data(STATE_KEY, {"version": STATE_VERSION, "plan": plan_name})
        assert (
            strategy.custom_roi(trade.pair, trade, datetime.now(UTC), 10, ENTRY_TAG, "short")
            == expected
        )
    trade.set_custom_data(
        STATE_KEY, {"version": STATE_VERSION, "plan": "measured_half_touch_full"}
    )
    assert (
        strategy.custom_roi(trade.pair, trade, datetime.now(UTC), 10, ENTRY_TAG, "short")
        == 0.08
    )
    for name, plan in EXIT_PLANS.items():
        if plan.role == "baseline":
            assert plan.max_hold_candles == 0, name
        else:
            assert plan.fixed_tp == 0.08, name
            assert plan.max_hold_candles == 336, name
    assert Sieve3V2IntegratedFromBosBearContinuationShort1H.evaluate_policy(
        EXIT_PLANS["measured_half_touch_full"],
        {"partial_pending": False, "partial_filled": False},
        {"max_hold": True},
    ) == ExitDecision("full", "s3v2_max_hold")
