# ruff: noqa: S101

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from shutil import copy2
from tempfile import TemporaryDirectory
from typing import Any
from zipfile import ZipFile

import numpy as np
import pandas as pd

from freqtrade.enums import ExitType
from freqtrade.persistence import Trade
from freqtrade.persistence.usedb_context import FtNoDBContext
from freqtrade.resolvers import StrategyResolver
from freqtrade.strategy import IStrategy
from user_data.strategies import (
    sieve3_V2_integrated_from_mtf_std_daily_macd_volume_breakout_long_1h as strategy_module,
)


ENTRY_SOURCE_STAGE = strategy_module.ENTRY_SOURCE_STAGE
EXIT_HYPOTHESIS = strategy_module.EXIT_HYPOTHESIS
EXIT_PLANS = strategy_module.EXIT_PLANS
PARTIAL_TAGS = strategy_module.PARTIAL_TAGS
RESEARCH_PATH = strategy_module.RESEARCH_PATH
SIEVE_STAGE = strategy_module.SIEVE_STAGE
SOURCE_RESULT_BATCH = strategy_module.SOURCE_RESULT_BATCH
SOURCE_STRATEGY = strategy_module.SOURCE_STRATEGY
STATE_KEY = strategy_module.STATE_KEY
STATE_VERSION = strategy_module.STATE_VERSION
TARGET_PROVIDER_CHAINS = strategy_module.TARGET_PROVIDER_CHAINS
ExitDecision = strategy_module.ExitDecision
Sieve3V2IntegratedFromMtfStdDailyMacdVolumeBreakoutLong1H = (
    strategy_module.Sieve3V2IntegratedFromMtfStdDailyMacdVolumeBreakoutLong1H
)


STRATEGY_PATH = Path(
    "user_data/strategies/sieve3_V2_integrated_from_mtf_std_daily_macd_volume_breakout_long_1h.py"
)
DATA_PATH = Path("user_data/data/binance/futures/BTC_USDT_USDT-1h-futures.feather")
DATA_4H_PATH = Path("user_data/data/binance/futures/BTC_USDT_USDT-4h-futures.feather")
DATA_1D_PATH = Path("user_data/data/binance/futures/BTC_USDT_USDT-1d-futures.feather")
ORACLE_ZIP_PATH = Path(
    "user_data/Custom_Launcher/launcher_v2/runtime/entry_sieve/backtests/"
    "sieve2_mtf_std_daily_macd_volume_breakout_long_1h__auto_generic_1h_2020_q2_q3/"
    "full_cycle_2020_2026/tp_3_sl_3/backtest-result-2026-06-12_08-38-35.zip"
)
ORACLE_SOURCE_SHA256 = "c8e78309683f277548e9ac9e3a929ae475a6daa4c197d6060eed4eaf7455b18f"
ORACLE_RESULT_SHA256 = "ea1713135c6646f6db870cc7fa3e3e673310ec70061eb31f627e49d1d1960f09"
ORACLE_LOCK_SHA256 = "932d20bed431b8f58d4b8bc0309fa9f289fdc9ea66e3a1fdf2df503932b15e02"
EFFECTIVE_BUY_LOCK_SHA256 = "e97f93bfde27c64de9ffa158a06ce4f9017363dcddcd9fabea71071cdbcf0a58"
ORACLE_CLASS = "Sieve2MtfStdDailyMacdVolumeBreakoutLong1h"


LOCKED_BUY_DEFAULTS: dict[str, Any] = {
    "bb_len": 52,
    "bb_width_max": 0.13,
    "ema_fast_len": 43,
    "ema_slow_len": 139,
    "retest_buffer_pct": 0.026,
    "rsi_len": 23,
    "rsi_long_min": 42,
    "rsi_short_max": 41,
    "use_daily_trend": False,
    "use_h4_compression": False,
    "use_momentum_filter": True,
    "use_retest": True,
    "use_volume_filter": True,
    "volume_ratio_min": 1.67,
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
            order for order in self.orders if order.ft_is_open and order.ft_order_side != "stoploss"
        ]

    def get_custom_data(self, key: str) -> Any:
        return self._custom.get(key)

    def set_custom_data(self, key: str, value: Any) -> None:
        self._custom[key] = value


class FakeDataProvider:
    def __init__(
        self,
        frame: pd.DataFrame,
        informative: dict[str, pd.DataFrame] | None = None,
    ) -> None:
        self.frame = frame
        self.informative = informative or {}

    def get_analyzed_dataframe(self, pair: str, timeframe: str):
        _ = pair, timeframe
        return self.frame.copy(), None

    def current_whitelist(self) -> list[str]:
        return ["BTC/USDT:USDT"]

    def get_pair_dataframe(self, pair: str, timeframe: str) -> pd.DataFrame:
        _ = pair
        return self.informative[timeframe].copy()


def _oracle_members() -> tuple[bytes, bytes, dict[str, Any]]:
    with ZipFile(ORACLE_ZIP_PATH) as archive:
        names = archive.namelist()
        source_name = next(name for name in names if name.endswith(f"_{ORACLE_CLASS}.py"))
        lock_name = next(name for name in names if name.endswith(f"_{ORACLE_CLASS}.json"))
        result_name = next(
            name
            for name in names
            if name.endswith(".json")
            and not name.endswith("_config.json")
            and not name.endswith(f"_{ORACLE_CLASS}.json")
        )
        return (
            archive.read(source_name),
            archive.read(lock_name),
            json.loads(archive.read(result_name)),
        )


def _oracle_strategy() -> IStrategy:
    source, lock_bytes, _ = _oracle_members()
    with TemporaryDirectory() as temp_dir:
        Path(temp_dir, "sieve2_mtf_std_daily_macd_volume_breakout_long_1h.py").write_bytes(source)
        strategy = StrategyResolver.load_strategy(
            {
                "strategy": ORACLE_CLASS,
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
        )
    locked = json.loads(lock_bytes)["params"]["buy"]
    for name, value in locked.items():
        getattr(strategy, name).value = value
    return strategy


def _historical_provider(frame: pd.DataFrame) -> FakeDataProvider:
    return FakeDataProvider(
        frame,
        {
            "4h": pd.read_feather(DATA_4H_PATH),
            "1d": pd.read_feather(DATA_1D_PATH),
        },
    )


def _strategy() -> Sieve3V2IntegratedFromMtfStdDailyMacdVolumeBreakoutLong1H:
    return Sieve3V2IntegratedFromMtfStdDailyMacdVolumeBreakoutLong1H({})


def test_authoritative_oracle_hashes_and_three_four_execution_correction() -> None:
    source, lock_bytes, result = _oracle_members()
    executed = result["strategy"][ORACLE_CLASS]

    assert sha256(ORACLE_ZIP_PATH.read_bytes()).hexdigest() == ORACLE_RESULT_SHA256
    assert sha256(source).hexdigest() == ORACLE_SOURCE_SHA256
    assert sha256(lock_bytes).hexdigest() == ORACLE_LOCK_SHA256
    assert json.loads(lock_bytes)["params"]["buy"] == LOCKED_BUY_DEFAULTS
    effective_lock = json.dumps(LOCKED_BUY_DEFAULTS, sort_keys=True, separators=(",", ":")).encode()
    assert sha256(effective_lock).hexdigest() == EFFECTIVE_BUY_LOCK_SHA256
    assert executed["minimal_roi"] == {"0": 0.03}
    assert executed["stoploss"] == -0.04
    assert executed["total_trades"] == 2192


def test_machine_readable_lineage_matches_the_promoted_source_and_behavior() -> None:
    expected = {
        "SIEVE_STAGE": "sieve3",
        "SOURCE_STRATEGY": "sieve2_mtf_std_daily_macd_volume_breakout_long_1h",
        "SOURCE_RESULT_BATCH": (
            "20260612T071622_entry_sieve2_top_level_clear100_cores8_20260612_01_retry01"
        ),
        "RESEARCH_PATH": "sieve3_v2_integrated_mtf_structure_exit",
        "ENTRY_SOURCE_STAGE": "sieve2",
        "EXIT_HYPOTHESIS": (
            "Frozen MTF prior highs, causal structural highs, and measured-range targets "
            "interact with multi-source invalidation through complete layered "
            "long-management policies."
        ),
    }
    module_values = {
        "SIEVE_STAGE": SIEVE_STAGE,
        "SOURCE_STRATEGY": SOURCE_STRATEGY,
        "SOURCE_RESULT_BATCH": SOURCE_RESULT_BATCH,
        "RESEARCH_PATH": RESEARCH_PATH,
        "ENTRY_SOURCE_STAGE": ENTRY_SOURCE_STAGE,
        "EXIT_HYPOTHESIS": EXIT_HYPOTHESIS,
    }
    assert module_values == expected
    strategy = _strategy()
    assert {name: getattr(strategy, name) for name in expected} == expected


def test_standalone_contract_and_locked_entry_surface() -> None:
    strategy = _strategy()
    assert strategy.__class__.__bases__ == (IStrategy,)
    source = STRATEGY_PATH.read_text(encoding="utf-8")
    assert "user_data.strategies" not in source
    assert "_sieve3_exit_rework_core" not in source
    assert "tlv2" not in source.lower()

    assert strategy.can_short is False
    assert strategy.use_exit_signal is True
    assert strategy.use_custom_stoploss is True
    assert strategy.use_custom_roi is True
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
    assert {name for name, value in parameters.items() if value.space == "buy"} == set(
        LOCKED_BUY_DEFAULTS
    )
    active = {name for name, value in parameters.items() if value.optimize}
    assert active == {"exit_policy_plan"}
    assert strategy.exit_policy_plan.space == "sell"


def test_freqtrade_strategy_resolver_loads_v2_module() -> None:
    with TemporaryDirectory() as temp_dir:
        copy2(STRATEGY_PATH, Path(temp_dir) / STRATEGY_PATH.name)
        config = {
            "strategy": "Sieve3V2IntegratedFromMtfStdDailyMacdVolumeBreakoutLong1H",
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


def test_exit_plans_are_compact_unique_whole_policies() -> None:
    assert len(EXIT_PLANS) == 32
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
    integrated_roles = {
        "target_partial",
        "invalidation_reduce",
        "target_tighten",
        "dual_target",
        "progress_failure",
    }
    integrated = [plan for plan in EXIT_PLANS.values() if plan.role in integrated_roles]
    assert len(integrated) == 20
    assert len(integrated) >= len(EXIT_PLANS) / 2

    target_families = {
        target
        for plan in EXIT_PLANS.values()
        for target in (plan.target_1, plan.target_2)
        if target is not None
    }
    assert target_families == set(TARGET_PROVIDER_CHAINS)
    raw_families = {
        "d1_high",
        "h4_high",
        "prior48",
        "prior96",
        "range_half",
        "range_full",
    }
    assert set(TARGET_PROVIDER_CHAINS["nearest"]) == raw_families
    for family in raw_families:
        assert TARGET_PROVIDER_CHAINS[family] == (family,)
    assert all(
        provider in raw_families for chain in TARGET_PROVIDER_CHAINS.values() for provider in chain
    )
    pure_targets = [plan for plan in EXIT_PLANS.values() if plan.role == "target_full"]
    assert pure_targets
    assert all(
        plan.invalidation == "none" and plan.invalidation_action == "hold" for plan in pure_targets
    )
    assert EXIT_PLANS["baseline_3_3"].fixed_tp == 0.03
    assert EXIT_PLANS["baseline_3_3"].hard_stop == 0.03
    assert EXIT_PLANS["promotion_oracle_3_4"].fixed_tp == 0.03
    assert EXIT_PLANS["promotion_oracle_3_4"].hard_stop == 0.04


def test_plan_names_match_executable_signatures_and_invalidation_is_multi_source() -> None:
    multi_source_modes = {"source_ltf", "ltf_h4", "h4_d1", "any_two"}
    for name, plan in EXIT_PLANS.items():
        if plan.invalidation != "none":
            assert plan.invalidation in multi_source_modes, name
            assert plan.invalidation in name, name
        if plan.partial_fraction == 0.33:
            assert "p33" in name or "reduce33" in name or "runner" in name, name
        if plan.partial_fraction == 0.50:
            assert "p50" in name or "reduce50" in name or "runner" in name, name
        if plan.remainder == "breakeven":
            assert "be" in name, name
        if plan.remainder == "trail" and plan.role != "dual_target":
            assert "trail" in name, name

    pure_invalidations = {
        name: plan for name, plan in EXIT_PLANS.items() if plan.role == "invalidation_full"
    }
    assert set(pure_invalidations) == {
        "source_ltf_full",
        "ltf_h4_full",
        "h4_d1_full",
        "any_two_full",
    }


def test_every_declared_plan_action_is_reachable() -> None:
    strategy_class = Sieve3V2IntegratedFromMtfStdDailyMacdVolumeBreakoutLong1H
    for name, plan in EXIT_PLANS.items():
        empty_state = {"partial_pending": False, "partial_filled": False}
        if plan.role == "baseline":
            assert plan.fixed_tp > 0.0, name
            continue
        if plan.target_action != "hold":
            decision = strategy_class.evaluate_policy(plan, empty_state, {"target_1": True})
            assert decision.action == plan.target_action, name
        if plan.invalidation != "none" and plan.invalidation_action != "hold":
            decision = strategy_class.evaluate_policy(plan, empty_state, {"invalidation": True})
            assert decision.action == plan.invalidation_action, name
        if plan.target_2:
            decision = strategy_class.evaluate_policy(
                plan,
                {"partial_pending": False, "partial_filled": True},
                {"target_2": True},
            )
            assert decision.action == "full", name
        if plan.progress_candles:
            decision = strategy_class.evaluate_policy(plan, empty_state, {"time_failure": True})
            assert decision.tag == "s3v2_progress_failure", name


def test_pure_evaluator_priority_and_actions() -> None:
    partial_plan = EXIT_PLANS["prior48_touch_p33_be_any_two"]
    pending = {"partial_pending": True, "partial_filled": False}
    legacy_deferred = {
        "partial_pending": False,
        "partial_filled": False,
        "terminal_exit_pending": True,
    }
    assert Sieve3V2IntegratedFromMtfStdDailyMacdVolumeBreakoutLong1H.evaluate_policy(
        partial_plan,
        legacy_deferred,
        {},
    ) == ExitDecision("full", "s3v2_hard_invalidation")
    assert Sieve3V2IntegratedFromMtfStdDailyMacdVolumeBreakoutLong1H.evaluate_policy(
        partial_plan,
        pending,
        {"hard_invalidation": True, "target_1": True},
    ) == ExitDecision("full", "s3v2_hard_invalidation")
    assert Sieve3V2IntegratedFromMtfStdDailyMacdVolumeBreakoutLong1H.evaluate_policy(
        partial_plan,
        pending,
        {"invalidation": True, "target_1": True},
    ) == ExitDecision("full", "s3v2_plan_invalidation")
    assert Sieve3V2IntegratedFromMtfStdDailyMacdVolumeBreakoutLong1H.evaluate_policy(
        partial_plan,
        pending,
        {"target_1": True},
    ) == ExitDecision(
        "partial",
        "s3v2_partial_target",
        0.33,
    )

    partial = Sieve3V2IntegratedFromMtfStdDailyMacdVolumeBreakoutLong1H.evaluate_policy(
        partial_plan,
        {"partial_pending": False, "partial_filled": False},
        {"target_1": True},
    )
    assert partial == ExitDecision("partial", "s3v2_partial_target", 0.33)

    tighten = Sieve3V2IntegratedFromMtfStdDailyMacdVolumeBreakoutLong1H.evaluate_policy(
        EXIT_PLANS["d1_touch_tighten_any_two"],
        {"partial_pending": False, "partial_filled": False},
        {"target_1": True},
    )
    assert tighten == ExitDecision("tighten", tighten="target")
    assert (
        Sieve3V2IntegratedFromMtfStdDailyMacdVolumeBreakoutLong1H.evaluate_policy(
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
            "prior_high_1h": [99.0, 500.0],
            "prior_high_4h": [108.0, 610.0],
            "prior_high_1d": [112.0, 620.0],
            "s3v2_prior_high_48": [106.0, 580.0],
            "s3v2_prior_high_96": [110.0, 570.0],
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
    assert state["targets"] == {
        "d1_high": 112.0,
        "h4_high": 108.0,
        "prior48": 106.0,
        "prior96": 110.0,
        "range_half": 101.0,
        "range_full": 103.0,
        "nearest": 101.0,
    }
    assert state["target_providers"] == {
        "d1_high": "d1_high",
        "h4_high": "h4_high",
        "prior48": "prior48",
        "prior96": "prior96",
        "range_half": "range_half",
        "range_full": "range_full",
        "nearest": "range_half",
    }

    changed_order = FakeOrder("buy", safe_price=200.0)
    strategy.order_filled(trade.pair, trade, changed_order, delayed_fill)
    assert trade.get_custom_data(STATE_KEY) == state

    runner = _strategy()
    runner.exit_policy_plan.value = "prior48_then_d1_runner_any_two"
    runner.dp = FakeDataProvider(frame)
    runner_trade = FakeTrade()
    runner.order_filled(runner_trade.pair, runner_trade, order, delayed_fill)
    runner_state = runner_trade.get_custom_data(STATE_KEY)
    assert runner_state["targets"]["prior48"] == 106.0
    assert runner_state["targets"]["d1_high"] == 112.0


def test_real_frozen_targets_drive_every_retained_provider_family() -> None:
    strategy = _strategy()
    row = pd.Series(
        {
            "prior_high_1h": 99.0,
            "prior_high_4h": 108.0,
            "prior_high_1d": 112.0,
            "s3v2_prior_high_48": 106.0,
            "s3v2_prior_high_96": 110.0,
            "s3v2_prior_low_48": 95.0,
        }
    )
    targets, providers, broken, support = strategy._frozen_targets(row, 100.0)
    cases = {
        "d1_high_touch_full": ("d1_high", "full"),
        "h4_high_reversal_full": ("h4_high", "full"),
        "prior48_touch_full": ("prior48", "full"),
        "prior96_reversal_full": ("prior96", "full"),
        "range_half_touch_p33_be_source_ltf": ("range_half", "partial"),
        "range_full_reversal_full": ("range_full", "full"),
        "nearest_reversal_p50_trail_any_two": ("range_half", "partial"),
    }

    for plan_name, (expected_provider, expected_action) in cases.items():
        plan = EXIT_PLANS[plan_name]
        target = targets[plan.target_1]
        assert target is not None
        assert providers[plan.target_1] == expected_provider
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01T01:00:00Z"], utc=True),
                "open": [target + 0.5],
                "high": [target + 1.0],
                "low": [target - 1.0],
                "close": [target - 0.5],
            }
        )
        state = {
            "version": STATE_VERSION,
            "plan": plan_name,
            "phase": "ENTRY",
            "entry_rate": 100.0,
            "entry_filled_at": "2024-01-01T01:00:00+00:00",
            "broken_resistance": broken,
            "structural_support": support,
            "targets": dict(targets),
            "target_providers": dict(providers),
            "target_1_touched_at": None,
            "target_2_touched_at": None,
            "partial_filled": False,
            "terminal_exit_pending": False,
            "terminal_exit_tag": None,
        }
        events, updated = strategy._events(
            FakeTrade(),
            state,
            frame,
            datetime(2024, 1, 1, 2, tzinfo=UTC),
            float(target),
            0.05,
        )
        assert events["target_1"] is True, plan_name
        assert updated["target_1_touched_at"] is not None, plan_name
        decision = strategy.evaluate_policy(plan, updated, events)
        assert decision.action == expected_action, plan_name


def test_partial_state_changes_only_after_exact_positive_fill() -> None:
    strategy = _strategy()
    trade = FakeTrade()
    trade.set_custom_data(
        STATE_KEY,
        {
            "version": STATE_VERSION,
            "plan": "prior48_touch_p33_be_any_two",
            "phase": "TARGET_ZONE",
            "partial_filled": False,
            "partial_target_stake": 330.0,
            "partial_realized_stake": 0.0,
            "partial_fill_order_ids": [],
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

    filled_at = datetime(2024, 1, 2, 3, tzinfo=UTC)
    strategy.order_filled(
        trade.pair,
        trade,
        FakeOrder(
            "sell",
            "s3v2_partial_target",
            safe_filled=3.3,
            order_filled_utc=filled_at,
            safe_amount=3.3,
            order_id="partial-complete",
        ),
        datetime(2024, 1, 2, 5, tzinfo=UTC),
    )
    state = trade.get_custom_data(STATE_KEY)
    assert state["partial_filled"] is True
    assert state["phase"] == "REMAINDER"
    assert state["partial_tag"] in PARTIAL_TAGS
    assert state["partial_filled_at"] == filled_at.isoformat()

    strategy.order_filled(
        trade.pair,
        trade,
        FakeOrder(
            "sell",
            "s3v2_partial_target",
            safe_filled=3.3,
            order_filled_utc=datetime(2024, 1, 2, 6, tzinfo=UTC),
            safe_amount=3.3,
            order_id="partial-complete",
        ),
        datetime(2024, 1, 2, 7, tzinfo=UTC),
    )
    assert trade.get_custom_data(STATE_KEY)["partial_filled_at"] == filled_at.isoformat()


def test_partial_fill_reconciles_quantity_before_remainder_transition() -> None:
    strategy = _strategy()
    trade = FakeTrade()
    trade.set_custom_data(
        STATE_KEY,
        {
            "version": STATE_VERSION,
            "plan": "prior48_touch_p33_be_any_two",
            "phase": "TARGET_ZONE",
            "partial_filled": False,
            "partial_target_stake": 330.0,
            "partial_realized_stake": 0.0,
            "partial_fill_order_ids": [],
        },
    )
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
    state = trade.get_custom_data(STATE_KEY)
    assert state["partial_filled"] is False
    assert state["partial_realized_stake"] == 165.0

    trade.stake_amount = 835.0
    strategy._decision_context = lambda *args, **kwargs: (  # type: ignore[method-assign]
        EXIT_PLANS["prior48_touch_p33_be_any_two"],
        trade.get_custom_data(STATE_KEY),
        {},
        ExitDecision("partial", "s3v2_partial_target", 0.33),
    )
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

    strategy.order_filled(
        trade.pair,
        trade,
        FakeOrder(
            "sell",
            "s3v2_partial_target",
            safe_filled=1.65,
            safe_amount=1.65,
            safe_price=100.0,
            order_id="partial-half-2",
        ),
        datetime(2024, 1, 2, 4, tzinfo=UTC),
    )
    assert trade.get_custom_data(STATE_KEY)["partial_filled"] is True


def test_open_order_blocks_duplicate_partial_request() -> None:
    strategy = _strategy()
    trade = FakeTrade()
    trade.orders.append(
        FakeOrder(
            "sell",
            "s3v2_partial_target",
            status="open",
            ft_is_open=True,
        )
    )

    stoploss_only = FakeTrade()
    stoploss_only.orders.append(FakeOrder("stoploss", status="open", ft_is_open=True))
    assert strategy._has_open_order(stoploss_only) is False
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


def test_hard_invalidation_waits_for_partial_then_remains_sticky() -> None:
    strategy = _strategy()
    trade = FakeTrade()
    trade.set_custom_data(
        STATE_KEY,
        {
            "version": STATE_VERSION,
            "plan": "prior48_touch_p33_be_any_two",
            "phase": "TARGET_ZONE",
            "entry_rate": 100.0,
            "entry_candle": "2024-01-01T00:00:00+00:00",
            "entry_filled_at": "2024-01-01T01:00:00+00:00",
            "broken_resistance": 99.0,
            "structural_support": 95.0,
            "targets": {"prior48": 110.0},
            "target_1_touched_at": None,
            "target_2_touched_at": None,
            "partial_filled": False,
            "partial_target_stake": 330.0,
            "partial_realized_stake": 0.0,
            "partial_fill_order_ids": [],
            "terminal_exit_pending": False,
            "terminal_exit_tag": None,
            "stop_floor": 95.0,
        },
    )
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
                "date": pd.to_datetime(["2024-01-01T01:00:00Z", "2024-01-01T02:00:00Z"], utc=True),
                "open": [96.0, 95.0],
                "high": [97.0, 96.0],
                "low": [89.0, 88.0],
                "close": [90.0, 89.0],
                "ema_fast_1d": [90.0, 90.0],
                "ema_slow_1d": [100.0, 100.0],
                "macd_hist_1d": [-1.0, -1.0],
            }
        )
    )
    assert (
        strategy.custom_exit(
            trade.pair,
            trade,
            datetime(2024, 1, 1, 3, tzinfo=UTC),
            90.0,
            -0.10,
        )
        is None
    )
    deferred = trade.get_custom_data(STATE_KEY)
    assert deferred["terminal_exit_pending"] is True
    assert deferred["terminal_exit_tag"] == "s3v2_hard_invalidation"

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
        == "s3v2_hard_invalidation"
    )


def test_transient_non_hard_invalidation_keeps_exact_exit_while_partial_pending() -> None:
    strategy = _strategy()
    trade = FakeTrade()
    trade.set_custom_data(
        STATE_KEY,
        {
            "version": STATE_VERSION,
            "plan": "prior48_touch_p33_be_any_two",
            "phase": "TARGET_ZONE",
            "entry_rate": 100.0,
            "entry_candle": "2024-01-01T00:00:00+00:00",
            "entry_filled_at": "2024-01-01T01:00:00+00:00",
            "broken_resistance": 90.0,
            "structural_support": 85.0,
            "targets": {"prior48": 120.0},
            "target_1_touched_at": None,
            "target_2_touched_at": None,
            "partial_filled": False,
            "partial_target_stake": 330.0,
            "partial_realized_stake": 0.0,
            "partial_fill_order_ids": [],
            "terminal_exit_pending": False,
            "terminal_exit_tag": None,
            "stop_floor": 95.0,
        },
    )
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
                "date": pd.to_datetime(["2024-01-01T01:00:00Z", "2024-01-01T02:00:00Z"], utc=True),
                "open": [100.0, 100.0],
                "high": [101.0, 101.0],
                "low": [99.0, 99.0],
                "close": [100.0, 100.0],
                "ema_fast_1h": [110.0, 110.0],
                "macd_hist_1h": [-1.0, -1.0],
                "volume_ratio_1h": [0.5, 0.5],
                "ema_fast_4h": [90.0, 90.0],
                "ema_slow_4h": [100.0, 100.0],
                "macd_hist_4h": [-1.0, -1.0],
                "ema_fast_1d": [110.0, 110.0],
                "ema_slow_1d": [100.0, 100.0],
                "macd_hist_1d": [1.0, 1.0],
            }
        )
    )
    context = strategy._decision_context(
        trade.pair,
        trade,
        datetime(2024, 1, 1, 3, tzinfo=UTC),
        100.0,
        0.0,
    )
    assert context is not None
    _, pending_state, events, decision = context
    assert events["invalidation"] is True
    assert events["hard_invalidation"] is False
    assert decision == ExitDecision("full", "s3v2_plan_invalidation")
    assert pending_state["partial_pending"] is True
    persisted = trade.get_custom_data(STATE_KEY)
    assert persisted["terminal_exit_pending"] is True
    assert persisted["terminal_exit_tag"] == "s3v2_plan_invalidation"
    assert (
        strategy.custom_exit(
            trade.pair,
            trade,
            datetime(2024, 1, 1, 3, tzinfo=UTC),
            100.0,
            0.0,
        )
        is None
    )

    trade.orders.clear()
    strategy.dp = FakeDataProvider(
        pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01T03:00:00Z"], utc=True),
                "open": [100.0],
                "high": [101.0],
                "low": [99.0],
                "close": [100.0],
                "ema_fast_1h": [90.0],
                "macd_hist_1h": [1.0],
                "volume_ratio_1h": [2.0],
                "ema_fast_4h": [110.0],
                "ema_slow_4h": [100.0],
                "macd_hist_4h": [1.0],
                "ema_fast_1d": [110.0],
                "ema_slow_1d": [100.0],
                "macd_hist_1d": [1.0],
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


def test_mtf_full_invalidation_requires_coherent_multi_source_failure() -> None:
    strategy = _strategy()
    trade = FakeTrade()
    state = {
        "version": STATE_VERSION,
        "plan": "any_two_full",
        "phase": "ENTRY",
        "entry_rate": 100.0,
        "entry_candle": "2024-01-01T00:00:00+00:00",
        "entry_filled_at": "2024-01-01T01:00:00+00:00",
        "broken_resistance": 99.0,
        "structural_support": 95.0,
        "targets": {},
        "target_1_touched_at": None,
        "target_2_touched_at": None,
        "partial_filled": False,
    }
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01T01:00:00Z", "2024-01-01T02:00:00Z"], utc=True),
            "open": [97.0, 96.0],
            "high": [98.0, 97.0],
            "low": [94.0, 93.0],
            "close": [95.0, 94.0],
            "ema_fast_1h": [90.0, 90.0],
            "macd_hist_1h": [1.0, 1.0],
            "volume_ratio_1h": [2.0, 2.0],
            "ema_fast_4h": [110.0, 110.0],
            "ema_slow_4h": [100.0, 100.0],
            "macd_hist_4h": [1.0, 1.0],
            "ema_fast_1d": [110.0, 110.0],
            "ema_slow_1d": [100.0, 100.0],
            "macd_hist_1d": [1.0, 1.0],
        }
    )

    source_only, _ = strategy._events(
        trade, dict(state), frame, datetime(2024, 1, 1, 3, tzinfo=UTC), 94.0, -0.06
    )
    assert source_only["source_failure"] is True
    assert source_only["failure_count"] == 1
    assert source_only["invalidation"] is False
    assert source_only["hard_invalidation"] is False

    source_and_daily = frame.copy()
    source_and_daily[["ema_fast_1d", "macd_hist_1d"]] = -1.0
    paired, _ = strategy._events(
        trade,
        dict(state),
        source_and_daily,
        datetime(2024, 1, 1, 3, tzinfo=UTC),
        94.0,
        -0.06,
    )
    assert paired["failure_count"] == 2
    assert paired["invalidation"] is True
    assert paired["hard_invalidation"] is True

    one_bar_flip = source_and_daily.iloc[-1:].copy()
    one_bar_flip["close"] = 100.0
    one_bar_flip[["ema_fast_1h", "macd_hist_1h", "volume_ratio_1h"]] = [
        110.0,
        -1.0,
        0.5,
    ]
    one_bar_flip[["ema_fast_1d", "ema_slow_1d", "macd_hist_1d"]] = [
        110.0,
        100.0,
        1.0,
    ]
    single, _ = strategy._events(
        trade,
        dict(state),
        one_bar_flip,
        datetime(2024, 1, 1, 3, tzinfo=UTC),
        100.0,
        0.0,
    )
    assert single["failure_count"] == 0
    assert single["invalidation"] is False


def test_target_confirmation_cannot_reuse_signal_candle_touch() -> None:
    strategy = _strategy()
    trade = FakeTrade()
    state = {
        "version": STATE_VERSION,
        "plan": "prior48_touch_full",
        "phase": "ENTRY",
        "entry_rate": 100.0,
        "entry_candle": "2024-01-01T00:00:00+00:00",
        "entry_filled_at": "2024-01-01T01:00:00+00:00",
        "broken_resistance": 99.0,
        "structural_support": 95.0,
        "targets": {"prior48": 110.0},
        "target_1_touched_at": None,
        "target_2_touched_at": None,
        "partial_filled": False,
    }
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01T00:00:00Z", "2024-01-01T01:00:00Z"], utc=True),
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


def test_delayed_fill_excludes_prefill_target_and_progress_candles() -> None:
    strategy = _strategy()
    trade = FakeTrade()
    trade.open_date_utc = datetime(2024, 1, 1, 1, tzinfo=UTC)
    state = {
        "version": STATE_VERSION,
        "plan": "progress48_nearest_any_two",
        "phase": "ENTRY",
        "entry_rate": 100.0,
        "entry_candle": "2024-01-01T00:00:00+00:00",
        "entry_filled_at": "2024-01-01T05:00:00+00:00",
        "broken_resistance": 99.0,
        "structural_support": 95.0,
        "targets": {"nearest": 110.0},
        "target_1_touched_at": None,
        "target_2_touched_at": None,
        "partial_filled": False,
    }
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01T01:00:00Z", periods=8, freq="1h"),
            "open": 100.0,
            "high": [120.0, 120.0, 120.0, 120.0, 105.0, 105.0, 105.0, 105.0],
            "low": 99.0,
            "close": 100.0,
        }
    )
    events, updated = strategy._events(
        trade,
        state,
        frame,
        datetime(2024, 1, 1, 9, tzinfo=UTC),
        100.0,
        0.0,
    )
    assert events["target_1"] is False
    assert events["time_failure"] is False
    assert updated["target_1_touched_at"] is None


def test_target_band_requires_positive_progress_and_reversal_after_touch() -> None:
    strategy = _strategy()
    dates = pd.date_range("2024-01-01", periods=6, freq="1h", tz="UTC")
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0, 100.0, 100.0, 100.1, 100.2, 100.1],
            "high": [100.1, 100.1, 100.1, 100.25, 100.3, 100.2],
            "close": [99.9, 99.9, 99.9, 100.2, 100.1, 100.0],
        }
    )

    untouched_at, confirmed = strategy._confirmation(
        frame.iloc[:3], 100.2, "touch", None, 0.03, 100.0
    )
    assert untouched_at is None
    assert confirmed is False

    touched_at, confirmed = strategy._confirmation(
        frame.iloc[:5], 100.2, "reversal2of3", None, 0.03, 100.0
    )
    assert touched_at == dates[3].isoformat()
    assert confirmed is False

    same_touch_at, confirmed = strategy._confirmation(
        frame, 100.2, "reversal2of3", touched_at, 0.03, 100.0
    )
    assert same_touch_at == touched_at
    assert confirmed is True


def test_all_plans_have_native_roi_fallback_and_bounded_hold() -> None:
    strategy = _strategy()
    trade = FakeTrade()
    for plan_name, expected in (("baseline_3_3", 0.03), ("promotion_oracle_3_4", 0.03)):
        trade.set_custom_data(
            STATE_KEY,
            {"version": STATE_VERSION, "plan": plan_name},
        )
        assert (
            strategy.custom_roi(trade.pair, trade, datetime.now(UTC), 10, None, "long") == expected
        )

    trade.set_custom_data(
        STATE_KEY,
        {"version": STATE_VERSION, "plan": "prior48_touch_full"},
    )
    assert strategy.custom_roi(trade.pair, trade, datetime.now(UTC), 10, None, "long") == 0.12
    for name, plan in EXIT_PLANS.items():
        if plan.role == "baseline":
            assert plan.max_hold_candles == 0, name
        else:
            assert plan.fixed_tp == 0.12, name
            assert plan.max_hold_candles == 720, name
    assert Sieve3V2IntegratedFromMtfStdDailyMacdVolumeBreakoutLong1H.evaluate_policy(
        EXIT_PLANS["prior48_touch_full"],
        {"partial_pending": False, "partial_filled": False},
        {"max_hold": True},
    ) == ExitDecision("full", "s3v2_max_hold")


def test_progress_uses_closed_candle_high_not_mutable_trade_max_rate() -> None:
    strategy = _strategy()
    trade = FakeTrade()
    trade.max_rate = 1000.0
    state = {
        "version": STATE_VERSION,
        "plan": "progress48_nearest_any_two",
        "phase": "ENTRY",
        "entry_rate": 100.0,
        "entry_candle": "2024-01-01T00:00:00+00:00",
        "entry_filled_at": "2024-01-01T01:00:00+00:00",
        "broken_resistance": 99.0,
        "structural_support": 95.0,
        "targets": {"nearest": 110.0},
        "target_1_touched_at": None,
        "target_2_touched_at": None,
        "partial_filled": False,
    }
    frame = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01T01:00:00Z", periods=50, freq="1h"),
            "open": 100.0,
            "high": 100.5,
            "low": 99.5,
            "close": 100.0,
        }
    )
    events, _ = strategy._events(
        trade,
        state,
        frame,
        datetime(2024, 1, 3, 3, tzinfo=UTC),
        100.0,
        0.0,
    )
    assert events["time_failure"] is True


def test_mixed_partial_actions_use_profit_state() -> None:
    target_plan = EXIT_PLANS["prior48_touch_p33_be_any_two"]
    decision = Sieve3V2IntegratedFromMtfStdDailyMacdVolumeBreakoutLong1H.evaluate_policy(
        target_plan,
        {"partial_pending": False, "partial_filled": False},
        {"target_1": True, "profit_bucket": "loss"},
    )
    assert decision == ExitDecision("full", "s3v2_target_reversal_loss")

    reduce_plan = EXIT_PLANS["source_ltf_reduce33_prior48"]
    decision = Sieve3V2IntegratedFromMtfStdDailyMacdVolumeBreakoutLong1H.evaluate_policy(
        reduce_plan,
        {"partial_pending": False, "partial_filled": False},
        {"invalidation": True, "profit_bucket": "loss"},
    )
    assert decision == ExitDecision("full", "s3v2_invalidation_loss")


def test_custom_stoploss_persists_monotonic_absolute_floor() -> None:
    strategy = _strategy()
    trade = FakeTrade()
    trade.set_custom_data(
        STATE_KEY,
        {
            "version": STATE_VERSION,
            "plan": "prior48_reversal_p50_trail_any_two",
            "phase": "REMAINDER",
            "entry_rate": 100.0,
            "entry_candle": "2024-01-01T00:00:00+00:00",
            "entry_filled_at": "2024-01-01T01:00:00+00:00",
            "broken_resistance": 99.0,
            "structural_support": 95.0,
            "targets": {"prior48": 110.0},
            "target_1_touched_at": "2024-01-01T10:00:00+00:00",
            "target_2_touched_at": None,
            "partial_filled": True,
            "partial_filled_at": "2024-01-01T10:00:00+00:00",
            "stop_floor": None,
        },
    )
    strategy.custom_stoploss(trade.pair, trade, datetime.now(UTC), 120.0, 0.20, False)
    first_floor = trade.get_custom_data(STATE_KEY)["stop_floor"]
    assert first_floor == 120.0 * 0.985

    strategy.custom_stoploss(trade.pair, trade, datetime.now(UTC), 110.0, 0.10, True)
    assert trade.get_custom_data(STATE_KEY)["stop_floor"] == first_floor


def test_tighten_floor_breach_becomes_explicit_exit() -> None:
    strategy = _strategy()
    trade = FakeTrade()
    state = {
        "version": STATE_VERSION,
        "plan": "prior48_touch_tighten_source_ltf",
        "phase": "ENTRY",
        "entry_rate": 100.0,
        "entry_candle": "2024-01-01T00:00:00+00:00",
        "entry_filled_at": "2024-01-01T01:00:00+00:00",
        "broken_resistance": 99.0,
        "structural_support": 95.0,
        "targets": {"prior48": 110.0},
        "target_1_touched_at": None,
        "target_2_touched_at": None,
        "partial_filled": False,
        "terminal_exit_pending": False,
        "stop_floor": None,
    }
    trade.set_custom_data(STATE_KEY, state)
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


def test_freqtrade_applies_and_keeps_monotonic_stop_floor() -> None:
    strategy = _strategy()
    with FtNoDBContext("1h"):
        trade = Trade(
            id=930001,
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
        trade.set_custom_data(
            STATE_KEY,
            {
                "version": STATE_VERSION,
                "plan": "prior48_reversal_p50_trail_any_two",
                "phase": "REMAINDER",
                "entry_rate": 100.0,
                "entry_candle": "2024-01-01T00:00:00+00:00",
                "entry_filled_at": "2024-01-01T01:00:00+00:00",
                "broken_resistance": 99.0,
                "structural_support": 95.0,
                "targets": {"prior48": 110.0},
                "target_1_touched_at": "2024-01-01T10:00:00+00:00",
                "target_2_touched_at": None,
                "partial_filled": True,
                "partial_filled_at": "2024-01-01T10:00:00+00:00",
                "terminal_exit_pending": False,
                "stop_floor": None,
            },
        )
        strategy.ft_stoploss_adjust(
            120.0,
            trade,
            datetime(2024, 1, 2, tzinfo=UTC),
            0.20,
            0,
        )
        assert trade.stop_loss == 118.2


def test_candle_high_trail_cannot_create_same_candle_custom_exit_at_open() -> None:
    strategy = _strategy()
    with FtNoDBContext("1h"):
        trade = Trade(
            id=930002,
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
        trade.set_custom_data(
            STATE_KEY,
            {
                "version": STATE_VERSION,
                "plan": "prior48_reversal_p50_trail_any_two",
                "phase": "REMAINDER",
                "entry_rate": 100.0,
                "entry_candle": "2024-01-01T00:00:00+00:00",
                "entry_filled_at": "2024-01-01T01:00:00+00:00",
                "broken_resistance": 99.0,
                "structural_support": 95.0,
                "targets": {"prior48": 130.0},
                "target_1_touched_at": "2024-01-01T10:00:00+00:00",
                "target_2_touched_at": None,
                "partial_filled": True,
                "partial_filled_at": "2024-01-01T10:00:00+00:00",
                "terminal_exit_pending": False,
                "stop_floor": None,
            },
        )
        exits = strategy.should_exit(
            trade,
            110.0,
            datetime(2024, 1, 2, tzinfo=UTC),
            enter=False,
            exit_=False,
            low=109.0,
            high=120.0,
        )
        assert all(exit_check.exit_type != ExitType.CUSTOM_EXIT for exit_check in exits)
        assert any(exit_check.exit_type == ExitType.TRAILING_STOP_LOSS for exit_check in exits)
        assert trade.get_custom_data(STATE_KEY)["stop_floor"] == 118.2

        strategy.ft_stoploss_adjust(
            110.0,
            trade,
            datetime(2024, 1, 2, 1, tzinfo=UTC),
            0.10,
            0,
            after_fill=True,
        )
        assert trade.stop_loss == 118.2


def test_entry_driving_indicators_are_prefix_invariant() -> None:
    frame = pd.read_feather(DATA_PATH).iloc[12000:13600].reset_index(drop=True)
    prefix = frame.iloc[:1200].copy()
    short_strategy = _strategy()
    short_strategy.dp = _historical_provider(prefix)
    full_strategy = _strategy()
    full_strategy.dp = _historical_provider(frame)
    metadata = {"pair": "BTC/USDT:USDT"}

    short_frame = short_strategy.populate_indicators(prefix.copy(), metadata)
    full_frame = full_strategy.populate_indicators(frame.copy(), metadata).iloc[:1200]
    for column in (
        "ema_fast_1h",
        "ema_slow_1h",
        "volume_ratio_1h",
        "prior_high_1h",
        "ema_fast_4h",
        "ema_slow_4h",
        "macd_hist_1d",
        "prior_high_1d",
        "s3v2_prior_high_48",
        "s3v2_prior_high_96",
    ):
        np.testing.assert_allclose(
            pd.to_numeric(short_frame[column], errors="coerce"),
            pd.to_numeric(full_frame[column], errors="coerce"),
            equal_nan=True,
            err_msg=column,
        )

    short_entries = short_strategy.populate_entry_trend(short_frame.copy(), metadata)
    full_entries = full_strategy.populate_entry_trend(full_frame.copy(), metadata)
    pd.testing.assert_series_equal(
        short_entries["enter_long"].fillna(0).astype(int),
        full_entries["enter_long"].fillna(0).astype(int),
    )
    pd.testing.assert_series_equal(
        short_entries["enter_tag"].fillna(""),
        full_entries["enter_tag"].fillna(""),
    )


def test_neutral_entry_parity_against_authoritative_executed_source() -> None:
    frame = pd.read_feather(DATA_PATH)
    frame = frame.loc[
        frame["date"].ge(pd.Timestamp("2020-01-01", tz="UTC"))
        & frame["date"].lt(pd.Timestamp("2020-10-01", tz="UTC"))
    ].reset_index(drop=True)
    metadata = {"pair": "BTC/USDT:USDT"}

    control = _oracle_strategy()
    control.dp = _historical_provider(frame)
    candidate = _strategy()
    candidate.dp = _historical_provider(frame)
    control_frame = control.populate_entry_trend(
        control.populate_indicators(frame.copy(), metadata), metadata
    )
    candidate_frame = candidate.populate_entry_trend(
        candidate.populate_indicators(frame.copy(), metadata), metadata
    )

    assert int(control_frame["enter_long"].fillna(0).sum()) > 0
    pd.testing.assert_series_equal(
        control_frame["enter_long"].fillna(0).astype(int),
        candidate_frame["enter_long"].fillna(0).astype(int),
    )
    pd.testing.assert_series_equal(
        control_frame["enter_tag"].fillna(""),
        candidate_frame["enter_tag"].fillna(""),
    )
