"""Paper integration guards: exact Sieve plans, context freshness and signal arbitration."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from user_data.strategies.integrated_paper import IntegratedPaper
from user_data.strategies.integrated_paper_context import LunaContext, load_luna_context


def _config() -> dict:
    with open("user_data/configs/integrated_paper_20260926_base.json", encoding="utf-8") as handle:
        base = json.load(handle)
    with open("user_data/configs/integrated_paper_20260926_auto.json", encoding="utf-8") as handle:
        base.update(json.load(handle))
    return base


def test_selected_sieve_exit_plans_are_loaded_exactly() -> None:
    strategy = IntegratedPaper(_config())
    assert strategy.stoploss == -0.06
    assert strategy._sources["breakout_long"]._focused_plan()["target_1_ratio"] == pytest.approx(0.05)
    assert strategy._sources["breakout_long"]._focused_plan()["hard_stop_ratio"] == pytest.approx(0.06)
    assert strategy._sources["pullback_long"]._focused_plan()["target_1_ratio"] == pytest.approx(0.03)
    assert strategy._sources["resistance_short"]._focused_plan()["target_1_ratio"] == pytest.approx(0.05)
    assert strategy._sources["resistance_short"]._focused_plan()["hard_stop_ratio"] == pytest.approx(0.02)


def test_source_specific_initial_stops_dispatch_and_fit_emergency_backstop() -> None:
    strategy = IntegratedPaper(_config())
    strategy.dp = object()
    now = datetime(2026, 9, 27, tzinfo=timezone.utc)
    for source_id, side, expected in (
        ("breakout_long", "long", 0.06),
        ("pullback_long", "long", 0.04),
        ("resistance_short", "short", 0.02),
    ):
        trade = SimpleNamespace(
            enter_tag=f"sieve:{source_id}", open_rate=100.0,
            is_short=side == "short", has_open_orders=False,
            nr_of_successful_exits=0, leverage=1.0,
        )
        stop = strategy.custom_stoploss("BTC/USDT:USDT", trade, now, 100.0, 0.0, False)
        assert stop == pytest.approx(expected)
        assert stop <= abs(strategy.stoploss) + 1e-9


def test_signal_arbitration_retains_source_and_abstains_on_conflict() -> None:
    strategy = IntegratedPaper(_config())
    frame = pd.DataFrame({
        "candidate_breakout_long": [True, False, True],
        "candidate_pullback_long": [True, True, False],
        "candidate_resistance_short": [False, False, True],
    })
    selected = strategy.populate_entry_trend(frame, {"pair": "BTC/USDT:USDT"})
    assert selected.loc[0, "enter_tag"] == "sieve:breakout_long"
    assert selected.loc[1, "enter_tag"] == "sieve:pullback_long"
    assert pd.isna(selected.loc[2, "enter_tag"])
    assert pd.isna(selected.loc[2, "enter_long"])
    assert pd.isna(selected.loc[2, "enter_short"])


def test_luna_context_requires_sourced_fresh_direction(tmp_path) -> None:
    now = datetime(2026, 9, 26, 21, 0, tzinfo=timezone.utc)
    path = tmp_path / "luna.json"
    assert load_luna_context(now, path).status == "missing"
    row = {
        "schema_version": 1,
        "observed_at_utc": (now - timedelta(minutes=15)).isoformat(),
        "valid_until_utc": (now + timedelta(hours=3)).isoformat(),
        "risk_bias": "risk_off",
        "event_scale": "major",
        "attention": "elevated",
        "event_id": "verified_test_event",
        "sources": [],
    }
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="requires source URLs"):
        load_luna_context(now, path)
    row["sources"] = ["https://example.com/source"]
    path.write_text(json.dumps(row), encoding="utf-8")
    assert load_luna_context(now, path) == LunaContext(
        status="observed", observed_at=now - timedelta(minutes=15),
        risk_bias="risk_off", event_scale="major", attention="elevated",
        event_id="verified_test_event", sources=("https://example.com/source",),
    )
    assert load_luna_context(now + timedelta(hours=4), path).status == "stale"


def test_non_dry_run_is_refused() -> None:
    strategy = IntegratedPaper(_config())
    with pytest.raises(RuntimeError, match="dry-run only"):
        strategy.bot_start()


def test_all_three_source_indicators_and_level_surface_can_run_together() -> None:
    def candles(periods: int, frequency: str, base: float) -> pd.DataFrame:
        offsets = np.arange(periods, dtype=float)
        close = base + offsets * 0.08 + np.sin(offsets / 5.0)
        return pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=periods, freq=frequency, tz="UTC"),
            "open": close - 0.1, "high": close + 0.8,
            "low": close - 0.8, "close": close,
            "volume": 1000.0 + offsets % 25,
        })

    class FakeProvider:
        data = {"1h": candles(450, "1h", 100.0),
                "4h": candles(250, "4h", 100.0),
                "1d": candles(120, "1D", 100.0)}

        def get_pair_dataframe(self, pair: str, timeframe: str) -> pd.DataFrame:
            assert pair == "BTC/USDT:USDT"
            return self.data[timeframe].copy()

        def current_whitelist(self) -> list[str]:
            return ["BTC/USDT:USDT"]

    strategy = IntegratedPaper(_config())
    strategy.dp = FakeProvider()
    frame = strategy.populate_indicators(
        FakeProvider.data["1h"].copy(), {"pair": "BTC/USDT:USDT"},
    )
    assert {f"candidate_{name}" for name in strategy._sources} <= set(frame.columns)
    assert {"paper_atr", "level_reaction_long", "level_reaction_short"} <= set(frame.columns)
    selected = strategy.populate_entry_trend(frame, {"pair": "BTC/USDT:USDT"})
    assert len(selected) == 450


def test_entry_context_uses_completed_leaders_and_explicit_unknown_sources(monkeypatch) -> None:
    import user_data.strategies.integrated_paper as module

    now = datetime(2026, 9, 26, 21, 0, tzinfo=timezone.utc)
    base = {
        "date": now - timedelta(hours=1), "paper_atr": 2.0,
        "level_reaction_long": False, "level_reaction_short": False,
        "level_reaction_tag": "",
    }
    base.update({name: np.nan for name in PaperLevelColumns})
    analyzed = pd.DataFrame([base])
    dates = pd.date_range(now - timedelta(hours=20), periods=20, freq="1h", tz="UTC")
    leader = pd.DataFrame({
        "date": dates, "open": np.arange(20.0), "high": np.arange(20.0) + 2,
        "low": np.arange(20.0), "close": np.arange(20.0) + 1,
        "volume": np.full(20, 100.0),
    })

    class FakeProvider:
        def get_analyzed_dataframe(self, pair: str, timeframe: str):
            return analyzed, now

        def get_pair_dataframe(self, pair: str, timeframe: str):
            return leader.copy()

    strategy = IntegratedPaper(_config())
    strategy.dp = FakeProvider()
    monkeypatch.setattr(module, "_recent_pressure", lambda pair, now: (None, "source_missing"))
    monkeypatch.setattr(module, "load_luna_context", lambda now: LunaContext(status="missing"))
    decision = strategy._decision("BTC/USDT:USDT", "long", "sieve:breakout_long", now, 20.0)
    assert decision["decision"] == "enter"
    assert decision["observed"]["luna"]["status"] == "missing"
    assert decision["observed"]["orderbook"]["status"] == "source_missing"


def test_unrecorded_stake_decision_cannot_confirm_entry(monkeypatch) -> None:
    strategy = IntegratedPaper(_config())
    now = datetime(2026, 9, 26, 21, 0, tzinfo=timezone.utc)
    strategy.wallets = type("Wallet", (), {"get_total_stake_amount": lambda self: 10000.0})()
    monkeypatch.setattr(strategy, "_decision", lambda *args: {
        "decision": "enter", "size_factor": 1.0,
    })

    def fail_to_record(_decision):
        raise OSError("decision journal unavailable")

    monkeypatch.setattr(strategy, "_record", fail_to_record)
    with pytest.raises(OSError, match="journal unavailable"):
        strategy.custom_stake_amount(
            "BTC/USDT:USDT", now, 100.0, 10000.0, 5.0, 10000.0,
            1.0, "sieve:breakout_long", "long",
        )
    assert not strategy.confirm_trade_entry(
        "BTC/USDT:USDT", "market", 1.0, 100.0, "gtc", now,
        "sieve:breakout_long", "long",
    )


PaperLevelColumns = (
    "prior_24h_high", "prior_24h_low", "rolling_20_high_4h", "rolling_20_low_4h",
    "vp_poc_4h", "vp_hvn_above_4h", "vp_hvn_below_4h",
    "vp_lvn_above_4h", "vp_lvn_below_4h",
)
