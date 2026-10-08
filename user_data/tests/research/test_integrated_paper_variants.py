"""Focused checks for the ten parked paper variants and their isolation."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from freqtrade.enums import RunMode
from user_data.strategies.integrated_paper import IntegratedPaper
from user_data.strategies.integrated_paper_context import LunaContext
from user_data.strategies.integrated_paper_variants import VARIANT_SPECS
import user_data.strategies.integrated_paper_variants as variants


ROOT = Path(__file__).resolve().parents[3]


def _config(variant_id: str) -> dict:
    base = json.loads((ROOT / "user_data/configs/integrated_paper_20260926_base.json").read_text())
    overlay = json.loads((ROOT / f"user_data/configs/integrated_paper_variants/v{variant_id}.json").read_text())
    base.update(overlay)
    return base


def _strategy(variant_id: str):
    cls = getattr(variants, f"PaperVariant{variant_id}")
    return cls(_config(variant_id))


def test_ten_profiles_have_separate_paper_accounts_and_exact_sources() -> None:
    assert len(VARIANT_SPECS) == 10
    dbs = set()
    for variant_id in VARIANT_SPECS:
        config = _config(variant_id)
        strategy = _strategy(variant_id)
        assert config["dry_run"] is True
        assert config["exchange"]["key"] == config["exchange"]["secret"] == ""
        assert config["bot_name"] == f"integrated_paper_v{variant_id}"
        assert config["strategy"] == type(strategy).__name__
        assert strategy.stoploss == -0.06
        assert config["force_entry_enable"] is False
        assert config["initial_state"] == ("paused" if variant_id in {"01", "10"} else "stopped")
        assert config["db_url"] not in dbs
        dbs.add(config["db_url"])
        assert set(strategy.spec["sources"]) <= set(strategy._sources)
        with pytest.raises(RuntimeError, match="dry-run only"):
            strategy.bot_start()
        strategy.config["runmode"] = RunMode.DRY_RUN
        strategy.bot_start()


def test_variant_refuses_cross_account_database() -> None:
    strategy = _strategy("01")
    strategy.config["runmode"] = RunMode.DRY_RUN
    strategy.config["db_url"] = _config("10")["db_url"]
    with pytest.raises(RuntimeError, match="not isolated"):
        strategy.bot_start()


def _decision(variant_id: str, monkeypatch, *, low_volume=False, level_same=False,
              barrier=False, leaders=None, event=False, book=None,
              luna=None, side="long", tag="sieve:breakout_long"):
    now = datetime(2026, 9, 27, 0, 0, tzinfo=timezone.utc)
    strategy = _strategy(variant_id)
    row = {
        "date": now - timedelta(hours=1), "volume": 50.0 if low_volume else 150.0,
        "prior_20_volume_median": 100.0, "paper_atr": 4.0,
        "level_reaction_long": level_same if side == "long" else False,
        "level_reaction_short": level_same if side == "short" else False,
        "level_reaction_tag": f"level_single_{side}" if level_same else "",
    }
    row.update({name: ((100.5 if side == "long" else 99.5)
                       if barrier and name == "vp_poc_4h" else np.nan)
                for name in strategy._levels.level_columns})

    class Provider:
        def get_analyzed_dataframe(self, pair, timeframe):
            return pd.DataFrame([row]), now

    strategy.dp = Provider()
    monkeypatch.setattr(strategy, "_independent_leaders", lambda pair, now, side: (
        leaders if leaders is not None else {"ETH/USDT:USDT": "aligned"}
    ))
    monkeypatch.setattr(variants, "_recent_pressure", lambda pair, now: (
        book if book is not None else (None, "source_missing")
    ))
    monkeypatch.setattr(variants, "load_luna_context", lambda now: (
        luna if luna is not None else LunaContext(status="missing")
    ))
    strategy._events = ((now - timedelta(minutes=30), "scheduled_test"),) if event else ()
    return strategy._decision("BTC/USDT:USDT", side, tag, now, 100.0)


def test_low_mix_profiles_do_not_claim_volume_or_levels_predict_direction(monkeypatch) -> None:
    assert _decision("01", monkeypatch, low_volume=True)["size_factor"] == 1.0
    volume = _decision("02", monkeypatch, low_volume=True)
    assert volume["size_factor"] == 0.5 and volume["decision"] == "enter"
    leader = _decision("03", monkeypatch, leaders={"ETH/USDT:USDT": "opposed"})
    assert leader["decision"] == "abstain"
    no_level = _decision("04", monkeypatch)
    same_level = _decision("04", monkeypatch, level_same=True)
    assert no_level["decision"] == "abstain" and same_level["decision"] == "enter"
    room = _strategy("05")
    # The short-only account must reject a long source before considering geometry.
    with pytest.raises(ValueError, match="inactive-source"):
        room._decision("BTC/USDT:USDT", "long", "sieve:breakout_long",
                       datetime(2026, 9, 27, tzinfo=timezone.utc), 100.0)
    short_room = _decision("05", monkeypatch, side="short", tag="sieve:resistance_short",
                           barrier=True)
    assert short_room["decision"] == "enter" and short_room["size_factor"] == 0.5


def test_event_book_media_and_high_mix_are_distinct(monkeypatch) -> None:
    assert _decision("06", monkeypatch)["decision"] == "abstain"
    assert _decision("06", monkeypatch, event=True)["decision"] == "enter"
    book = _decision("07", monkeypatch, book=(-0.20, "observed"))
    assert book["size_factor"] == 0.5
    assert _decision("07", monkeypatch, book=(None, "low_coverage"))["size_factor"] == 1.0
    adverse = LunaContext(status="observed", risk_bias="risk_off", event_scale="major",
                          sources=("https://example.com/event",))
    assert _decision("08", monkeypatch, luna=adverse,
                     leaders={"ETH/USDT:USDT": "opposed"})["decision"] == "abstain"
    assert _decision("09", monkeypatch, low_volume=True, level_same=True)["decision"] == "abstain"
    assert _decision("09", monkeypatch, level_same=True)["decision"] == "enter"
    assert _decision("10", monkeypatch, luna=adverse,
                     leaders={"ETH/USDT:USDT": "opposed"})["decision"] == "abstain"
    assert _decision("10", monkeypatch, luna=LunaContext(status="missing"))[
        "observed"]["luna"]["status"] == "missing"


def test_bitcoin_does_not_confirm_itself_as_an_independent_leader() -> None:
    assert not variants.PaperVariant03._leader_alignment(
        "BTC/USDT:USDT", {"ETH/USDT:USDT": "quiet"}
    )[0]
    assert variants.PaperVariant03._leader_alignment(
        "SOL/USDT:USDT", {"BTC/USDT:USDT": "aligned", "ETH/USDT:USDT": "quiet"}
    )[0]


def test_sieve_exit_dispatch_is_kept_except_in_explicit_context_exit_variant(monkeypatch) -> None:
    now = datetime(2026, 9, 27, tzinfo=timezone.utc)
    plain = _strategy("01")
    plain.config.pop("paper_force_close", None)
    plain.dp = object()
    fake_source = SimpleNamespace(dp=None, custom_exit=lambda *args, **kwargs: "sieve_exit")
    monkeypatch.setattr(plain, "_trade_source", lambda trade: fake_source)
    assert plain.custom_exit("BTC/USDT:USDT", object(), now, 100.0, 0.01) == "sieve_exit"

    contextual = _strategy("10")
    contextual.config.pop("paper_force_close", None)
    monkeypatch.setattr(IntegratedPaper, "custom_exit", lambda *args, **kwargs: "context_checked")
    assert contextual.custom_exit("BTC/USDT:USDT", object(), now, 100.0, 0.01) == "context_checked"


def test_live_indicator_shape_for_light_and_heavy_profiles() -> None:
    def candles(count: int, freq: str) -> pd.DataFrame:
        x = np.arange(count, dtype=float)
        close = 100 + x * 0.08 + np.sin(x / 5)
        return pd.DataFrame({
            "date": pd.date_range("2025-01-01", periods=count, freq=freq, tz="UTC"),
            "open": close - 0.1, "high": close + 0.8, "low": close - 0.8,
            "close": close, "volume": 1000 + x % 25,
        })

    class Provider:
        data = {"1h": candles(450, "1h"), "4h": candles(250, "4h"),
                "1d": candles(120, "1D")}

        def get_pair_dataframe(self, pair, timeframe):
            assert pair == "BTC/USDT:USDT"
            return self.data[timeframe].copy()

        def current_whitelist(self):
            return ["BTC/USDT:USDT"]

    for variant_id in ("01", "04", "10"):
        strategy = _strategy(variant_id)
        strategy.dp = Provider()
        frame = strategy.populate_indicators(
            Provider.data["1h"].copy(), {"pair": "BTC/USDT:USDT"}
        )
        selected = strategy.populate_entry_trend(frame, {"pair": "BTC/USDT:USDT"})
        assert len(selected) == 450
        assert {f"candidate_{name}" for name in strategy._sources} <= set(selected)
        assert ("paper_atr" in selected) == ("level" in strategy.spec["inputs"])
