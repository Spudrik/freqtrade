"""Checks for the separate directional, high-risk paper hypothesis."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from freqtrade.enums import RunMode
from user_data.strategies.integrated_paper_context import LunaContext
from user_data.strategies.paper_leader_impulse import PaperLeaderImpulse
from user_data.strategies.paper_inverse_signals import PaperLeaderInverse
from user_data.Custom_Launcher.research.context_features.paper_leader_impulse_signal_review import (
    _outcome,
    _separated,
    _summarize,
)


def _config() -> dict:
    with open("user_data/configs/integrated_paper_20260926_base.json", encoding="utf-8") as handle:
        config = json.load(handle)
    with open("user_data/configs/integrated_paper_leader_impulse.json", encoding="utf-8") as handle:
        config.update(json.load(handle))
    config["runmode"] = RunMode.DRY_RUN
    return config


def test_paper_only_and_two_times_leverage() -> None:
    strategy = PaperLeaderImpulse(_config())
    strategy.bot_start()
    assert strategy.leverage("SOL/USDT:USDT", None, 100.0, 1.0, 10.0, None, "long") == 2.0
    for field, value in (("dry_run", False), ("runmode", RunMode.LIVE), ("force_entry_enable", True)):
        config = _config()
        config[field] = value
        with pytest.raises(RuntimeError):
            PaperLeaderImpulse(config).bot_start()


def test_leader_and_local_break_must_align() -> None:
    strategy = PaperLeaderImpulse(_config())
    frame = pd.DataFrame({
        "paper_atr": [1.0, 1.0, 1.0],
        "local_prior_volume": [100.0, 100.0, 100.0],
        "btc_prior_volume": [100.0, 100.0, 100.0],
        "volume": [120.0, 120.0, 120.0],
        "btc_volume": [150.0, 150.0, 150.0],
        "btc_change": [0.007, -0.007, 0.007],
        "close": [102.0, 98.0, 99.0],
        "local_prior_6h_high": [101.0, 101.0, 101.0],
        "local_prior_6h_low": [99.0, 99.0, 99.0],
    })
    result = strategy.populate_entry_trend(frame, {"pair": "SOL/USDT:USDT"})
    assert result.loc[0, "enter_tag"] == "btc_impulse_local_break_long"
    assert result.loc[1, "enter_tag"] == "btc_impulse_local_break_short"
    assert pd.isna(result.loc[2, "enter_tag"])
    btc_result = strategy.populate_entry_trend(frame.copy(), {"pair": "BTC/USDT:USDT"})
    assert btc_result["enter_long"].eq(0).all()
    assert btc_result["enter_short"].eq(0).all()
    assert btc_result["enter_tag"].isna().all()


def test_inverse_leader_reverses_only_the_completed_original_signals() -> None:
    config = _config()
    config.update(bot_name="paper_leader_inverse", strategy="PaperLeaderInverse",
                  db_url="sqlite:///user_data/research_news_data/context_features/"
                         "integrated_paper_20260926/leader_inverse_trades.sqlite")
    inverse = PaperLeaderInverse(config)
    inverse.bot_start()
    frame = pd.DataFrame({
        "paper_atr": [1., 1.], "local_prior_volume": [100., 100.],
        "btc_prior_volume": [100., 100.], "volume": [120., 120.],
        "btc_volume": [150., 150.], "btc_change": [.007, -.007],
        "close": [102., 98.], "local_prior_6h_high": [101., 101.],
        "local_prior_6h_low": [99., 99.],
    })
    result = inverse.populate_entry_trend(frame, {"pair": "SOL/USDT:USDT"})
    assert list(result["enter_short"]) == [1, 0]
    assert list(result["enter_long"]) == [0, 1]
    assert list(result["enter_tag"]) == ["inverse_btc_impulse_local_break_short", "inverse_btc_impulse_local_break_long"]
    assert inverse.leverage("SOL/USDT:USDT", None, 100., 1., 10., None, "short") == 2.
    config["dry_run"] = False
    with pytest.raises(RuntimeError):
        PaperLeaderInverse(config).bot_start()


def test_read_only_signal_review_uses_next_open_and_separates_overlaps() -> None:
    dates = pd.date_range("2026-01-01", periods=60, freq="h", tz="UTC")
    frame = pd.DataFrame({
        "date": dates,
        "open": [100.0] * 60,
        "close": [110.0] * 60,
    })
    selected = _separated(pd.Index([0, 1, 47, 48, 59]), frame["date"])
    assert selected.tolist() == [0, 48]
    sides = pd.Series([1] * 48 + [-1] * 12)
    result = _outcome(frame, selected, sides, 6)
    assert result.tolist() == pytest.approx([10.0, -10.0])


def test_read_only_signal_review_keeps_long_and_short_outcomes_separate() -> None:
    frame = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=100, freq="h", tz="UTC"),
        "open": [100.0] * 100,
        "close": [110.0] * 100,
    })
    mask = pd.Series(False, index=frame.index)
    mask.iloc[[0, 48]] = True
    sides = pd.Series(0, index=frame.index)
    sides.iloc[0] = 1
    sides.iloc[48] = -1
    result = _summarize(frame, mask, sides)
    assert result["long_n48"] == result["short_n48"] == 1
    assert result["long_mean48"] == pytest.approx(10.0)
    assert result["short_mean48"] == pytest.approx(-10.0)


@pytest.mark.parametrize("is_short,target", [(False, 106.0), (True, 94.0)])
def test_entry_atr_stop_target_and_time_limit(is_short: bool, target: float) -> None:
    strategy = PaperLeaderImpulse(_config())
    now = datetime(2026, 9, 27, 2, 0, tzinfo=timezone.utc)
    custom_data: dict[str, float] = {}
    trade = SimpleNamespace(
        id=12,
        is_short=is_short,
        leverage=2.0,
        open_rate=100.0,
        open_date_utc=now,
        entry_side="sell" if is_short else "buy",
        get_custom_data=lambda key: custom_data.get(key),
        set_custom_data=lambda key, value: custom_data.__setitem__(key, value),
    )
    frame = pd.DataFrame({"paper_atr": [2.0]})
    strategy.dp = SimpleNamespace(get_analyzed_dataframe=lambda pair, timeframe: (frame, now))
    order = SimpleNamespace(ft_order_side=trade.entry_side)
    strategy.order_filled("SOL/USDT:USDT", trade, order, now)
    assert custom_data["paper_entry_atr"] == 2.0
    assert strategy.custom_stoploss("SOL/USDT:USDT", trade, now, 100.0, 0.0, False) == pytest.approx(0.06)
    assert strategy.custom_exit("SOL/USDT:USDT", trade, now, target, 0.0) == "paper_atr_target"
    assert strategy.custom_exit("SOL/USDT:USDT", trade, now + timedelta(hours=48), 100.0, 0.0) == "paper_time_limit"


def test_future_candles_do_not_change_prior_signal_or_four_hour_levels() -> None:
    strategy = PaperLeaderImpulse(_config())
    dates = pd.date_range("2026-01-01", periods=490, freq="h", tz="UTC")
    coin = pd.DataFrame({
        "date": dates,
        "open": [100.0] * 490,
        "high": [101.0] * 490,
        "low": [99.0] * 490,
        "close": [100.0] * 490,
        "volume": [100.0] * 490,
    })
    leader = coin.copy()
    coin.loc[479, ["high", "close", "volume"]] = [103.0, 103.0, 150.0]
    leader.loc[479, ["high", "close", "volume"]] = [102.0, 101.0, 200.0]
    coin.loc[480:, ["high", "close", "volume"]] = [160.0, 150.0, 999.0]
    leader.loc[480:, ["high", "close", "volume"]] = [160.0, 150.0, 999.0]
    four_hour = pd.DataFrame({
        "date": pd.date_range("2026-01-01", periods=123, freq="4h", tz="UTC"),
        "open": [100.0] * 123,
        "high": [101.0] * 123,
        "low": [99.0] * 123,
        "close": [100.0] * 123,
        "volume": [100.0] * 123,
    })
    four_hour.loc[120:, ["high", "close", "volume"]] = [160.0, 150.0, 999.0]
    metadata = {"pair": "SOL/USDT:USDT"}

    def analyze(length: int) -> pd.DataFrame:
        strategy.dp = SimpleNamespace(
            get_pair_dataframe=lambda pair, timeframe: (
                leader.iloc[:length].copy() if timeframe == "1h" else
                four_hour.loc[four_hour["date"] < dates[length - 1] + timedelta(hours=1)].copy()
            )
        )
        populated = strategy.populate_indicators(coin.iloc[:length].copy(), metadata)
        return strategy.populate_entry_trend(populated, metadata)

    before = analyze(480)
    after = analyze(490)
    assert before.loc[479, "enter_tag"] == after.loc[479, "enter_tag"] == "btc_impulse_local_break_long"
    for column in ("btc_change", "btc_prior_volume", "rolling_20_high_4h", "vp_poc_4h"):
        pd.testing.assert_series_equal(before[column], after.loc[:479, column])


def test_sizing_treats_nearby_level_and_opposed_sourced_event_as_cautions(monkeypatch) -> None:
    strategy = PaperLeaderImpulse(_config())
    now = datetime(2026, 9, 27, 2, 0, tzinfo=timezone.utc)
    frame = pd.DataFrame({
        "date": [pd.Timestamp("2026-09-27 01:00:00", tz="UTC")],
        "paper_atr": [2.0],
        "rolling_20_high_4h": [101.0],
        "vp_hvn_above_4h": [None],
        "vp_lvn_above_4h": [None],
    })
    strategy.dp = SimpleNamespace(get_analyzed_dataframe=lambda pair, timeframe: (frame, now))
    strategy.wallets = SimpleNamespace(get_total_stake_amount=lambda: 10000.0)
    monkeypatch.setattr(
        "user_data.strategies.paper_leader_impulse.load_luna_context",
        lambda *_: LunaContext("observed", now, "risk_off", "major", "elevated", "event", ("https://example.com",)),
    )
    tag = "btc_impulse_local_break_long"
    stake = strategy.custom_stake_amount("SOL/USDT:USDT", now, 100.0, 3333.0, 10.0, 3333.0, 2.0, tag, "long")
    assert stake == pytest.approx(500.0)
    assert strategy.confirm_trade_entry("SOL/USDT:USDT", "market", 5.0, 100.0, "gtc", now, tag, "long")
    assert not strategy.confirm_trade_entry("SOL/USDT:USDT", "market", 5.0, 100.0, "gtc", now, tag, "long")

    frame.loc[0, "rolling_20_high_4h"] = 110.0
    assert strategy.custom_stake_amount("SOL/USDT:USDT", now, 100.0, 3333.0, 10.0, 3333.0, 2.0, tag, "long") == pytest.approx(500.0)
    strategy.confirm_trade_entry("SOL/USDT:USDT", "market", 5.0, 100.0, "gtc", now, tag, "long")
    monkeypatch.setattr(
        "user_data.strategies.paper_leader_impulse.load_luna_context",
        lambda *_: LunaContext("stale", None, "unknown", "none", "unknown", "", ()),
    )
    assert strategy.custom_stake_amount("SOL/USDT:USDT", now, 100.0, 3333.0, 10.0, 3333.0, 2.0, tag, "long") == pytest.approx(1000.0)
