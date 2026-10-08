"""Causal signals and entry-frozen protection for the two new 1h PAPER rules."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from freqtrade.enums import RunMode
from user_data.strategies import paper_fast_reaction as fast
from user_data.strategies import paper_swing_levels as swing

NOW = datetime(2026, 10, 6, 12, tzinfo=timezone.utc)
PAIRS = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
         "DOGE/USDT:USDT", "1000PEPE/USDT:USDT"]


def config(account, strategy):
    return {"dry_run": True, "runmode": RunMode.DRY_RUN, "trading_mode": "futures",
        "margin_mode": "isolated", "bot_name": f"paper_{account}", "max_open_trades": 3,
        "db_url": f"sqlite:///user_data/research_news_data/context_features/integrated_paper_20260926/{account}_trades.sqlite",
        "strategy": strategy, "exchange": {"name": "binance", "key": "", "secret": "",
        "pair_whitelist": PAIRS.copy()}}


def signal_frame(closes, opens=None, highs=None, lows=None):
    count = len(closes)
    frame = pd.DataFrame({"date": pd.date_range("2026-10-06 08:00", periods=count, freq="1h", tz="UTC"),
        "open": opens or closes, "high": highs or [max(c + 1, o) for c, o in zip(closes, opens or closes)],
        "low": lows or [min(c - 1, o) for c, o in zip(closes, opens or closes)],
        "close": closes, "volume": [150.] * count, "paper_atr": [1.] * count,
        "prior_volume": [100.] * count})
    for column in swing.LEVEL_COLUMNS:
        frame[column] = 130. if "high" in column or "above" in column else 70.
    frame["range_high_20_4h"] = 130.
    frame["range_low_20_4h"] = 70.
    frame["day_high_1d"] = 130.
    frame["day_low_1d"] = 70.
    return frame


def test_strategies_are_isolated_1h_paper_only_and_use_no_5m_informative_data():
    bounce = swing.PaperSwingLevelBounce(config("swing_level_bounce", "PaperSwingLevelBounce"))
    break_hold = swing.PaperSwingLevelBreakHold(config("swing_level_break_hold", "PaperSwingLevelBreakHold"))
    bounce.bot_start()
    break_hold.bot_start()
    assert bounce.timeframe == break_hold.timeframe == "1h"
    assert bounce.leverage("SOL/USDT:USDT", NOW, 100., 1., 5., "swing_bounce_long", "long") == 3.
    bounce.dp = SimpleNamespace(current_whitelist=lambda: ["SOL/USDT:USDT"])
    assert set(bounce.informative_pairs()) == {("SOL/USDT:USDT", "4h"), ("SOL/USDT:USDT", "1d")}


@pytest.mark.parametrize("account,strategy_name", [
    ("swing_level_bounce", "PaperSwingLevelBounce"),
    ("swing_level_break_hold", "PaperSwingLevelBreakHold"),
])
def test_new_overlay_inherits_the_isolated_10000_usdt_paper_base(account, strategy_name):
    from pathlib import Path
    from freqtrade.configuration.load_config import load_from_files
    from freqtrade.configuration import validate_config_consistency
    from freqtrade.resolvers import StrategyResolver

    root = Path(__file__).resolve().parents[3]
    overlay = root / "user_data/configs" / f"paper_{account}.json"
    config_files = [str(root / "user_data/configs/integrated_paper_20260926_base.json"), str(overlay)]
    merged = load_from_files(config_files)
    assert merged["bot_name"] == f"paper_{account}"
    assert merged["strategy"] == strategy_name
    assert merged["dry_run"] is True
    assert merged["trading_mode"] == "futures" and merged["margin_mode"] == "isolated"
    assert merged["stake_currency"] == "USDT" and float(merged["dry_run_wallet"]) == 10000.
    assert merged["max_open_trades"] == 3
    assert merged["exchange"]["name"] == "binance" and set(merged["exchange"]["pair_whitelist"]) == set(PAIRS)
    assert not any(merged["exchange"].get(key) for key in ("key", "secret", "password", "privateKey", "private_key"))
    assert not merged.get("force_entry_enable") and not merged.get("api_server", {}).get("enabled")
    merged.update(runmode=RunMode.DRY_RUN, strategy_path=str(root / "user_data/strategies"),
                  user_data_dir=root / "user_data")
    strategy = StrategyResolver.load_strategy(merged)
    assert strategy.__class__.__name__ == strategy_name
    strategy.bot_start()
    strategy.ft_load_hyper_params()
    StrategyResolver.validate_strategy(strategy)
    validate_config_consistency(merged)


def test_bounce_is_touch_and_rejection_without_direction_flip_or_future_dependency():
    strategy = swing.PaperSwingLevelBounce(config("swing_level_bounce", "PaperSwingLevelBounce"))
    frame = signal_frame([102., 102., 101.], [102., 102., 100.5], [103., 103., 102.], [101.5, 101.5, 99.9])
    frame["range_low_20_4h"] = 100.
    frame["day_high_1d"] = 104.
    result = strategy.populate_entry_trend(frame.copy(), {"pair": "SOL/USDT:USDT"})
    assert result.loc[2, "enter_long"] == 1
    assert result.loc[2, "enter_short"] == 0
    assert result.loc[2, "enter_tag"] == "swing_bounce_long"
    assert result.loc[2, "fast_trigger_level"] == 100.
    assert result.loc[2, "swing_level_source"] == "4h_rolling20_low"
    assert result.loc[2, "swing_target_level"] == 104.
    assert strategy._entry_plan_context(result.loc[2], "long") == {
        "level_source": "4h_rolling20_low", "signal_kind": "bounce",
        "target_level": 104., "target_source": "previous_daily_high"}

    no_next_level = frame.copy()
    no_next_level.loc[:, swing.LEVEL_COLUMNS] = 50.
    no_next_level["range_low_20_4h"] = 100.
    fallback = strategy.populate_entry_trend(no_next_level, {})
    assert fallback.loc[2, "enter_long"] == 1
    assert pd.isna(fallback.loc[2, "swing_target_level"])
    assert pd.isna(fallback.loc[2, "swing_target_source"])
    assert strategy._entry_plan_context(fallback.loc[2], "long")["target_level"] is None

    appended = pd.concat([frame, frame.iloc[-1:].assign(date=frame.iloc[-1]["date"] + pd.Timedelta(hours=1),
        close=130., high=131., volume=9999.)], ignore_index=True)
    after = strategy.populate_entry_trend(appended, {"pair": "SOL/USDT:USDT"})
    pd.testing.assert_series_equal(result["enter_tag"], after.loc[:2, "enter_tag"])


def test_bounce_short_mirror_and_same_bar_opposition_are_suppressed():
    strategy = swing.PaperSwingLevelBounce(config("swing_level_bounce", "PaperSwingLevelBounce"))
    frame = signal_frame([98., 98., 99.], [98., 98., 99.5], [98.5, 98.5, 100.1], [97., 97., 98.5])
    frame["range_high_20_4h"] = 100.
    result = strategy.populate_entry_trend(frame.copy(), {})
    assert result.loc[2, "enter_short"] == 1 and result.loc[2, "enter_long"] == 0
    assert result.loc[2, "enter_tag"] == "swing_bounce_short"
    assert result.loc[2, "fast_trigger_level"] == 100.
    assert result.loc[2, "swing_level_source"] == "4h_rolling20_high"

    conflict = signal_frame([100., 100., 100.], [100., 100., 100.], [100., 100., 101.5], [100., 100., 98.5])
    conflict["range_low_20_4h"] = 99.
    conflict["range_high_20_4h"] = 101.
    result = strategy.populate_entry_trend(conflict, {})
    assert result.loc[2, "enter_long"] == result.loc[2, "enter_short"] == 0


def test_break_hold_requires_a_cross_then_next_hour_close_beyond_same_frozen_level():
    strategy = swing.PaperSwingLevelBreakHold(config("swing_level_break_hold", "PaperSwingLevelBreakHold"))
    frame = signal_frame([99., 101., 101.2], [99., 100., 100.5], [100., 102., 101.5], [98.5, 99., 100.5])
    frame["range_high_20_4h"] = 100.
    frame["range_low_20_4h"] = 70.
    frame["day_high_1d"] = 104.
    result = strategy.populate_entry_trend(frame.copy(), {})
    assert result.loc[1, "enter_long"] == 0
    assert result.loc[2, "enter_long"] == 1
    assert result.loc[2, "enter_tag"] == "swing_break_hold_long"
    assert result.loc[2, "fast_trigger_level"] == 100.
    assert result.loc[2, "swing_level_source"] == "4h_rolling20_high"

    not_held = frame.copy()
    not_held.loc[2, "close"] = 100.
    assert strategy.populate_entry_trend(not_held, {})["enter_long"].sum() == 0

    # A level update on the crossing candle cannot move the comparison price.
    changed = frame.copy()
    changed.loc[1:, "range_high_20_4h"] = 105.
    changed.loc[1:, "day_high_1d"] = 105.
    result = strategy.populate_entry_trend(changed, {})
    assert result.loc[2, "enter_long"] == 1
    assert result.loc[2, "fast_trigger_level"] == 100.


def test_higher_timeframe_levels_join_only_after_their_source_candle_closes(monkeypatch):
    monkeypatch.setattr(swing, "add_volume_profile", lambda frame: frame.assign(
        vp_poc=frame["close"], vp_hvn_above=frame["close"] + 1., vp_hvn_below=frame["close"] - 1.))
    strategy = swing.PaperSwingLevelBounce(config("swing_level_bounce", "PaperSwingLevelBounce"))
    base_dates = pd.date_range("2026-09-27", "2026-09-30 23:00", freq="1h", tz="UTC")
    base = pd.DataFrame({"date": base_dates, "open": 100., "high": 101., "low": 99.,
                         "close": 100., "volume": 100.})
    def source(pair, timeframe):
        step = {"4h": "4h", "1d": "1D"}[timeframe]
        dates = pd.date_range("2026-09-24", "2026-09-30", freq=step, tz="UTC")
        result = pd.DataFrame({"date": dates, "open": 100., "high": 101., "low": 99.,
            "close": 100., "volume": 100.})
        if timeframe == "4h":
            result.loc[result["date"] == pd.Timestamp("2026-09-29 04:00", tz="UTC"),
                       ["high", "close"]] = [200., 190.]
        else:
            result.loc[result["date"] == pd.Timestamp("2026-09-29", tz="UTC"), "high"] = 210.
        return result
    strategy.dp = SimpleNamespace(get_pair_dataframe=source)
    result = strategy.populate_indicators(base, {"pair": "SOL/USDT:USDT"})
    by_date = result.set_index("date")
    assert by_date.loc[pd.Timestamp("2026-09-29 06:00", tz="UTC"), "vp_poc_4h"] == 100.
    assert by_date.loc[pd.Timestamp("2026-09-29 07:00", tz="UTC"), "vp_poc_4h"] == 190.
    assert by_date.loc[pd.Timestamp("2026-09-29 22:00", tz="UTC"), "day_high_1d"] == 101.
    assert by_date.loc[pd.Timestamp("2026-09-29 23:00", tz="UTC"), "day_high_1d"] == 210.
    assert "vp_lvn_above_4h" not in result.columns and "vp_lvn_below_4h" not in result.columns


@pytest.mark.parametrize("boundary,publication_hour,level_column,expected_source", [
    ("4h", "2026-09-29 07:00", "range_low_20_4h", "4h_rolling20_low"),
    ("1d", "2026-09-29 23:00", "day_low_1d", "previous_daily_low"),
])
def test_bounce_uses_level_known_before_reaction_hour_at_higher_timeframe_publication(
    monkeypatch, boundary, publication_hour, level_column, expected_source,
):
    monkeypatch.setattr(swing, "add_volume_profile", lambda frame: frame.assign(
        vp_poc=frame["close"], vp_hvn_above=frame["close"] + 5., vp_hvn_below=frame["close"] - 5.))
    strategy = swing.PaperSwingLevelBounce(config("swing_level_bounce", "PaperSwingLevelBounce"))
    base_dates = pd.date_range("2026-09-27", "2026-09-30 23:00", freq="1h", tz="UTC")
    base = pd.DataFrame({"date": base_dates, "open": 150., "high": 151., "low": 149.,
                         "close": 150., "volume": 100.})

    def source(pair, timeframe):
        step = {"4h": "4h", "1d": "1D"}[timeframe]
        dates = pd.date_range("2026-09-24", "2026-10-01", freq=step, tz="UTC")
        result = pd.DataFrame({"date": dates, "open": 150., "high": 160., "low": 140.,
            "close": 150., "volume": 100.})
        if boundary == "4h" and timeframe == "4h":
            result.loc[result["date"] == pd.Timestamp("2026-09-29 04:00", tz="UTC"), "low"] = 100.
        if boundary == "1d" and timeframe == "1d":
            result.loc[result["date"] == pd.Timestamp("2026-09-29", tz="UTC"), "low"] = 100.
        return result

    strategy.dp = SimpleNamespace(get_pair_dataframe=source)
    merged = strategy.populate_indicators(base, {"pair": "SOL/USDT:USDT"})
    merged["paper_atr"] = 1.
    merged["prior_volume"] = 100.
    merged["volume"] = 150.

    publication = pd.Timestamp(publication_hour, tz="UTC")
    reaction = merged["date"].eq(publication)
    next_hour = merged["date"].eq(publication + pd.Timedelta(hours=1))
    prior_hour = merged["date"].eq(publication - pd.Timedelta(hours=1))
    assert merged.loc[prior_hour, level_column].iloc[0] == 140.
    assert merged.loc[reaction, level_column].iloc[0] == 100.
    assert merged.loc[next_hour, level_column].iloc[0] == 100.

    merged.loc[prior_hour, ["open", "high", "low", "close"]] = [101., 102., 100.7, 101.]
    merged.loc[reaction, ["open", "high", "low", "close"]] = [101., 101.2, 99.9, 100.5]
    merged.loc[next_hour, ["open", "high", "low", "close"]] = [100.5, 101.2, 99.9, 100.5]
    result = strategy.populate_entry_trend(merged, {})

    assert result.loc[reaction, "enter_long"].iloc[0] == 0
    assert result.loc[next_hour, "enter_long"].iloc[0] == 1
    assert result.loc[next_hour, "fast_trigger_level"].iloc[0] == 100.
    assert result.loc[next_hour, "swing_level_source"].iloc[0] == expected_source
    assert swing.swing_stop_price("long", 100.5, 1., result.loc[next_hour, "fast_trigger_level"].iloc[0]) == 99.0


def test_wider_structural_stop_reduces_size_inputs_and_target_requires_cost_adjusted_15r():
    near = swing.swing_stop_price("long", 100., 1., 99.8)
    wide = swing.swing_stop_price("long", 100., 1., 97.)
    assert near == pytest.approx(98.5)
    assert wide == pytest.approx(96.5)
    assert abs(100. - wide) > abs(100. - near)
    assert swing.swing_stop_price("short", 100., 1., 100.2) == pytest.approx(101.5)
    assert swing.swing_target_price("long", 100., near, 105.) == (105., "substantial_level")
    assert swing.swing_target_price("long", 100., near, 102.) == (103., "two_r_volatility")
    assert swing.swing_stop_within_emergency(100., 94., 3.)
    assert not swing.swing_stop_within_emergency(100., 93., 3.)

    strategy = swing.PaperSwingLevelBounce(config("swing_level_bounce", "PaperSwingLevelBounce"))
    near_geometry = strategy._sizing_geometry("long", 100., 1., 99.8, 104., 3.)
    wide_geometry = strategy._sizing_geometry("long", 100., 1., 97., 104., 3.)
    assert wide_geometry["unit_loss_fraction"] > near_geometry["unit_loss_fraction"]
    assert near_geometry["target_kind"] == "substantial_level"
    assert wide_geometry["target_kind"] == "two_r_volatility"
    with pytest.raises(ValueError, match="fixed 3x"):
        strategy._sizing_geometry("long", 100., 1., 99.8, 104., 2.)
    for fill in (99.7, 99.75, 100.25, 100.3):
        stop = swing.swing_stop_price("long", fill, 1., 99.8)
        actual_unit_loss = 3. * (abs(fill-stop)/fill + swing.FEE_AND_SLIPPAGE)
        assert actual_unit_loss <= near_geometry["unit_loss_fraction"] + 1e-9
        adjusted, changed = swing.cap_swing_stop_to_reserve(
            "long", fill, stop, 500., 3., 500.*near_geometry["unit_loss_fraction"])
        assert not changed and adjusted == pytest.approx(stop)
    with pytest.raises(ValueError, match="20% emergency"):
        strategy._sizing_geometry("long", 100., 1., 93., 110., 3.)


@pytest.mark.parametrize("side,entry,stop", [("long", 100.4, 96.5), ("short", 99.6, 103.5)])
def test_post_fill_slippage_tightens_to_reserved_loss_without_widening(side, entry, stop):
    stake, leverage, reserve = 500., 3., 40.
    adjusted, changed = swing.cap_swing_stop_to_reserve(side, entry, stop, stake, leverage, reserve)
    assert changed
    assert abs(entry-adjusted) < abs(entry-stop)
    assert stake * leverage * (abs(entry-adjusted)/entry + swing.FEE_AND_SLIPPAGE) == pytest.approx(reserve)


@pytest.mark.parametrize("side,level", [("long", 99.8), ("short", 100.2)])
def test_size_reserve_covers_both_confirm_and_fill_rounding_bounds(side, level):
    account = "swing_level_bounce"
    strategy = swing.PaperSwingLevelBounce(config(account, "PaperSwingLevelBounce"))
    geometry = strategy._sizing_geometry(side, 100., 1., level, None, 3.)
    for fill in (99.7, 99.75, 100.25, 100.3):
        stop = swing.swing_stop_price(side, fill, 1., level)
        actual_unit_loss = 3. * (abs(fill-stop)/fill + swing.FEE_AND_SLIPPAGE)
        assert actual_unit_loss <= geometry["unit_loss_fraction"] + 1e-9


def test_out_of_reserve_fill_persists_a_tighter_marked_plan_instead_of_exceeding_loss_cap():
    plan = make_plan(entry=101., level=100.)
    adjusted, changed = swing.cap_swing_stop_to_reserve("long", 101.5, plan["stop_price"],
        plan["stake_usdt"], plan["leverage"], plan["planned_loss_usdt"])
    assert changed and adjusted > plan["stop_price"]
    target, target_kind = swing.swing_target_price("long", 101.5, adjusted, None)
    plan.update(open_rate=101.5, stop_price=adjusted, target_price=target,
        target_kind=target_kind, protection_adjustment="risk_capped_slippage")
    swing.validate_swing_plan(plan)


def make_plan(side="long", *, entry=101., level=100., atr=1., stake=1000., leverage=3.):
    stop = swing.swing_stop_price(side, entry, atr, level)
    target, target_kind = swing.swing_target_price(side, entry, stop, None)
    planned_loss = stake * leverage * (abs(entry - stop) / entry + swing.FEE_AND_SLIPPAGE)
    return {"side": side, "open_rate": entry, "entry_atr": atr, "stop_price": stop,
        "target_price": target, "planned_loss_usdt": planned_loss, "entry_equity_usdt": 10000.,
        "leverage": leverage, "trigger_level": level, "level_source": "4h_rolling20_low" if side == "long" else "4h_rolling20_high",
        "signal_kind": "bounce", "target_kind": target_kind, "stake_usdt": stake,
        "target_level": None}


def test_persisted_plan_checks_stop_buffer_emergency_ceiling_and_reserved_loss():
    plan = make_plan()
    swing.validate_swing_plan(plan)
    swing.validate_swing_plan(plan, actual_leverage=3.)
    with pytest.raises(ValueError, match="differs from the persisted trade"):
        swing.validate_swing_plan(plan, actual_leverage=2.)
    wrong_leverage = plan.copy()
    wrong_leverage["leverage"] = 2.
    with pytest.raises(ValueError, match="fixed 3x"):
        swing.validate_swing_plan(wrong_leverage)
    bad_stop = plan.copy()
    bad_stop["stop_price"] = 94.
    bad_stop["target_price"] = 115.
    bad_stop["stake_usdt"] = 500.
    bad_stop["planned_loss_usdt"] = bad_stop["stake_usdt"] * bad_stop["leverage"] * (
        abs(bad_stop["open_rate"] - bad_stop["stop_price"]) / bad_stop["open_rate"] + swing.FEE_AND_SLIPPAGE
    )
    with pytest.raises(ValueError, match="emergency margin-loss ceiling"):
        swing.validate_swing_plan(bad_stop)
    bad_reserve = plan.copy()
    bad_reserve["planned_loss_usdt"] -= 1.
    with pytest.raises(ValueError, match="reserved planned loss"):
        swing.validate_swing_plan(bad_reserve)

    level_target = make_plan(entry=101., level=100.)
    level_target["target_level"] = 104.
    level_target["target_source"] = "previous_daily_high"
    level_target["target_price"], level_target["target_kind"] = swing.swing_target_price(
        "long", level_target["open_rate"], level_target["stop_price"], level_target["target_level"])
    swing.validate_swing_plan(level_target)


def test_invalidation_waits_for_two_completed_hourly_closes_and_no_one_hour_stall():
    strategy = swing.PaperSwingLevelBounce(config("swing_level_bounce", "PaperSwingLevelBounce"))
    plan = make_plan(entry=101., level=100.)
    stored = {swing.PLAN_KEY: plan}
    trade = SimpleNamespace(is_short=False, leverage=3., open_date_utc=datetime(2026, 10, 6, 9, 30, tzinfo=timezone.utc),
        get_custom_data=lambda key: stored.get(key))
    dates = pd.date_range("2026-10-06 09:00", periods=3, freq="1h", tz="UTC")
    frame = pd.DataFrame({"date": dates, "close": [99., 101., 99.]})
    strategy.dp = SimpleNamespace(get_analyzed_dataframe=lambda pair, timeframe: (frame, NOW))
    assert strategy.custom_exit("SOL/USDT:USDT", trade, NOW, 104., 0.) == "swing_price_target"
    assert strategy.custom_exit("SOL/USDT:USDT", trade, datetime(2026, 10, 6, 11, 30, tzinfo=timezone.utc), 101., 0.) is None
    frame.loc[1, "close"] = 99.
    assert strategy.custom_exit("SOL/USDT:USDT", trade, datetime(2026, 10, 6, 11, 30, tzinfo=timezone.utc), 101., 0.) == "swing_level_invalidated"
    assert strategy.custom_exit("SOL/USDT:USDT", trade, trade.open_date_utc + timedelta(hours=1), 101., 0.) is None
    assert strategy.custom_exit("SOL/USDT:USDT", trade, trade.open_date_utc + timedelta(hours=48), 101., 0.) == "swing_time_limit"


def test_current_row_clock_uses_one_hour_closes_and_rejects_stale_data():
    strategy = swing.PaperSwingLevelBounce(config("swing_level_bounce", "PaperSwingLevelBounce"))
    frame = pd.DataFrame({"date": [pd.Timestamp("2026-10-06 11:00", tz="UTC")], "close": [100.]})
    strategy.dp = SimpleNamespace(get_analyzed_dataframe=lambda pair, timeframe: (frame, NOW))
    assert strategy._current_row("SOL/USDT:USDT", datetime(2026, 10, 6, 12, tzinfo=timezone.utc))["close"] == 100.
    with pytest.raises(ValueError, match="Stale"):
        strategy._current_row("SOL/USDT:USDT", datetime(2026, 10, 6, 13, tzinfo=timezone.utc))


def _prepared_swing_entry(monkeypatch, *, side="long"):
    account = "swing_level_bounce"
    strategy = swing.PaperSwingLevelBounce(config(account, "PaperSwingLevelBounce"))
    reference = 100.2 if side == "long" else 99.8
    frame = pd.DataFrame({"date": [pd.Timestamp(NOW - timedelta(hours=1))], "close": [reference],
        "paper_atr": [2.], "fast_trigger_level": [100.], "fast_strong": [False],
        "swing_level_source": ["4h_rolling20_low" if side == "long" else "4h_rolling20_high"],
        "swing_target_level": [112. if side == "long" else 88.],
        "swing_target_source": ["previous_daily_high" if side == "long" else "previous_daily_low"]})
    strategy.dp = SimpleNamespace(get_analyzed_dataframe=lambda pair, timeframe: (frame, NOW))
    strategy.wallets = SimpleNamespace(get_total_stake_amount=lambda: 10000.)
    monkeypatch.setattr(fast.Trade, "get_open_trades", lambda: [])
    tag = f"swing_bounce_{side}"
    stake = strategy.custom_stake_amount("SOL/USDT:USDT", NOW, reference, 3333., 10., 3333., 3., tag, side)
    return strategy, frame, side, tag, stake


@pytest.mark.parametrize("side", ["long", "short"])
def test_inherited_entry_and_fill_hooks_use_frozen_swing_protection_within_slippage_reserve(monkeypatch, side):
    strategy, _, side, tag, stake = _prepared_swing_entry(monkeypatch, side=side)
    pending = strategy._pending[("SOL/USDT:USDT", side, tag)]
    assert 0. < stake <= 2500.
    assert pending["planned_loss_usdt"] <= 200. + 1e-6
    lower, upper = pending["fill_rate_bounds"]
    expected_bounds = (99.6, 100.8) if side == "long" else (99.2, 100.4)
    assert lower == pytest.approx(expected_bounds[0]) and upper == pytest.approx(expected_bounds[1])

    side_sign = -1. if side == "short" else 1.
    reference = 100.2 if side == "long" else 99.8
    accepted_rate = reference + side_sign * .5  # exactly the inherited ±0.25 ATR quote envelope
    assert strategy.confirm_trade_entry("SOL/USDT:USDT", "market", stake * 3 / accepted_rate,
        accepted_rate, "gtc", NOW, tag, side)
    stored = {}
    trade = SimpleNamespace(id=1, open_rate=accepted_rate, stake_amount=stake, is_short=side == "short",
        leverage=3., enter_tag=tag, entry_side="sell" if side == "short" else "buy", open_date_utc=NOW,
        get_custom_data=lambda key: stored.get(key),
        set_custom_data=lambda key, value: stored.__setitem__(key, value))
    strategy.order_filled("SOL/USDT:USDT", trade, SimpleNamespace(ft_order_side=trade.entry_side), NOW)
    plan = stored[swing.PLAN_KEY]
    swing.validate_swing_plan(plan, actual_leverage=3.)
    actual_loss = stake * 3. * (abs(accepted_rate - plan["stop_price"]) / accepted_rate + swing.FEE_AND_SLIPPAGE)
    assert actual_loss <= plan["planned_loss_usdt"] + 1e-6
    assert plan.get("protection_adjustment") is None  # normal permitted quote/fill does not churn-exit
    assert plan["target_kind"] == "substantial_level"
    assert strategy.custom_stoploss("SOL/USDT:USDT", trade, NOW, accepted_rate, 0., False) > 0.
    beyond = plan["target_price"] + side_sign * .1
    assert strategy.custom_exit("SOL/USDT:USDT", trade, NOW, beyond, 0.) == "swing_price_target"


def test_out_of_reserve_filled_order_tightens_stop_and_exits_without_exceeding_reserved_loss(monkeypatch):
    strategy, _, side, tag, stake = _prepared_swing_entry(monkeypatch)
    pending = strategy._pending[("SOL/USDT:USDT", side, tag)]
    fill = pending["fill_rate_bounds"][0] - .2
    stored = {}
    trade = SimpleNamespace(id=1, open_rate=fill, stake_amount=stake, is_short=False,
        leverage=3., enter_tag=tag, entry_side="buy", open_date_utc=NOW,
        get_custom_data=lambda key: stored.get(key),
        set_custom_data=lambda key, value: stored.__setitem__(key, value))
    strategy.order_filled("SOL/USDT:USDT", trade, SimpleNamespace(ft_order_side="buy"), NOW)
    plan = stored[swing.PLAN_KEY]
    assert plan["protection_adjustment"] == "risk_capped_slippage"
    assert plan["stop_price"] > swing.swing_stop_price("long", fill, 2., 100.)
    actual_loss = stake * 3. * (abs(fill - plan["stop_price"]) / fill + swing.FEE_AND_SLIPPAGE)
    assert actual_loss <= plan["planned_loss_usdt"] + 1e-6
    swing.validate_swing_plan(plan, actual_leverage=3.)
    assert strategy.custom_exit("SOL/USDT:USDT", trade, NOW, fill, 0.) == "swing_slippage_risk_cap"


def test_fill_replaces_a_too_close_level_candidate_with_a_plain_two_r_target(monkeypatch):
    strategy, _, side, tag, stake = _prepared_swing_entry(monkeypatch)
    pending = strategy._pending[("SOL/USDT:USDT", side, tag)]
    pending.update(target_level=104., target_source="previous_daily_high")
    fill = 100.7
    stored = {}
    trade = SimpleNamespace(id=1, open_rate=fill, stake_amount=stake, is_short=False,
        leverage=3., enter_tag=tag, entry_side="buy", open_date_utc=NOW,
        get_custom_data=lambda key: stored.get(key),
        set_custom_data=lambda key, value: stored.__setitem__(key, value))
    strategy.order_filled("SOL/USDT:USDT", trade, SimpleNamespace(ft_order_side="buy"), NOW)
    plan = stored[swing.PLAN_KEY]
    assert plan["target_kind"] == "two_r_volatility"
    assert plan["target_level"] is None and plan["target_source"] is None
    assert plan["target_price"] == pytest.approx(fill + 2. * abs(fill - plan["stop_price"]))
    swing.validate_swing_plan(plan, actual_leverage=3.)
