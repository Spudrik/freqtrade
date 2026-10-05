"""Focused guards/signals for prospective paper rules; no performance claims."""
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from freqtrade.enums import RunMode
from user_data.strategies.paper_fast_context import FastControl, parse_fast_control, load_fast_control, LEVERAGES
from user_data.strategies import paper_fast_context as control_module
from user_data.strategies.paper_fast_reaction import PaperFastAuto, PaperFastContext, validate_fast_plan, PLAN_KEY
from user_data.strategies.paper_inverse_signals import PaperFastLevelInverse

NOW = datetime(2026, 9, 29, 12, tzinfo=timezone.utc)


def config(contextual=False):
    account = "fast_context" if contextual else "fast_auto"
    return {"dry_run": True, "runmode": RunMode.DRY_RUN, "trading_mode": "futures",
            "margin_mode": "isolated", "bot_name": f"paper_{account}", "max_open_trades": 3,
            "db_url": f"sqlite:///user_data/research_news_data/context_features/integrated_paper_20260926/{account}_trades.sqlite",
            "exchange": {"name": "binance", "key": "", "secret": "",
                         "pair_whitelist": ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT", "DOGE/USDT:USDT", "1000PEPE/USDT:USDT"]}}


def control_row():
    return {"schema_version": 1, "author": "main_agent", "decision_id": "test_approved",
            "observed_at_utc": NOW.isoformat(), "valid_until_utc": (NOW+timedelta(hours=4)).isoformat(),
            "bias": 0, "exposure": "normal", "entry_permission": "normal", "reason": "test",
            "sources": ["https://example.com"], "blackouts": []}


@pytest.mark.parametrize("bias", range(-2,3))
def test_leverage_map(bias):
    row = control_row(); row["bias"] = bias
    parsed = parse_fast_control(row, NOW)
    for side in ("long", "short"):
        assert parsed.leverage_for(side, NOW) == LEVERAGES[bias][side]


@pytest.mark.parametrize("field,value", [("bias", True), ("bias", 3), ("author", "luna"),
    ("exposure", "double"), ("entry_permission", "anything"), ("sources", []),
    ("valid_until_utc", (NOW+timedelta(hours=5)).isoformat())])
def test_invalid_control_refused(field,value):
    row = control_row(); row[field] = value
    with pytest.raises(ValueError): parse_fast_control(row,NOW)


def test_stale_missing_malformed_and_blackout_block_only_new_entries(tmp_path):
    path = tmp_path/"control.json"
    assert load_fast_control(NOW,path).status == "missing"
    path.write_text("{")
    assert load_fast_control(NOW,path).status == "malformed"
    row = control_row(); row["blackouts"] = [{"start_utc": NOW.isoformat(), "end_utc": (NOW+timedelta(minutes=15)).isoformat()}]
    parsed = parse_fast_control(row,NOW)
    assert parsed.leverage_for("long",NOW) == 0
    assert parsed.leverage_for("long",NOW+timedelta(minutes=15)) == 3
    assert parse_fast_control(row,NOW+timedelta(hours=4)).status == "stale"


def test_control_publication_is_sourced_journalled_and_never_retried(tmp_path,monkeypatch):
    path,journal,luna=tmp_path/"control.json",tmp_path/"decisions.jsonl",tmp_path/"luna.json"
    monkeypatch.setattr(control_module,"CONTEXT_FILE",luna)
    monkeypatch.setattr(control_module,"load_luna_context",lambda now:SimpleNamespace(status="observed",sources=("https://example.com",)))
    luna.write_text(json.dumps({"valid_until_utc":(NOW+timedelta(hours=4)).isoformat()}))
    row=control_row()
    control_module.publish_fast_control(row,now=NOW,path=path,journal=journal)
    assert json.loads(path.read_text())==row
    assert len(journal.read_text().splitlines())==1
    with pytest.raises(ValueError,match="already journalled"):
        control_module.publish_fast_control(row,now=NOW,path=path,journal=journal)
    assert len(journal.read_text().splitlines())==1


@pytest.mark.parametrize("problem",["stale_luna","unlisted_source","outlives_luna"])
def test_invalid_publication_does_not_write_control_or_journal(tmp_path,monkeypatch,problem):
    luna=tmp_path/"luna.json"
    monkeypatch.setattr(control_module,"CONTEXT_FILE",luna)
    monkeypatch.setattr(control_module,"load_luna_context",lambda now:SimpleNamespace(status="stale" if problem=="stale_luna" else "observed",sources=("https://example.com",)))
    luna.write_text(json.dumps({"valid_until_utc":(NOW+timedelta(hours=1 if problem=="outlives_luna" else 4)).isoformat()}))
    row=control_row()
    if problem=="unlisted_source":row["sources"]=["https://not-observed.example"]
    path,journal=tmp_path/"control.json",tmp_path/"decisions.jsonl"
    with pytest.raises(ValueError):
        control_module.publish_fast_control(row,now=NOW,path=path,journal=journal)
    assert not path.exists() and not journal.exists()


def signal_frame():
    frame = pd.DataFrame({"date": pd.date_range("2026-09-29 11:45",periods=3,freq="5min",tz="UTC"),
        "open": [102.,102.,100.8], "high": [103.,103.,102.], "low": [101.5,101.5,99.9],
        "close": [102.,102.,101.5], "volume": [200.,200.,200.], "paper_atr": [1.,1.,1.],
        "prior_volume": [100.,100.,100.], "confirm_change_15m": [.001,.001,.001],
        "btc_change": [0.,0.,0.], "eth_change": [0.,0.,0.],
        "range_high": [110.,110.,110.], "range_low": [90.,90.,90.]})
    for name in ("hour_high_1h", "hour_low_1h", "four_high_4h", "four_low_4h", "vp_poc_4h",
                 "vp_hvn_above_4h", "vp_hvn_below_4h", "vp_lvn_above_4h", "vp_lvn_below_4h", "day_high_1d", "day_low_1d"):
        frame[name]=100. if name == "vp_poc_4h" else 120.
    frame.loc[2,["close","open"]]=[101.,100.]
    return frame


def test_shared_signals_long_and_short_mirror_and_no_future_data():
    auto, context = PaperFastAuto(config()), PaperFastContext(config(True))
    frame = signal_frame()
    result = auto.populate_entry_trend(frame.copy(),{"pair":"SOL/USDT:USDT"})
    other = context.populate_entry_trend(frame.copy(),{"pair":"SOL/USDT:USDT"})
    assert result.loc[2,"enter_tag"] == "fast_level_long"
    pd.testing.assert_frame_equal(result,other)
    short = frame.copy()
    for name in ("open","high","low","close"):
        short[name]=200.-frame[name]
    short["high"],short["low"] = 200.-frame["low"],200.-frame["high"]
    short["confirm_change_15m"]=-.001
    mirrored = auto.populate_entry_trend(short,{"pair":"SOL/USDT:USDT"})
    assert mirrored.loc[2,"enter_tag"] == "fast_level_short"
    # Appended future bars must never change the previous decisions.
    appended = pd.concat([frame,frame.iloc[-1:].assign(close=130.,high=131.,volume=9999.)],ignore_index=True)
    after = auto.populate_entry_trend(appended,{"pair":"SOL/USDT:USDT"})
    pd.testing.assert_series_equal(result["enter_tag"],after.loc[:2,"enter_tag"])


def test_inverse_level_is_separate_and_uses_signal_extreme_for_invalidation():
    inverse_config = config()
    inverse_config["bot_name"] = "paper_fast_level_inverse"
    inverse_config["db_url"] = inverse_config["db_url"].replace("fast_auto", "fast_level_inverse")
    inverse = PaperFastLevelInverse(inverse_config)
    inverse.bot_start()
    frame = signal_frame()
    original = PaperFastAuto(config()).populate_entry_trend(frame.copy(), {"pair": "SOL/USDT:USDT"})
    result = inverse.populate_entry_trend(frame.copy(), {"pair": "SOL/USDT:USDT"})
    assert original.loc[2, "enter_tag"] == "fast_level_long"
    assert result.loc[2, "enter_short"] == 1
    assert result.loc[2, "enter_long"] == 0
    assert result.loc[2, "enter_tag"] == "inverse_fast_level_short"
    assert result.loc[2, "fast_trigger_level"] == frame.loc[2, "high"]
    assert inverse.leverage("SOL/USDT:USDT", NOW, 100., 1., 5., result.loc[2, "enter_tag"], "short") == 3.
    inverse_config["dry_run"] = False
    with pytest.raises(RuntimeError):
        PaperFastLevelInverse(inverse_config).bot_start()


def test_inverse_level_excludes_break_retest_family():
    inverse_config = config()
    inverse_config["bot_name"] = "paper_fast_level_inverse"
    inverse_config["db_url"] = inverse_config["db_url"].replace("fast_auto", "fast_level_inverse")
    frame = signal_frame()
    frame.loc[:, "vp_poc_4h"] = 120.
    frame["range_high"] = 100.
    frame.loc[0, "close"] = 99.9
    frame.loc[1, "close"] = 101.
    frame.loc[2, ["open", "close", "low"]] = [100.1, 101., 99.9]
    original = PaperFastAuto(config()).populate_entry_trend(frame.copy(), {})
    assert original.loc[2, "enter_tag"] == "fast_break_retest_long"
    result = PaperFastLevelInverse(inverse_config).populate_entry_trend(frame, {})
    assert result["enter_long"].eq(0).all() and result["enter_short"].eq(0).all()


def test_cost_floor_and_both_leaders_opposed_block_entry():
    strategy=PaperFastAuto(config()); frame=signal_frame()
    frame.loc[2,["btc_change","eth_change"]]=[-.004,-.004]
    assert strategy.populate_entry_trend(frame,{})["enter_long"].sum()==0
    frame=signal_frame();frame["paper_atr"]=.001
    assert strategy.populate_entry_trend(frame,{})["enter_long"].sum()==0


def test_held_break_retest_is_required():
    strategy=PaperFastAuto(config());frame=signal_frame()
    frame.loc[:,"vp_poc_4h"]=120.
    frame["range_high"]=100.
    frame.loc[0,"close"]=99.9
    frame.loc[1,"close"]=101.
    frame.loc[2,["open","close","low"]]=[100.1,101.,99.9]
    result=strategy.populate_entry_trend(frame,{})
    assert result.loc[2,"enter_tag"]=="fast_break_retest_long"
    frame.loc[2,"low"]=100.5
    assert strategy.populate_entry_trend(frame,{})["enter_long"].sum()==0


def prepared_strategy(contextual=False, control=None):
    strategy=(PaperFastContext if contextual else PaperFastAuto)(config(contextual))
    frame=pd.DataFrame({"date":[pd.Timestamp(NOW-timedelta(minutes=5))],"paper_atr":[1.],
                        "fast_trigger_level":[99.],"fast_strong":[True],"close":[100.]})
    strategy.dp=SimpleNamespace(get_analyzed_dataframe=lambda pair,timeframe:(frame,NOW))
    strategy.wallets=SimpleNamespace(get_total_stake_amount=lambda:10000.)
    strategy._committed_risk=lambda equity:0.
    if control is not None: strategy._control=lambda now:control
    return strategy,frame


def test_size_risk_cap_reduced_and_confirm_exact_once():
    strategy,frame=prepared_strategy()
    stake=strategy.custom_stake_amount("SOL/USDT:USDT",NOW,100.,3333.,10.,3333.,3.,"fast_level_long","long")
    plan=strategy._pending[("SOL/USDT:USDT","long","fast_level_long")]
    assert plan["planned_loss_usdt"] <= 200.
    assert stake<=2500.
    assert strategy.confirm_trade_entry("SOL/USDT:USDT","market",stake*3/100,100.,"gtc",NOW,"fast_level_long","long")
    assert not strategy.confirm_trade_entry("SOL/USDT:USDT","market",stake*3/100,100.,"gtc",NOW,"fast_level_long","long")
    control=FastControl("observed","reduced",0,"reduced","normal")
    reduced,_=prepared_strategy(True,control)
    assert reduced.custom_stake_amount("SOL/USDT:USDT",NOW,100.,3333.,10.,3333.,3.,"fast_level_long","long")==pytest.approx(stake/2)


def test_aggregate_risk_and_paused_or_weak_setups():
    strategy,frame=prepared_strategy()
    strategy._committed_risk=lambda equity:400.
    assert strategy.custom_stake_amount("SOL/USDT:USDT",NOW,100.,3333.,10.,3333.,3.,"fast_level_long","long")==0.
    strategy,frame=prepared_strategy(True,FastControl("observed","strong",0,"normal","strong_only"))
    frame.loc[0,"fast_strong"]=False
    assert strategy.custom_stake_amount("SOL/USDT:USDT",NOW,100.,3333.,10.,3333.,3.,"fast_level_long","long")==0.
    strategy,_=prepared_strategy(True,FastControl("stale"))
    assert strategy.custom_stake_amount("SOL/USDT:USDT",NOW,100.,3333.,10.,3333.,1.,"fast_level_long","long")==0.


def test_stale_candle_and_changed_control_or_price_are_refused():
    strategy,frame=prepared_strategy()
    frame.loc[0,"date"]=pd.Timestamp(NOW-timedelta(hours=1))
    with pytest.raises(ValueError,match="Stale"):
        strategy.custom_stake_amount("SOL/USDT:USDT",NOW,100.,3333.,10.,3333.,3.,"fast_level_long","long")
    strategy,frame=prepared_strategy()
    strategy.custom_stake_amount("SOL/USDT:USDT",NOW,100.,3333.,10.,3333.,3.,"fast_level_long","long")
    assert not strategy.confirm_trade_entry("SOL/USDT:USDT","market",1.,101.,"gtc",NOW,"fast_level_long","long")
    strategy._control=lambda now:FastControl("observed","changed",0,"normal","normal")
    assert not strategy.confirm_trade_entry("SOL/USDT:USDT","market",1.,100.,"gtc",NOW,"fast_level_long","long")


@pytest.mark.parametrize("side",["long","short"])
def test_fill_freezes_protection_and_context_expiry_does_not_disable_exits(side):
    strategy,frame=prepared_strategy(True,FastControl("observed","test",0,"normal","normal"))
    level=99. if side=="long" else 101.
    frame.loc[0,"fast_trigger_level"]=level
    tag=f"fast_level_{side}"
    stake=strategy.custom_stake_amount("SOL/USDT:USDT",NOW,100.,3333.,10.,3333.,3.,tag,side)
    stored={}
    trade=SimpleNamespace(id=1,open_rate=100.,is_short=side=="short",leverage=3.,enter_tag=tag,
        entry_side="sell" if side=="short" else "buy",open_date_utc=NOW,
        get_custom_data=lambda key:stored.get(key),set_custom_data=lambda key,value:stored.__setitem__(key,value))
    strategy.order_filled("SOL/USDT:USDT",trade,SimpleNamespace(ft_order_side=trade.entry_side),NOW)
    validate_fast_plan(stored[PLAN_KEY])
    assert strategy.custom_stoploss("SOL/USDT:USDT",trade,NOW,100.,0.,False)==pytest.approx(.045)
    strategy._control=lambda now:FastControl("stale")
    target=97. if side=="short" else 103.
    assert strategy.custom_exit("SOL/USDT:USDT",trade,NOW,target,0.)=="fast_atr_target"
    assert strategy.custom_exit("SOL/USDT:USDT",trade,NOW+timedelta(hours=4),100.,0.)=="fast_time_limit"


def test_boot_contract_forbids_live_or_order_api():
    from freqtrade.resolvers import StrategyResolver
    strategy=PaperFastAuto(config());strategy.bot_start()
    StrategyResolver.validate_strategy(strategy)
    StrategyResolver.validate_strategy(PaperFastContext(config(True)))
    strategy.config["dry_run"]=False
    with pytest.raises(RuntimeError):strategy.bot_start()
    strategy=PaperFastAuto(config());strategy.config["api_server"]={"enabled":True}
    with pytest.raises(RuntimeError):strategy.bot_start()


def test_higher_timeframe_levels_are_only_visible_after_close(monkeypatch):
    from user_data.strategies import paper_fast_reaction as module
    monkeypatch.setattr(module,"add_volume_profile",lambda frame:frame.assign(vp_poc=frame["close"],vp_hvn_above=110.,vp_hvn_below=90.,vp_lvn_above=115.,vp_lvn_below=85.))
    strategy=PaperFastAuto(config())
    dates=pd.date_range("2026-09-28",periods=432,freq="5min",tz="UTC")
    base=pd.DataFrame({"date":dates,"open":100.,"high":101.,"low":99.,"close":100.,"volume":100.})
    def source(pair,timeframe):
        if timeframe=="5m":return base.copy()
        times=pd.date_range("2026-09-20",end="2026-09-29 06:00",freq={"15m":"15min","1h":"1h","4h":"4h","1d":"1D"}[timeframe],tz="UTC")
        result=pd.DataFrame({"date":times,"open":100.,"high":101.,"low":99.,"close":100.,"volume":100.})
        result.loc[result["date"]>=pd.Timestamp("2026-09-29 04:00",tz="UTC"),["high","close"]]=[200.,190.]
        return result
    strategy.dp=SimpleNamespace(current_whitelist=lambda:["SOL/USDT:USDT"],get_pair_dataframe=source)
    result=strategy.populate_indicators(base,{"pair":"SOL/USDT:USDT"})
    # The 04:00-08:00 4h candle cannot affect a decision before 08:00.
    assert result.loc[result["date"]<pd.Timestamp("2026-09-29 07:55",tz="UTC"),"vp_poc_4h"].dropna().eq(100.).all()
    assert result.loc[result["date"]>=pd.Timestamp("2026-09-29 07:55",tz="UTC"),"vp_poc_4h"].dropna().eq(190.).all()
