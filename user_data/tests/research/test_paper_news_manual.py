"""Manual-only account isolation, explicit approvals and unattended protection."""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from types import SimpleNamespace

import pandas as pd
import pytest

from freqtrade.enums import RunMode
from user_data.strategies import paper_news_manual as broker
from user_data.strategies.integrated_paper_context import publish_luna_context, load_luna_context
from user_data.Custom_Launcher.research import paper_trial_control as control

NOW = datetime(2026, 9, 28, 12, tzinfo=timezone.utc)
SOURCE = "https://example.com/official-event"


def config(account="news_manual"):
    base, overlay = control._configs(account)
    merged = deepcopy(base)
    for key, value in overlay.items():
        if key == "exchange" and isinstance(value, dict):
            merged.setdefault(key, {}).update(value)
        else:
            merged[key] = value
    return {**merged, "runmode": RunMode.DRY_RUN}


def plan(side="long"):
    return {"pair": "BTC/USDT:USDT", "side": side, "reference_rate": 100.,
            "stop_price": 98. if side == "long" else 102.,
            "take_profit_price": 104. if side == "long" else 96.,
            "stake_pct": .2, "leverage": 3.,
            "valid_until_utc": (NOW+timedelta(minutes=5)).isoformat(),
            "review_due_at_utc": (NOW+timedelta(hours=4)).isoformat()}


@pytest.fixture
def journals(tmp_path, monkeypatch):
    paths = {name: (port, tmp_path/(name+".jsonl"))
             for name,(port,_) in broker.MANUAL_ACCOUNTS.items()}
    monkeypatch.setattr(broker, "MANUAL_ACCOUNTS", paths)
    monkeypatch.setattr(control, "MANUAL_ACCOUNTS", paths)
    monkeypatch.setattr(broker, "load_luna_context", lambda now: SimpleNamespace(status="observed",sources=(SOURCE,)))
    return paths


def proposal(journals, account="paper_news_manual", side="long", status="proposed"):
    row={"account": account,"decision_id":"decision1","action":"enter","status":status,
         "source":SOURCE,"reason":"Conditional news view confirmed by price; conflicting influences noted", "plan":plan(side)}
    control._record(row,account.removeprefix("paper_"))
    return row


@pytest.mark.parametrize("account", ["news_manual","news_lab"])
def test_accounts_are_dry_run_and_isolated(account):
    strategy=broker.PaperNewsManual(config(account))
    strategy.bot_start()
    for key,value in [("dry_run",False),("runmode",RunMode.LIVE),("force_entry_enable",False),("db_url","sqlite:///wrong.sqlite")]:
        changed=config(account);changed[key]=value
        with pytest.raises(RuntimeError):broker.PaperNewsManual(changed).bot_start()
    changed=config(account);changed["api_server"]=dict(changed["api_server"],listen_ip_address="0.0.0.0")
    with pytest.raises(RuntimeError):broker.PaperNewsManual(changed).bot_start()


@pytest.mark.parametrize("account", ["news_manual", "news_fast"])
def test_no_indicator_or_opinion_can_generate_entry(account):
    s=broker.PaperNewsManual(config(account))
    frame=pd.DataFrame({"enter_long":[1,1],"enter_short":[1,1],"close":[100,101]})
    result=s.populate_entry_trend(frame,{})
    assert not result.enter_long.any() and not result.enter_short.any()
    assert not s.populate_exit_trend(result,{}).exit_long.any()


def test_unapproved_replayed_and_cross_account_entries_refused(journals):
    s=broker.PaperNewsManual(config());s.wallets=SimpleNamespace(get_total_stake_amount=lambda:10000.)
    args=("BTC/USDT:USDT","market",60.,100.,"gtc",NOW,"news_manual:decision1","long")
    assert s.confirm_trade_entry(*args) is False
    proposal(journals)
    assert s.confirm_trade_entry(*args) is True
    lab=broker.PaperNewsManual(config("news_lab"));lab.wallets=s.wallets
    assert lab.confirm_trade_entry(*args) is False
    assert s.confirm_trade_entry(*args[:5],NOW+timedelta(minutes=6),*args[6:]) is False
    control._record({**proposal(journals),"status":"submitted"},"news_manual")
    assert s.confirm_trade_entry(*args) is False


def test_explicit_size_leverage_and_stop_limits(journals):
    s=broker.PaperNewsManual(config());s.wallets=SimpleNamespace(get_total_stake_amount=lambda:10000.)
    proposal(journals)
    args=("BTC/USDT:USDT",NOW,100.,2000.,10.,3000.,3.,"news_manual:decision1","long")
    assert s.custom_stake_amount(*args)==2000.
    assert s.custom_stake_amount(*args[:6],2.,*args[7:])==0.
    assert not s.confirm_trade_entry("BTC/USDT:USDT","market",61.,100.,"gtc",NOW,"news_manual:decision1","long")
    bad=plan();bad["stop_price"]=101.
    with pytest.raises(ValueError):broker.validate_manual_plan(bad,100.)
    bad=plan();bad["stop_price"]=None
    with pytest.raises(ValueError):broker.validate_manual_plan(bad,100.)
    bad=plan();bad["stop_price"]=90.
    with pytest.raises(ValueError):broker.validate_manual_plan(bad,100.)


@pytest.mark.parametrize("account", ["news_manual", "news_lab"])
def test_existing_manual_account_caps_remain_five_x(account):
    strategy=broker.PaperNewsManual(config(account))
    strategy.bot_start()
    old_plan=plan()
    old_plan.update(stake_pct=.25,leverage=5.,stop_price=98.)
    broker.validate_manual_plan(old_plan,100.,f"paper_{account}")
    old_plan["leverage"]=5.01
    with pytest.raises(ValueError,match="risk envelope"):
        broker.validate_manual_plan(old_plan,100.,f"paper_{account}")
    changed=config(account)
    changed["paper_manual_limits"]["max_leverage"]=10
    with pytest.raises(RuntimeError,match="risk envelope"):
        broker.PaperNewsManual(changed).bot_start()


def test_news_fast_ten_x_is_account_scoped_and_stop_stays_inside_margin_floor():
    strategy=broker.PaperNewsManual(config("news_fast"))
    strategy.bot_start()
    for side, stop in (("long",98.),("short",102.)):
        fast_plan=plan(side)
        fast_plan.update(stake_pct=.15,leverage=10.,stop_price=stop)
        broker.validate_manual_plan(fast_plan,100.,"paper_news_fast")
    too_wide=plan("long")
    too_wide.update(stake_pct=.15,leverage=10.,stop_price=97.4)
    with pytest.raises(ValueError,match="25% initial-margin"):
        broker.validate_manual_plan(too_wide,100.,"paper_news_fast")
    oversized_existing_position=plan("long")
    oversized_existing_position.update(stake_pct=.16,leverage=10.,stop_price=98.)
    # Existing-position protection remains valid if losses make its stake
    # exceed the account's fresh-entry percentage cap.
    broker.validate_manual_plan(oversized_existing_position,100.,"paper_news_fast")
    old_account_plan=plan("long")
    old_account_plan.update(stake_pct=.15,leverage=10.,stop_price=98.)
    with pytest.raises(ValueError,match="risk envelope"):
        broker.validate_manual_plan(old_account_plan,100.,"paper_news_manual")


@pytest.mark.parametrize(("account", "stake_pct"), [
    ("news_manual", .26), ("news_lab", .26), ("news_fast", .16),
])
def test_fresh_oversize_manual_entries_remain_refused(journals, account, stake_pct):
    account_name = f"paper_{account}"
    row = proposal(journals, account=account_name)
    oversized_plan = {**row["plan"], "stake_pct": stake_pct}
    control._record({**row, "plan": oversized_plan}, account)

    strategy = broker.PaperNewsManual(config(account))
    strategy.wallets = SimpleNamespace(get_total_stake_amount=lambda: 10000.)
    tag = "news_manual:decision1"
    stake_args = ("BTC/USDT:USDT", NOW, 100., 10000., 1., 10000., 3., tag, "long")
    entry_args = ("BTC/USDT:USDT", "market", 30., 100., "gtc", NOW, tag, "long")
    assert strategy.custom_stake_amount(*stake_args) == 0.
    assert strategy.confirm_trade_entry(*entry_args) is False


@pytest.mark.parametrize(("account", "stake_pct", "leverage"), [
    ("news_manual", .26, 5.), ("news_lab", .26, 5.), ("news_fast", .16, 10.),
])
def test_controller_rejects_oversize_fresh_news_entry(journals, monkeypatch,
                                                       account, stake_pct, leverage):
    monkeypatch.setattr(control, "_assert_manual_lifecycle_allows", lambda *args: None)
    monkeypatch.setattr(control, "_verify_running_account", lambda overlay: {})
    monkeypatch.setattr(control, "load_luna_context",
                        lambda now: SimpleNamespace(status="observed", sources=(SOURCE,)))
    monkeypatch.setattr(control, "_api", lambda overlay, route, payload=None:
                        [] if route == "status" else {"total_bot": 10000.})
    args = ["enter", "--account", account, "--decision-id", "too-large",
            "--pair", "BTC/USDT:USDT", "--side", "long", "--stake-pct", str(stake_pct),
            "--leverage", str(leverage), "--reference-rate", "100", "--stop-price", "98",
            "--reason", "Sourced thesis", "--source", SOURCE]

    with pytest.raises(ValueError, match="explicit paper envelope"):
        control.main(args)
    assert not journals[f"paper_{account}"][1].exists()


def test_news_fast_full_size_position_can_tighten_protection_after_equity_loss(journals, monkeypatch):
    monkeypatch.setattr(control, "_verify_running_account", lambda overlay: {})
    monkeypatch.setattr(control, "load_luna_context",
                        lambda now: SimpleNamespace(status="observed", sources=(SOURCE,)))
    trade = {"trade_id": 17, "pair": "BTC/USDT:USDT", "is_short": False,
             "stake_amount": 1500., "leverage": 10., "stop_loss_abs": 98.}
    monkeypatch.setattr(control, "_api", lambda overlay, route, payload=None:
                        [trade] if route == "status" else {"total_bot": 9000.})
    args = ["protect", "--account", "news_fast", "--decision-id", "post-loss-protect",
            "--trade-id", "17", "--reference-rate", "100", "--stop-price", "99",
            "--take-profit-price", "105", "--reason", "Tighten existing protection",
            "--source", SOURCE]

    assert control.main(args) == 0
    recorded = broker.decision_rows(journals["paper_news_fast"][1])[-1]
    assert recorded["status"] == "recorded_protection"
    assert recorded["plan"]["stake_pct"] == pytest.approx(1500. / 9000.)
    assert recorded["plan"]["stop_price"] == 99.


def test_news_fast_uses_its_own_decision_journal_and_three_pair_whitelist(journals):
    assert control._journal("news_fast")==journals["paper_news_fast"][1]
    assert control._journal("news_fast")!=control._journal("news_manual")
    assert broker.MANUAL_ACCOUNT_PAIRS["paper_news_fast"]==broker.SIEVE_PAIRS
    assert set(control._configs("news_fast")[1]["exchange"]["pair_whitelist"])==broker.SIEVE_PAIRS


def test_news_fast_controller_refuses_pairs_outside_its_effective_whitelist(journals,monkeypatch):
    monkeypatch.setattr(control,"_assert_manual_lifecycle_allows",lambda *args:None)
    monkeypatch.setattr(control,"_verify_running_account",lambda overlay:{})
    monkeypatch.setattr(control,"load_luna_context",lambda now:SimpleNamespace(status="observed",sources=(SOURCE,)))
    monkeypatch.setattr(control,"_api",lambda overlay,route,payload=None:
        [] if route=="status" else {"total_bot":10000.})
    args=["enter","--account","news_fast","--decision-id","bad-pair","--pair","BNB/USDT:USDT",
        "--side","long","--stake-pct",".1","--leverage","10","--reference-rate","100",
        "--stop-price","98","--reason","Best guess","--source",SOURCE]
    with pytest.raises(ValueError,match="explicit paper envelope"):
        control.main(args)
    assert not journals["paper_news_fast"][1].exists()


def test_news_fast_lifecycle_is_checked_against_its_own_registry_key(tmp_path,monkeypatch):
    record={"process_recovery":{"allowed_accounts":["manual","news_fast"],
        "account_lifecycle":{"manual":"DRAINING","news_fast":"ACTIVE"}}}
    path=tmp_path/"run_record.json"
    path.write_text(json.dumps(record))
    monkeypatch.setattr(control,"RUN_RECORD",path)
    control._assert_manual_lifecycle_allows("news_fast","enter")
    with pytest.raises(RuntimeError,match="manual enter is disabled"):
        control._assert_manual_lifecycle_allows("manual","enter")
    record["process_recovery"]["account_lifecycle"]["news_fast"]="PARKED"
    path.write_text(json.dumps(record))
    with pytest.raises(RuntimeError,match="news_fast enter is disabled"):
        control._assert_manual_lifecycle_allows("news_fast","enter")


@pytest.mark.parametrize("side",["long","short"])
def test_protection_and_fill_survive_restart(side,journals):
    s=broker.PaperNewsManual(config());proposal(journals,side=side)
    store={}
    trade=SimpleNamespace(id=7,pair="BTC/USDT:USDT",enter_tag="news_manual:decision1",entry_side="buy" if side=="long" else "sell",is_short=side=="short",leverage=3.,get_custom_data=lambda key:deepcopy(store.get(key)),set_custom_data=lambda key,value:store.update({key:deepcopy(value)}))
    s.order_filled(trade.pair,trade,SimpleNamespace(ft_order_side=trade.entry_side),NOW)
    restarted=broker.PaperNewsManual(config())
    trade.leverage=5.
    with pytest.raises(ValueError,match="actual position leverage"):
        restarted._trade_plan(trade)
    trade.leverage=3.
    assert restarted.custom_stoploss(trade.pair,trade,NOW,100.,0.,False)==pytest.approx(.06)
    stop=98. if side=="long" else 102.
    target=104. if side=="long" else 96.
    assert restarted.custom_exit(trade.pair,trade,NOW,stop,0.)=="manual_news_invalidation"
    assert restarted.custom_exit(trade.pair,trade,NOW,target,0.)=="manual_news_approved_target"
    assert restarted.custom_exit(trade.pair,trade,NOW,100.,0.) is None


def test_atomic_luna_publication_and_invalid_input_preserve_last_good_file(tmp_path):
    path=tmp_path/"luna.json"
    row={"schema_version":1,"observed_at_utc":NOW.isoformat(),"valid_until_utc":(NOW+timedelta(hours=4)).isoformat(),"risk_bias":"mixed","event_scale":"major","attention":"elevated","event_id":"event1","sources":[SOURCE],"brief":{"crypto_view":"mixed","uncertainties":["Consensus not available"]},"watch_proposals":[]}
    publish_luna_context(row,path,NOW)
    assert load_luna_context(NOW,path).brief["crypto_view"]=="mixed"
    previous=path.read_text()
    invalid={**row,"sources":["not-a-source"]}
    with pytest.raises(ValueError):publish_luna_context(invalid,path,NOW)
    assert path.read_text()==previous


def test_uncertain_api_write_not_retried_and_decision_blocks_reuse(journals,monkeypatch):
    monkeypatch.setattr(control,"load_luna_context",lambda now:SimpleNamespace(status="observed",sources=(SOURCE,)))
    writes=[]
    def api(overlay,route,payload=None):
        if route=="show_config":return {"dry_run":True,"strategy":"PaperNewsManual","bot_name":"paper_news_lab","trading_mode":"futures","exchange":"binance"}
        if route=="status":return []
        if route=="balance":return {"total_bot":10000.}
        writes.append((route,payload));raise RuntimeError("Uncertain response")
    monkeypatch.setattr(control,"_api",api)
    args=["enter","--account","news_lab","--decision-id","uncertain1","--pair","BTC/USDT:USDT","--side","short","--stake-pct",".2","--leverage","3","--reference-rate","100","--stop-price","102","--take-profit-price","96","--reason","Sourced thesis","--source",SOURCE]
    with pytest.raises(RuntimeError,match="Uncertain"):control.main(args)
    assert len(writes)==1
    with pytest.raises(ValueError,match="already recorded"):control.main(args)
    assert len(writes)==1
    rows=broker.decision_rows(journals["paper_news_lab"][1])
    assert rows[-1]["status"]=="uncertain_manual_check_required"
