"""Read-only briefing guards with mocked market data, never live orders."""
from datetime import datetime, timedelta, timezone
import io
import json

import pytest

from user_data.Custom_Launcher.research import paper_trial_snapshot as snapshot


def _write_article_source(root, key, folder, articles):
    source_dir = root/"user_data"/folder
    source_dir.mkdir(parents=True, exist_ok=True)
    db_path = source_dir/"articles.sqlite"
    with snapshot.sqlite3.connect(db_path) as con:
        con.execute("CREATE TABLE articles (title TEXT, canonical_url TEXT, source_url TEXT, published_at TEXT, collected_at TEXT, source_group TEXT)")
        con.executemany("INSERT INTO articles VALUES (?, ?, ?, ?, ?, ?)", articles)
    (source_dir/"collector_status.json").write_text(json.dumps({"status":"running","db_path":str(db_path)}),encoding="utf-8")


def test_source_snapshot_uses_web_cadence_and_keeps_publication_age_separate(monkeypatch,tmp_path):
    now=datetime(2026,10,5,12,tzinfo=timezone.utc)
    config_dir=tmp_path/"user_data"/"Custom_Launcher"/"research"/"config"
    config_dir.mkdir(parents=True)
    (config_dir/"web_research_sources.json").write_text(json.dumps({"poll_interval_seconds":21600}),encoding="utf-8")
    five_hours_ago=(now-timedelta(hours=5)).isoformat()
    web_articles=[("web collected 5h ago","https://web/5h",None,
        (now-timedelta(days=2)).isoformat(),five_hours_ago,"web")]
    for index in range(15):
        published="2030-01-01T00:00:00+00:00" if index==0 else "invalid-time" if index==1 else (now-timedelta(hours=1)).isoformat()
        web_articles.append((f"web recent {index}",f"https://web/recent/{index}",None,published,
            (now-timedelta(minutes=index+1)).isoformat(),"web"))
    for minutes in (20,30):
        web_articles.append((f"web older but in cadence {minutes}",f"https://web/older/{minutes}",None,
            (now-timedelta(hours=5,minutes=minutes)).isoformat(),
            (now-timedelta(hours=5,minutes=minutes)).isoformat(),"web"))
    web_articles.append(("web beyond cadence","https://web/7h",None,
        (now-timedelta(hours=7)).isoformat(),(now-timedelta(hours=7)).isoformat(),"web"))
    _write_article_source(tmp_path,"web","research_news_data/web",web_articles)
    _write_article_source(tmp_path,"news","research_news_data/news",[("news collected 5h ago","https://news/5h",None,
        (now-timedelta(hours=5)).isoformat(),five_hours_ago,"news")])
    monkeypatch.setattr(snapshot,"ROOT",tmp_path)
    monkeypatch.setattr(snapshot,"_recent_pressure",lambda *_:(None,0))

    result=snapshot.source_snapshot(now)
    headlines=result["recent_unique_headlines"]
    titles={row["title"]:row for row in headlines}
    assert result["web"]["headline_window_seconds"]==21600
    assert len(headlines)==16  # Per-collector SELECT LIMIT 16 remains in force.
    assert "web collected 5h ago" in titles
    assert "web beyond cadence" not in titles
    assert "news collected 5h ago" not in titles
    five_hour_story=titles["web collected 5h ago"]
    assert five_hour_story["published_at"]==(now-timedelta(days=2)).isoformat()
    assert five_hour_story["collected_at"]==five_hours_ago
    assert five_hour_story["published_age_hours"]==48.0
    assert five_hour_story["collected_age_hours"]==5.0
    assert five_hour_story["published_age_status"]==five_hour_story["collected_age_status"]=="known"
    assert {row["published_age_status"] for row in headlines} >= {"future","invalid"}


@pytest.mark.parametrize("config,message",[
    ([],"Web research config must be an object"),
    ({"poll_interval_seconds":0},"poll_interval_seconds"),
    ({"poll_interval_seconds":"21600"},"poll_interval_seconds"),
])
def test_source_snapshot_rejects_invalid_web_cadence_config(monkeypatch,tmp_path,config,message):
    config_dir=tmp_path/"user_data"/"Custom_Launcher"/"research"/"config"
    config_dir.mkdir(parents=True)
    (config_dir/"web_research_sources.json").write_text(json.dumps(config),encoding="utf-8")
    monkeypatch.setattr(snapshot,"ROOT",tmp_path)
    with pytest.raises(ValueError,match=message):
        snapshot.source_snapshot(datetime(2026,10,5,12,tzinfo=timezone.utc))


def test_source_snapshot_requires_web_cadence_config(monkeypatch,tmp_path):
    monkeypatch.setattr(snapshot,"ROOT",tmp_path)
    with pytest.raises(FileNotFoundError):
        snapshot.source_snapshot(datetime(2026,10,5,12,tzinfo=timezone.utc))


def test_higher_snapshot_excludes_open_candles_and_bounds_requests(monkeypatch):
    now=datetime(2026,9,29,12,tzinfo=timezone.utc)
    closed=int(now.timestamp()*1000)-1
    requests=[]
    def fetch(url,timeout):
        requests.append((url,timeout))
        return io.BytesIO(json.dumps([[0,"100","110","90","105","1",closed-1000],
            [1,"105","120","95","110","2",closed],
            [2,"110","999","1","900","3",closed+1000]]).encode())
    monkeypatch.setattr(snapshot,"urlopen",fetch)
    rows=snapshot.higher_snapshot(now)
    assert len(requests)==len(rows)==4
    assert all(timeout==12 and "limit=9" in url for url,timeout in requests)
    assert all(row["close"]==110 and row["recent_high"]==120 and row["recent_low"]==90 for row in rows)
    assert {row["recent_completed_hours"] for row in rows}=={8,48}


def test_higher_snapshot_does_not_invent_missing_history(monkeypatch):
    monkeypatch.setattr(snapshot,"urlopen",lambda *a,**k:io.BytesIO(b"[]"))
    with pytest.raises(ValueError,match="Insufficient completed"):
        snapshot.higher_snapshot(datetime.now(timezone.utc))


def test_no_flags_means_no_network_or_dataset_scan(monkeypatch,capsys):
    monkeypatch.setattr("sys.argv",["paper_trial_snapshot"])
    monkeypatch.setattr(snapshot,"urlopen",lambda *a,**k:pytest.fail("Unexpected online call"))
    monkeypatch.setattr(snapshot,"source_snapshot",lambda *a:pytest.fail("Unexpected archive scan"))
    monkeypatch.setattr(snapshot,"account_snapshot",lambda *a:pytest.fail("Unexpected account access"))
    snapshot.main()
    assert set(json.loads(capsys.readouterr().out))=={"observed_at_utc"}


def test_profit_factor_statuses_are_finite_and_closed_only():
    assert snapshot._profit_factor(3,-2,2)=={"value":1.5,"status":"ok"}
    assert snapshot._profit_factor(0,0,1)=={"value":None,"status":"no_loss_trades"}
    assert snapshot._profit_factor(0,0,0)=={"value":None,"status":"no_closed_trades"}
    assert snapshot._profit_factor(None,None,1,"missing_closed_profit")["status"]=="missing_closed_profit"
    assert snapshot._profit_factor(float("inf"),0,1,"nonfinite_closed_profit")["status"]=="nonfinite_closed_profit"


def test_runtime_span_uses_valid_account_clock_and_lifecycle_endpoints(monkeypatch):
    spec=type("Spec",(),{"key":"manual"})()
    now=datetime(2026,10,5,12,tzinfo=timezone.utc)
    record={"account":{"started_at_utc":"2026-09-28T10:00:00Z","lifecycle_changed_at_utc":"2026-10-05T09:00:00Z"},
            "process_recovery":{"events":[{"account":"manual","status":"draining_paused","at_utc":"2026-10-05T09:15:00Z"}]}}
    monkeypatch.setattr(snapshot,"account_record",lambda *_:record["account"])
    monkeypatch.setattr(snapshot,"lifecycle_for",lambda *_:"DRAINING")
    span=snapshot._runtime_span(record,spec,now)
    assert span["status"]=="known"
    assert span["start_source"]=="account_started_at_utc"
    assert span["end_source"].startswith("first_recorded_draining_paused_confirmation")
    assert span["ended_at_utc"]=="2026-10-05T09:15:00+00:00"
    record["process_recovery"]["events"].append({"account":"manual","status":"draining_paused","at_utc":"2026-10-05T08:59:00Z"})
    assert snapshot._runtime_span(record,spec,now)["ended_at_utc"]=="2026-10-05T09:15:00+00:00"


@pytest.mark.parametrize("boundary",[None,"bad","2026-10-06T00:00:00Z"])
def test_draining_runtime_requires_valid_current_boundary(monkeypatch,boundary):
    now=datetime(2026,10,5,12,tzinfo=timezone.utc)
    spec=type("Spec",(),{"key":"manual"})()
    row={"started_at_utc":"2026-09-28T10:00:00Z"}
    if boundary is not None:
        row["lifecycle_changed_at_utc"]=boundary
    monkeypatch.setattr(snapshot,"account_record",lambda *_:row)
    monkeypatch.setattr(snapshot,"lifecycle_for",lambda *_:"DRAINING")
    record={"process_recovery":{"events":[{"account":"manual","status":"draining_paused","at_utc":"2026-10-05T09:00:00Z"}]}}
    assert snapshot._runtime_span(record,spec,now)["status"]=="unknown"


def test_parked_runtime_ignores_old_stop_and_uses_verified_current_phase(monkeypatch):
    now=datetime(2026,10,5,12,tzinfo=timezone.utc)
    spec=type("Spec",(),{"key":"manual"})()
    row={"started_at_utc":"2026-09-28T10:00:00Z","lifecycle_changed_at_utc":"2026-10-05T09:30:00Z",
         "last_recovery":{"status":"parked_flat","at_utc":"2026-10-05T09:30:00Z"}}
    monkeypatch.setattr(snapshot,"account_record",lambda *_:row)
    monkeypatch.setattr(snapshot,"lifecycle_for",lambda *_:"PARKED")
    record={"process_recovery":{"events":[
        {"account":"manual","status":"parked_flat","at_utc":"2026-10-05T09:00:00Z"},
        {"account":"manual","status":"parked_flat","at_utc":"2026-10-05T10:00:00Z"}]}}
    span=snapshot._runtime_span(record,spec,now)
    assert span["ended_at_utc"]=="2026-10-05T10:00:00+00:00"
    record["process_recovery"]["events"]=[]
    assert snapshot._runtime_span(record,spec,now)["end_source"]=="current_verified_parked_flat_lifecycle_change"
    row["last_recovery"]["at_utc"]="2026-10-05T09:00:00Z"
    assert snapshot._runtime_span(record,spec,now)["status"]=="unknown"


@pytest.mark.parametrize("boundary",[None,"bad","2026-10-06T00:00:00Z"])
def test_parked_runtime_requires_valid_current_boundary(monkeypatch,boundary):
    now=datetime(2026,10,5,12,tzinfo=timezone.utc)
    spec=type("Spec",(),{"key":"manual"})()
    row={"started_at_utc":"2026-09-28T10:00:00Z",
         "last_recovery":{"status":"parked_flat","at_utc":"2026-10-05T09:00:00Z"}}
    if boundary is not None:
        row["lifecycle_changed_at_utc"]=boundary
    monkeypatch.setattr(snapshot,"account_record",lambda *_:row)
    monkeypatch.setattr(snapshot,"lifecycle_for",lambda *_:"PARKED")
    record={"process_recovery":{"events":[{"account":"manual","status":"parked_flat","at_utc":"2026-10-05T10:00:00Z"}]}}
    assert snapshot._runtime_span(record,spec,now)["status"]=="unknown"


def test_runtime_span_first_recorded_allowlist_and_restart_guard(monkeypatch):
    now=datetime(2026,10,5,12,tzinfo=timezone.utc)
    def run(key, events, account=None):
        spec=type("Spec",(),{"key":key})()
        record={"row":account or {},"process_recovery":{"events":events}}
        monkeypatch.setattr(snapshot,"account_record",lambda *_:record["row"])
        monkeypatch.setattr(snapshot,"lifecycle_for",lambda *_:"ACTIVE")
        return snapshot._runtime_span(record,spec,now)
    events=[{"account":"fast_auto","status":"failed","at_utc":"2026-09-29T10:00:00Z"},
            {"account":"fast_auto","status":"running","at_utc":"2026-09-29T10:05:00Z"}]
    span=run("fast_auto",events)
    assert span["started_at_utc"]=="2026-09-29T10:05:00+00:00"
    assert span["start_source"]=="first_recorded_running_event_not_guaranteed_original_start"
    legacy_events=[{"account":"auto","status":"running","at_utc":"2026-10-04T20:57:45Z"}]
    assert run("auto",legacy_events)["status"]=="unknown"
    assert run("auto",legacy_events,{"started_at_utc":"bad","restarted_at_utc":"2026-10-04T20:57:45Z"})["status"]=="unknown"


@pytest.mark.parametrize("started",[None,"bad","2026-10-05 10:00:00","2026-10-06T00:00:00Z"])
def test_explicit_bad_start_does_not_fall_back_to_running_event(monkeypatch,started):
    now=datetime(2026,10,5,12,tzinfo=timezone.utc)
    spec=type("Spec",(),{"key":"fast_auto"})()
    row={"started_at_utc":started}
    monkeypatch.setattr(snapshot,"account_record",lambda *_:row)
    monkeypatch.setattr(snapshot,"lifecycle_for",lambda *_:"ACTIVE")
    event={"account":"fast_auto","status":"running","at_utc":"2026-10-04T20:00:00Z"}
    assert snapshot._runtime_span({"process_recovery":{"events":[event]}},spec,now)["status"]=="unknown"


@pytest.mark.parametrize("started,expected",[
    (None,"unknown"),("bad","unknown"),("2026-10-06T00:00:00Z","unknown"),
    ("2026-10-05T12:00:00Z","known")])
def test_runtime_span_unknown_missing_invalid_future_or_zero_duration(monkeypatch,started,expected):
    spec=type("Spec",(),{"key":"plain"})()
    now=datetime(2026,10,5,12,tzinfo=timezone.utc)
    monkeypatch.setattr(snapshot,"account_record",lambda *_:{"started_at_utc":started})
    monkeypatch.setattr(snapshot,"lifecycle_for",lambda *_:"ACTIVE")
    span=snapshot._runtime_span({"process_recovery":{"events":[]}},spec,now)
    assert span["status"]==expected
    if expected == "unknown":
        assert span.get("elapsed_seconds") is None
    else:
        assert span["elapsed_seconds"] == 0


def test_runtime_span_parked_endpoint_and_no_confirmation_unknown(monkeypatch):
    spec=type("Spec",(),{"key":"manual"})()
    now=datetime(2026,10,5,12,tzinfo=timezone.utc)
    row={"started_at_utc":"2026-09-28T10:00:00Z","lifecycle_changed_at_utc":"2026-10-05T09:30:00Z",
         "last_recovery":{"status":"parked_flat","at_utc":"2026-10-05T09:30:00Z"}}
    monkeypatch.setattr(snapshot,"account_record",lambda *_:row)
    monkeypatch.setattr(snapshot,"lifecycle_for",lambda *_:"PARKED")
    rec={"process_recovery":{"events":[{"account":"manual","status":"parked_flat","at_utc":"2026-10-05T09:30:00Z"}]}}
    assert snapshot._runtime_span(rec,spec,now)["end_source"]=="recorded_parked_flat_stop"
    rec["process_recovery"]["events"]=[]
    assert snapshot._runtime_span(rec,spec,now)["end_source"]=="current_verified_parked_flat_lifecycle_change"
    row["last_recovery"]={"status":"running"}
    assert snapshot._runtime_span(rec,spec,now)["status"]=="unknown"


def test_markdown_table_pools_raw_closed_totals_and_keeps_runtime_rate_unknown():
    rows=[{"account":"a","lifecycle":"ACTIVE","open_longs":1,"open_shorts":0,"closed_longs":1,"closed_shorts":0,
           "wins":1,"losses":0,"banked_pnl_usdt":3.0,"estimated_open_pnl_usdt":1.0,"trades_per_day":2.0,
           "runtime_span":{"status":"known","elapsed_seconds":86400,"start_source":"account_started_at_utc"},
           "profit_factor":{"value":None,"status":"no_loss_trades"},"closed_profit_positive_usdt":3.0,"closed_profit_negative_usdt":0.0},
          {"account":"b","lifecycle":"DRAINING","open_longs":0,"open_shorts":1,"closed_longs":0,"closed_shorts":1,
           "wins":0,"losses":1,"banked_pnl_usdt":-1.0,"estimated_open_pnl_usdt":None,"trades_per_day":None,
           "runtime_span":{"status":"unknown"},"profit_factor":{"value":0.0,"status":"ok"},
           "closed_profit_positive_usdt":0.0,"closed_profit_negative_usdt":-1.0}]
    table=snapshot.format_account_table(rows,"2026-10-05T12:00:00+00:00")
    assert "| TOTAL | — | — | 3.000 | 1/1 | 1/1 | 1/1 | 2.00 | — | — |" in table
    assert "Runtime is elapsed span" in table and "Trades/day" in table


def test_markdown_account_report_groups_and_sorts_without_mutating_rows():
    rows = [
        {"account":"active_short","lifecycle":"ACTIVE","open_longs":0,"open_shorts":0,"closed_longs":0,"closed_shorts":0,
         "wins":0,"losses":0,"banked_pnl_usdt":0.0,"estimated_open_pnl_usdt":0.0,"trades_per_day":None,
         "runtime_span":{"status":"known","elapsed_seconds":0},"profit_factor":{"status":"no_closed_trades"},
         "closed_profit_positive_usdt":0.0,"closed_profit_negative_usdt":0.0},
        {"account":"drain_long","lifecycle":"DRAINING","open_longs":0,"open_shorts":0,"closed_longs":1,"closed_shorts":0,
         "wins":1,"losses":0,"banked_pnl_usdt":3.0,"estimated_open_pnl_usdt":0.0,"trades_per_day":None,
         "runtime_span":{"status":"known","elapsed_seconds":30},"profit_factor":{"status":"no_loss_trades"},
         "closed_profit_positive_usdt":3.0,"closed_profit_negative_usdt":0.0},
        {"account":"active_tie_b","lifecycle":"ACTIVE","open_longs":0,"open_shorts":0,"closed_longs":0,"closed_shorts":1,
         "wins":0,"losses":1,"banked_pnl_usdt":-1.0,"estimated_open_pnl_usdt":0.0,"trades_per_day":None,
         "runtime_span":{"status":"known","elapsed_seconds":10},"profit_factor":{"value":0.0,"status":"ok"},
         "closed_profit_positive_usdt":0.0,"closed_profit_negative_usdt":-1.0},
        {"account":"parked_unknown","lifecycle":"PARKED","open_longs":0,"open_shorts":0,"closed_longs":0,"closed_shorts":0,
         "wins":0,"losses":0,"banked_pnl_usdt":0.0,"estimated_open_pnl_usdt":0.0,"trades_per_day":None,
         "runtime_span":{"status":"unknown"},"profit_factor":{"status":"no_closed_trades"},
         "closed_profit_positive_usdt":0.0,"closed_profit_negative_usdt":0.0},
        {"account":"active_tie_a","lifecycle":"ACTIVE","open_longs":0,"open_shorts":0,"closed_longs":0,"closed_shorts":0,
         "wins":0,"losses":0,"banked_pnl_usdt":0.0,"estimated_open_pnl_usdt":0.0,"trades_per_day":None,
         "runtime_span":{"status":"known","elapsed_seconds":10},"profit_factor":{"status":"no_closed_trades"},
         "closed_profit_positive_usdt":0.0,"closed_profit_negative_usdt":0.0},
        {"account":"odd","lifecycle":"NEW_STATE","open_longs":0,"open_shorts":0,"closed_longs":0,"closed_shorts":0,
         "wins":0,"losses":0,"banked_pnl_usdt":0.0,"estimated_open_pnl_usdt":0.0,"trades_per_day":None,
         "runtime_span":{"status":"unknown"},"profit_factor":{"status":"no_closed_trades"},
         "closed_profit_positive_usdt":0.0,"closed_profit_negative_usdt":0.0},
        {"account":"active_unknown","lifecycle":"ACTIVE","open_longs":0,"open_shorts":0,"closed_longs":0,"closed_shorts":0,
         "wins":0,"losses":0,"banked_pnl_usdt":0.0,"estimated_open_pnl_usdt":0.0,"trades_per_day":None,
         "runtime_span":{"status":"unknown"},"profit_factor":{"status":"no_closed_trades"},
         "closed_profit_positive_usdt":0.0,"closed_profit_negative_usdt":0.0},
    ]
    original = [dict(row) for row in rows]
    table = snapshot.format_account_table(rows, "2026-10-05T12:00:00+00:00")
    assert rows == original
    assert table.index("## Active") < table.index("## Draining — new entries paused") < table.index("## Parked — stopped")
    assert table.index("## Parked — stopped") < table.index("## Unknown lifecycle") < table.index("## Overall totals")
    active = table.split("## Active", 1)[1].split("## Draining", 1)[0]
    assert active.index("active_tie_a") < active.index("active_tie_b") < active.index("active_short") < active.index("active_unknown")
    assert table.count("| odd |") == table.count("| drain_long |") == table.count("| parked_unknown |") == 1
    total_row = next(line for line in table.splitlines() if line.startswith("| TOTAL |"))
    assert "3.000" in total_row and "1/1" in total_row and "2.00" in total_row


def test_profit_factor_and_pool_never_publish_overflow_or_assume_missing_is_zero():
    assert snapshot._profit_factor(1e308,-1e-308,2)=={"value":None,"status":"nonfinite_profit_factor"}
    rows=[{"profit_factor":{"status":"nonfinite_profit_factor"},"closed_longs":1,"closed_shorts":0,
           "closed_profit_positive_usdt":1e308,"closed_profit_negative_usdt":-1e308},
          {"profit_factor":{"status":"ok"},"closed_longs":1,"closed_shorts":0,
           "closed_profit_positive_usdt":0.0,"closed_profit_negative_usdt":0.0}]
    assert snapshot._pooled_factor(rows)=={"value":1.0,"status":"ok"}
    rows[1]["closed_profit_positive_usdt"]=None
    assert snapshot._pooled_factor(rows)["status"]=="incomplete_closed_profit"
    rows[1]["closed_profit_positive_usdt"]=1e308
    assert snapshot._pooled_factor(rows)["status"]=="nonfinite_closed_profit"


def test_account_table_separates_timestamp_and_labels_uninitialized_database():
    table=snapshot.format_account_table([{"account":"new_one","database_state":"not_initialized",
        "runtime_span":{"status":"unknown"},"profit_factor":{"status":"database_not_initialized"}}],
        "2026-10-05T12:00:00+00:00")
    assert "2026-10-05T12:00:00+00:00\n\n## Unknown lifecycle\n\n| Account |" in table
    assert "new_one (DB not initialized)" in table


def test_account_snapshot_passes_one_snapshot_clock_to_worker_health(monkeypatch,tmp_path):
    now=datetime(2026,10,5,12,tzinfo=timezone.utc)
    spec=type("Spec",(),{"key":"new_one","db":"new_one.sqlite","new_identity":True})()
    row={"db":"new_one.sqlite"}
    calls=[]
    monkeypatch.setattr(snapshot,"read_json",lambda *_:{})
    monkeypatch.setattr(snapshot,"account_specs_for_record",lambda *_:[spec])
    monkeypatch.setattr(snapshot,"account_record",lambda *_:row)
    monkeypatch.setattr(snapshot,"process_inventory",lambda:[])
    monkeypatch.setattr(snapshot,"matching_processes",lambda *_:None)
    monkeypatch.setattr(snapshot,"lifecycle_for",lambda *_:"ACTIVE")
    monkeypatch.setattr(snapshot,"REPORT",tmp_path)
    monkeypatch.setattr(snapshot,"_worker_log_health",lambda spec,tree,clock,lifecycle:(calls.append(clock) or {"state":"missing_process_or_log"}))
    monkeypatch.setattr(snapshot,"_runtime_span",lambda *_:{"status":"unknown"})
    snapshot.account_snapshot(now=now)
    assert calls==[now]


def test_account_table_cli_uses_one_snapshot_and_no_extra_fetch(monkeypatch,capsys):
    calls=[]
    monkeypatch.setattr("sys.argv",["paper_trial_snapshot","--account-table"])
    monkeypatch.setattr(snapshot,"account_snapshot",lambda prices,now:(calls.append((prices,now)) or []))
    snapshot.main()
    assert len(calls)==1 and calls[0][0] is False
    assert "| TOTAL |" in capsys.readouterr().out


def test_accounts_json_adds_table_without_changing_accounts_array(monkeypatch,capsys):
    rows=[{"account":"x","lifecycle":"ACTIVE","open_longs":0,"open_shorts":0,"closed_longs":0,"closed_shorts":0,
           "wins":0,"losses":0,"banked_pnl_usdt":0.0,"estimated_open_pnl_usdt":0.0,"runtime_span":{"status":"unknown"},
           "trades_per_day":None,"profit_factor":{"value":None,"status":"no_closed_trades"}}]
    monkeypatch.setattr("sys.argv",["paper_trial_snapshot","--accounts"])
    monkeypatch.setattr(snapshot,"account_snapshot",lambda prices,now:rows)
    snapshot.main()
    payload=json.loads(capsys.readouterr().out)
    assert payload["accounts"]==rows and "account_table" in payload
