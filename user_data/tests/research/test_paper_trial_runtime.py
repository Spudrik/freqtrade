"""Recovery guards; no real trading process is launched or stopped by these tests."""
from contextlib import closing
from datetime import datetime, timedelta, timezone
import hashlib
import json
import sqlite3
from types import SimpleNamespace

import pytest

from user_data.Custom_Launcher.research import paper_trial_runtime as runtime


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "ROOT", tmp_path)
    report = tmp_path / "user_data/research_news_data/context_features/integrated_paper_20260926"
    report.mkdir(parents=True)
    monkeypatch.setattr(runtime, "REPORT", report)
    monkeypatch.setattr(runtime, "RECORD", report / "run_record.json")
    python = tmp_path / ".venv/Scripts/python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"test-only; not executable")
    base = {"dry_run": True, "trading_mode": "futures", "margin_mode": "isolated",
            "exchange": {"name": "binance", "key": "", "secret": "",
                         "pair_whitelist": sorted(runtime.ALLOWED_PAIRS)}}
    base_path = tmp_path / runtime.BASE
    base_path.parent.mkdir(parents=True)
    base_path.write_text(json.dumps(base))
    record = {"trial": "integrated_paper_20260926", "working_directory": str(tmp_path),
              "python_exe": str(python), "process_recovery": {"enabled": True,
              "allowed_accounts": [s.key for s in runtime.ACCOUNTS], "paused_accounts": [], "config_sha256": {}}}
    for spec in runtime.ACCOUNTS:
        path = tmp_path / spec.config
        path.parent.mkdir(parents=True, exist_ok=True)
        config = {"strategy": spec.strategy, "bot_name": spec.bot_name,
                  "db_url": f"sqlite:///user_data/research_news_data/context_features/integrated_paper_20260926/{spec.key}_trades.sqlite",
                  "force_entry_enable": spec.port is not None,
                  "api_server": {"enabled": spec.port is not None, "listen_ip_address": "127.0.0.1", "listen_port": spec.port}}
        path.write_text(json.dumps(config))
        row = record
        for key in spec.record_keys:
            row = row.setdefault(key, {})
        row.update(config=spec.config, strategy=spec.strategy, db=f"{spec.key}_trades.sqlite", log=f"{spec.key}.log", parent_pid=99, worker_pid=98)
        with closing(sqlite3.connect(report / f"{spec.key}_trades.sqlite")) as connection, connection:
            connection.executescript("CREATE TABLE trades (id INTEGER, is_open INTEGER); CREATE TABLE trade_custom_data (ft_trade_id INTEGER, cd_key TEXT, cd_value TEXT);")
    for filename in (runtime.BASE, *(s.config for s in runtime.ACCOUNTS)):
        record["process_recovery"]["config_sha256"][filename] = hashlib.sha256((tmp_path / filename).read_bytes()).hexdigest()
    runtime.RECORD.write_text(json.dumps(record))
    monkeypatch.setattr(runtime, "process_inventory", lambda: [])
    return record


def process(spec, pid=100, parent=42, command=None):
    return {"pid": pid, "ppid": parent, "cmdline": command or runtime.command_for(spec),
            "cwd": str(runtime.ROOT), "create_time": 1700000000.}


def modify_config(record, spec, changes):
    path = runtime.ROOT / spec.config
    config = json.loads(path.read_text())
    config.update(changes)
    path.write_text(json.dumps(config))
    record["process_recovery"]["config_sha256"][spec.config] = hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("spec", runtime.ACCOUNTS, ids=lambda s: s.key)
def test_all_twelve_exact_configurations_are_accepted(setup, spec):
    assert runtime.validate_config(setup, spec)["dry_run"] is True


@pytest.mark.parametrize("changes", [{"dry_run": False}, {"trading_mode": "spot"},
    {"exchange": {"key": "forbidden"}}, {"db_url": "sqlite:///elsewhere.sqlite"},
    {"strategy": "OtherStrategy"}, {"api_server": {"enabled": True, "listen_ip_address": "0.0.0.0", "listen_port": 8097}}])
def test_live_or_drifted_settings_refused_even_with_updated_hash(setup, changes):
    spec = runtime.ACCOUNTS[6]
    modify_config(setup, spec, changes)
    with pytest.raises((RuntimeError, KeyError)):
        runtime.validate_config(setup, spec)


def test_config_mutation_and_missing_database_block_reset(setup):
    spec = runtime.ACCOUNTS[0]
    path = runtime.ROOT / spec.config
    path.write_text(path.read_text() + "\n")
    with pytest.raises(RuntimeError, match="config changed"):
        runtime.validate_config(setup, spec)
    setup["process_recovery"]["config_sha256"][spec.config] = hashlib.sha256(path.read_bytes()).hexdigest()
    (runtime.REPORT / f"{spec.key}_trades.sqlite").unlink()
    with pytest.raises(RuntimeError, match="silent new balance"):
        runtime.validate_config(setup, spec)


def test_news_position_requires_persisted_explicit_protection(setup):
    spec = runtime.ACCOUNTS[6]
    path = runtime.REPORT / f"{spec.key}_trades.sqlite"
    with closing(sqlite3.connect(path)) as con, con:
        con.execute("INSERT INTO trades VALUES (1,1)")
    with pytest.raises(RuntimeError, match="persisted protection"):
        runtime.validate_config(setup, spec)
    now = datetime.now(timezone.utc)
    plan = {"side": "short", "pair": "BTC/USDT:USDT", "reference_rate": 100.,
            "stake_pct": .2, "leverage": 3., "stop_price": 102., "take_profit_price": 98.,
            "valid_until_utc": (now-timedelta(hours=1)).isoformat(),
            "review_due_at_utc": (now+timedelta(hours=3)).isoformat()}
    with closing(sqlite3.connect(path)) as con, con:
        con.execute("INSERT INTO trade_custom_data VALUES (1,'paper_news_manual_plan',?)", (json.dumps(plan),))
    # Entry approval expiry must NOT erase protection on an existing position.
    assert runtime.validate_config(setup, spec)["strategy"] == "PaperNewsManual"


def test_venv_parent_child_are_one_job_not_two(setup):
    spec = runtime.ACCOUNTS[0]
    parent = process(spec)
    child = process(spec, pid=101, parent=100)
    child["cmdline"] = [r"C:\Python3-11\python.exe", *child["cmdline"][1:]]
    assert runtime.matching_processes(spec, [parent, child]) == (parent, child)
    other = process(runtime.ACCOUNTS[1], pid=200)
    assert runtime.matching_processes(spec, [other]) is None


def test_duplicate_or_conflicting_job_refused(setup):
    spec = runtime.ACCOUNTS[0]
    with pytest.raises(RuntimeError, match="multiple process"):
        runtime.matching_processes(spec, [process(spec), process(spec, pid=200)])
    wrong = runtime.command_for(spec)
    wrong[wrong.index("--strategy")+1] = "DifferentStrategy"
    with pytest.raises(RuntimeError, match="conflicting process"):
        runtime.matching_processes(spec, [process(spec, command=wrong)])


def test_check_mode_does_not_launch_or_write(setup, monkeypatch):
    before = runtime.RECORD.read_bytes()
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *a, **k: pytest.fail("Unexpected launch"))
    rows = runtime.run_check(setup, apply=False, accounts=["auto"])
    assert rows[0]["status"] == "missing_would_start"
    assert runtime.RECORD.read_bytes() == before


def test_apply_adopts_exact_existing_worker_without_launch(setup, monkeypatch):
    spec = runtime.ACCOUNTS[0]
    monkeypatch.setattr(runtime, "process_inventory", lambda: [process(spec), process(spec, pid=101, parent=100)])
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *a, **k: pytest.fail("Duplicate launch"))
    assert runtime.run_check(setup, apply=True, accounts=["auto"])[0]["status"] == "already_running"
    assert runtime.account_record(setup, spec)["worker_pid"] == 101


def test_paused_unapproved_and_environment_override_refused(setup, monkeypatch):
    setup["process_recovery"]["paused_accounts"] = ["auto"]
    assert runtime.run_check(setup, apply=True, accounts=["auto"])[0]["status"] == "paused_do_not_restart"
    monkeypatch.setenv("FREQTRADE__DRY_RUN", "false")
    with pytest.raises(RuntimeError, match="environment overrides"):
        runtime.run_check(setup, apply=True, accounts=["manual"])
    monkeypatch.delenv("FREQTRADE__DRY_RUN")
    setup["process_recovery"]["allowed_accounts"].append("retired_A")
    with pytest.raises(RuntimeError, match="fixed account set"):
        runtime.run_check(setup, apply=True, accounts=["auto"])


def test_failed_launch_or_uncertain_order_cannot_be_retried(setup, monkeypatch):
    spec = runtime.ACCOUNTS[0]
    runtime.account_record(setup, spec)["last_recovery"] = {"status": "failed"}
    with pytest.raises(RuntimeError, match="no automatic retry"):
        runtime.start_missing(setup, spec)
    spec = runtime.ACCOUNTS[6]
    (runtime.REPORT / "news_manual_decisions.jsonl").write_text(json.dumps({"decision_id": "unknown", "status": "uncertain_manual_check_required"})+"\n")
    with pytest.raises(RuntimeError, match="unresolved manual API"):
        runtime.start_missing(setup, spec)


def test_resource_reserve_defers_start(setup, monkeypatch):
    monkeypatch.setattr(runtime.psutil, "cpu_count", lambda: 20)
    monkeypatch.setattr(runtime.psutil, "cpu_percent", lambda interval: 85.)
    monkeypatch.setattr(runtime.subprocess, "Popen", lambda *a, **k: pytest.fail("Resource-unsafe launch"))
    assert runtime.start_missing(setup, runtime.ACCOUNTS[0])["status"] == "deferred_resource_reserve"


def test_new_launch_uses_fixed_args_threads_and_fresh_running_marker(setup, monkeypatch):
    spec = runtime.ACCOUNTS[0]
    calls = []
    monkeypatch.setattr(runtime.psutil, "cpu_count", lambda: 20)
    monkeypatch.setattr(runtime.psutil, "cpu_percent", lambda interval: 10.)
    def launch(command, **kwargs):
        calls.append((command, kwargs))
        (runtime.REPORT / "auto.log").write_text("Changing state to: RUNNING\n")
        return SimpleNamespace(pid=100, poll=lambda: None)
    monkeypatch.setattr(runtime.subprocess, "Popen", launch)
    monkeypatch.setattr(runtime, "process_inventory", lambda: [process(spec), process(spec, pid=101, parent=100)])
    result = runtime.start_missing(setup, spec)
    assert result["status"] == "running" and result["worker_pid"] == 101
    assert calls[0][0] == runtime.command_for(spec)
    assert all(calls[0][1]["env"][key] == "1" for key in runtime.THREADS)
    assert len(calls) == 1


def test_windows_lock_refuses_a_second_recovery_without_any_launch():
    with runtime.recovery_lock():
        with pytest.raises(RuntimeError, match="owns the lock"):
            with runtime.recovery_lock():
                pytest.fail("Concurrent launch was allowed")


@pytest.mark.parametrize("accounts,reason", [(["auto"],"repair"),(["fast_auto"],"")])
def test_startup_acknowledgement_is_named_and_scoped(setup,accounts,reason):
    with pytest.raises(ValueError,match="named root-cause"):
        runtime.acknowledge_fixed_startup(setup,accounts,reason)


def test_startup_acknowledgement_never_relaunches_existing_or_traded_bot(setup,monkeypatch):
    spec=next(item for item in runtime.ACCOUNTS if item.key == "fast_auto")
    row=runtime.account_record(setup,spec)
    row["last_recovery"]={"status":"failed","error":"startup error: diagnosed"}
    monkeypatch.setattr(runtime,"process_inventory",lambda:[process(spec)])
    with pytest.raises(RuntimeError,match="Existing worker"):
        runtime.acknowledge_fixed_startup(setup,[spec.key],"named repair")
    monkeypatch.setattr(runtime,"process_inventory",lambda:[])
    with closing(sqlite3.connect(runtime.REPORT/row["db"])) as con,con:
        con.execute("INSERT INTO trades VALUES (1,0)")
    with pytest.raises(RuntimeError,match="Account has traded"):
        runtime.acknowledge_fixed_startup(setup,[spec.key],"named repair")


@pytest.mark.parametrize("key", ["fast_context", "fast_level_inverse"])
def test_fast_position_requires_valid_retained_plan(setup, key):
    spec=next(item for item in runtime.ACCOUNTS if item.key == key)
    with closing(sqlite3.connect(runtime.REPORT/f"{spec.key}_trades.sqlite")) as con,con:
        con.execute("INSERT INTO trades VALUES (1,1)")
    with pytest.raises(RuntimeError,match="persisted protection"):
        runtime.validate_config(setup,spec)
