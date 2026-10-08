"""Recovery guards; no real trading process is launched or stopped by these tests."""
from contextlib import closing, nullcontext
from datetime import datetime, timedelta, timezone
import hashlib
import json
import sqlite3
from types import SimpleNamespace

import pytest

from user_data.Custom_Launcher.research import paper_trial_runtime as runtime
from user_data.strategies import paper_aggressive_context as aggressive_context
from user_data.strategies.paper_aggressive_context import AGGRESSIVE_LIMITS


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
              "allowed_accounts": [s.key for s in runtime.LEGACY_ACCOUNTS], "paused_accounts": [], "config_sha256": {}}}
    for spec in runtime.ACCOUNTS:
        path = tmp_path / spec.config
        path.parent.mkdir(parents=True, exist_ok=True)
        config = {"strategy": spec.strategy, "bot_name": spec.bot_name,
                  "db_url": f"sqlite:///user_data/research_news_data/context_features/integrated_paper_20260926/{spec.key}_trades.sqlite",
                  "force_entry_enable": spec.port is not None,
                  "api_server": {"enabled": spec.port is not None, "listen_ip_address": "127.0.0.1", "listen_port": spec.port}}
        if spec.new_identity:
            config.update(max_open_trades=3, exchange={"pair_whitelist": sorted(spec.pairs)})
        if spec.key in runtime.AGGRESSIVE_ACCOUNT_KEYS:
            config.update(timeframe="5m", initial_state="running", paper_aggressive_limits=AGGRESSIVE_LIMITS)
            config["api_server"].update(username="test-user", password="test-password",
                jwt_secret_key="test-jwt", ws_token="test-ws", verbosity="error", enable_openapi=False,
                CORS_origins=[])
        if spec.bot_name in runtime.MANUAL_ACCOUNT_CAPS:
            config["paper_manual_limits"] = {
                **runtime.MANUAL_ACCOUNT_CAPS[spec.bot_name],
                "emergency_margin_loss_pct": runtime.MANUAL_EMERGENCY_MARGIN_LOSS_PCT,
            }
        path.write_text(json.dumps(config))
        row = record
        for key in spec.record_keys:
            row = row.setdefault(key, {})
        row.update(config=spec.config, strategy=spec.strategy, db=f"{spec.key}_trades.sqlite", log=f"{spec.key}.log", parent_pid=99, worker_pid=98)
        with closing(sqlite3.connect(report / f"{spec.key}_trades.sqlite")) as connection, connection:
            if spec.key in runtime.AGGRESSIVE_ACCOUNT_KEYS:
                connection.executescript("CREATE TABLE trades (id INTEGER, pair TEXT, is_short INTEGER, open_rate REAL, amount REAL, stake_amount REAL, leverage REAL, contract_size REAL, is_open INTEGER); CREATE TABLE trade_custom_data (ft_trade_id INTEGER, cd_key TEXT, cd_value TEXT);")
            elif spec.key in {"news_manual", "news_lab", "news_fast"}:
                connection.executescript("CREATE TABLE trades (id INTEGER, is_open INTEGER, leverage REAL); CREATE TABLE trade_custom_data (ft_trade_id INTEGER, cd_key TEXT, cd_value TEXT);")
            elif spec.key in {"swing_level_bounce", "swing_level_break_hold"}:
                connection.executescript("CREATE TABLE trades (id INTEGER, is_open INTEGER, leverage REAL); CREATE TABLE trade_custom_data (ft_trade_id INTEGER, cd_key TEXT, cd_value TEXT);")
            else:
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


def _registry(record, keys):
    policy = record["process_recovery"]
    policy["allowed_accounts"] = sorted(keys)
    policy["account_lifecycle"] = {
        key: "PARKED" if key in runtime.DRAINING_ACCOUNT_KEYS else "ACTIVE" for key in keys
    }
    policy["paused_accounts"] = sorted(key for key in keys if policy["account_lifecycle"][key] == "PARKED")


def test_optional_news_fast_registry_preserves_legacy16_checks_and_new17_identity(setup):
    _registry(setup, runtime.REVIEWED_16_ACCOUNT_KEYS)
    specs = runtime.account_specs_for_record(setup)
    assert len(specs) == 16 and "news_fast" not in {spec.key for spec in specs}
    rows = runtime.run_check(setup, apply=False, accounts=["auto"])
    assert rows == [{"account": "auto", "status": "paused_do_not_restart"}]
    setup["process_recovery"]["paused_accounts"] = []
    with pytest.raises(RuntimeError, match="exact PARKED lifecycle set"):
        runtime.run_check(setup, apply=False, accounts=["auto"])

    _registry(setup, runtime.ALL_ACCOUNT_KEYS)
    specs = runtime.account_specs_for_record(setup)
    assert len(specs) == 17 and next(spec for spec in specs if spec.key == "news_fast").new_identity
    inverse = next(spec for spec in specs if spec.key == "fast_level_inverse")
    assert runtime.lifecycle_for(setup, inverse) == "ACTIVE"
    assert runtime.validate_config(setup, inverse)["strategy"] == inverse.strategy


def test_attended_initialization_accepts_existing16_and_optional17_registries(setup, monkeypatch):
    _registry(setup, runtime.REVIEWED_16_ACCOUNT_KEYS)
    sieve = next(spec for spec in runtime.NEW_SIEVE_ACCOUNTS if spec.key == "sieve_pivot_partial")
    (runtime.REPORT / f"{sieve.key}_trades.sqlite").unlink()
    _registry(setup, runtime.REVIEWED_16_ACCOUNT_KEYS)
    launched = []
    monkeypatch.setattr(runtime, "_preflight_write_environment", lambda: None)
    monkeypatch.setattr(runtime, "validate_config", lambda *args: {})
    monkeypatch.setattr(runtime, "matching_processes", lambda *args: None)
    monkeypatch.setattr(runtime, "start_missing", lambda record, spec, **kwargs:
                        launched.append(spec.key) or {"account": spec.key, "status": "running"})
    assert runtime.initialize_new_accounts(setup, [sieve.key])[0]["status"] == "running"
    assert launched == [sieve.key]

    _registry(setup, runtime.ALL_ACCOUNT_KEYS)
    fast = next(spec for spec in runtime.NEW_MANUAL_ACCOUNTS if spec.key == "news_fast")
    (runtime.REPORT / f"{fast.key}_trades.sqlite").unlink()
    assert runtime.initialize_new_accounts(setup, [fast.key])[0]["account"] == "news_fast"
    assert launched == [sieve.key, fast.key]

    _registry(setup, runtime.REVIEWED_19_ACCOUNT_KEYS)
    swing_specs = runtime.NEW_SWING_ACCOUNTS
    for spec in swing_specs:
        (runtime.REPORT / f"{spec.key}_trades.sqlite").unlink()
    assert runtime.initialize_new_accounts(setup, [swing_specs[0].key, swing_specs[1].key]) == [
        {"account": swing_specs[0].key, "status": "running"},
        {"account": swing_specs[1].key, "status": "running"},
    ]
    assert launched == [sieve.key, fast.key, *(spec.key for spec in swing_specs)]


def test_explicit19_registry_preserves_17_active_inverse_and_requires_both_new_swing_accounts(setup):
    _registry(setup, runtime.ALL_ACCOUNT_KEYS)
    inverse = next(spec for spec in runtime.REVIEWED_17_ACCOUNTS if spec.key == "fast_level_inverse")
    assert runtime.lifecycle_for(setup, inverse) == "ACTIVE"
    assert len(runtime.account_specs_for_record(setup)) == 17

    _registry(setup, runtime.REVIEWED_19_ACCOUNT_KEYS)
    policy = setup["process_recovery"]
    policy["account_lifecycle"]["fast_level_inverse"] = "PARKED"
    policy["paused_accounts"] = sorted(key for key, state in policy["account_lifecycle"].items()
                                        if state == "PARKED")
    specs = runtime.account_specs_for_record(setup)
    assert len(specs) == 19
    assert runtime.lifecycle_for(setup, inverse) == "PARKED"
    for spec in runtime.NEW_SWING_ACCOUNTS:
        assert runtime.lifecycle_for(setup, spec) == "ACTIVE"
        assert runtime.validate_config(setup, spec)["strategy"] == spec.strategy

    incomplete = runtime.REVIEWED_17_ACCOUNT_KEYS | {"swing_level_bounce"}
    _registry(setup, incomplete)
    with pytest.raises(RuntimeError, match="neither exact legacy12"):
        runtime.account_specs_for_record(setup)


def test_explicit23_registry_preserves19_and_validates_all_four_aggressive_accounts(setup):
    _registry(setup, runtime.REVIEWED_19_ACCOUNT_KEYS)
    assert len(runtime.account_specs_for_record(setup)) == 19

    _registry(setup, runtime.REVIEWED_23_ACCOUNT_KEYS)
    specs = runtime.account_specs_for_record(setup)
    assert len(specs) == 23
    for spec in runtime.NEW_AGGRESSIVE_ACCOUNTS:
        assert runtime.lifecycle_for(setup, spec) == "ACTIVE"
        assert runtime.validate_config(setup, spec)["strategy"] == spec.strategy

    incomplete = runtime.REVIEWED_19_ACCOUNT_KEYS | {runtime.NEW_AGGRESSIVE_ACCOUNTS[0].key}
    _registry(setup, incomplete)
    with pytest.raises(RuntimeError, match="neither exact legacy12"):
        runtime.account_specs_for_record(setup)


def test_runtime_validates_persisted_open_aggressive_position_contract_size(setup):
    _registry(setup, runtime.REVIEWED_23_ACCOUNT_KEYS)
    spec = runtime.NEW_AGGRESSIVE_ACCOUNTS[0]
    pair = sorted(aggressive_context.AGGRESSIVE_PAIRS)[0]
    pending = {"family": aggressive_context.AGGRESSIVE_ACCOUNTS[spec.key]["family"],
        "mode": "primary", "route": "profile_acceptance", "pair": pair, "side": "long",
        "entry_tag": "aggressive:vacuum:primary:profile_acceptance:long",
        "provenance": "automatic_signal", "reason": "runtime validation test", "features": {},
        "reference_rate": 100., "entry_atr": 2., "atr15": 4., "structural_anchor": 97.,
        "stop_price": 97., "target_price": 110., "target_kind": "next_observed_4h_range",
        "risk_reserve_usdt": 180., "entry_equity_usdt": 10000., "stake_usdt": 1000.,
        "requested_leverage": 5.}
    plan = aggressive_context.build_aggressive_filled_plan(
        pending, pair=pair, side="long", open_rate=100., quantity=50., stake=1000.,
        leverage=5., contract_size=1., filled_at=datetime(2026, 10, 8, tzinfo=timezone.utc))
    db = runtime.REPORT / f"{spec.key}_trades.sqlite"
    with closing(sqlite3.connect(db)) as connection, connection:
        connection.execute("INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)",
                           (1, pair, 0, 100., 50., 1000., 5., 1.))
        connection.execute("INSERT INTO trade_custom_data VALUES (?, ?, ?)",
                           (1, aggressive_context.PLAN_KEY, json.dumps(plan)))

    assert runtime.validate_config(setup, spec)["strategy"] == spec.strategy


def test_swing_recovery_revalidates_persisted_plan_against_actual_leverage(setup):
    from user_data.strategies.paper_swing_levels import (
        FEE_AND_SLIPPAGE, PLAN_KEY, swing_stop_price, swing_target_price,
    )

    _registry(setup, runtime.REVIEWED_19_ACCOUNT_KEYS)
    spec = runtime.NEW_SWING_ACCOUNTS[0]
    stake, entry, atr, level, leverage = 1000., 101., 1., 100., 3.
    stop = swing_stop_price("long", entry, atr, level)
    target, target_kind = swing_target_price("long", entry, stop, None)
    plan = {"side": "long", "open_rate": entry, "entry_atr": atr, "stop_price": stop,
        "target_price": target, "planned_loss_usdt": stake * leverage * (abs(entry-stop)/entry + FEE_AND_SLIPPAGE),
        "entry_equity_usdt": 10000., "leverage": leverage, "trigger_level": level,
        "level_source": "4h_rolling20_low", "signal_kind": "bounce", "target_kind": target_kind,
        "stake_usdt": stake, "target_level": None, "target_source": None}
    db = runtime.REPORT / f"{spec.key}_trades.sqlite"
    encoded = json.dumps(plan)
    with closing(sqlite3.connect(db)) as connection, connection:
        connection.execute("INSERT INTO trades VALUES (1, 1, 3)")
        connection.execute("INSERT INTO trade_custom_data VALUES (1, ?, ?)", (PLAN_KEY, encoded))
    assert runtime.validate_config(setup, spec)["strategy"] == spec.strategy
    with closing(sqlite3.connect(db)) as connection, connection:
        connection.execute("UPDATE trades SET leverage=2 WHERE id=1")
    with pytest.raises(ValueError, match="differs from the persisted trade"):
        runtime.validate_config(setup, spec)


@pytest.mark.parametrize("spec", runtime.ACCOUNTS, ids=lambda s: s.key)
def test_all_registered_exact_configurations_are_accepted(setup, spec):
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
        con.execute("INSERT INTO trades(id,is_open,leverage) VALUES (1,1,3)")
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
    mismatched_plan = {**plan, "leverage": 5.}
    with closing(sqlite3.connect(path)) as con, con:
        con.execute("UPDATE trade_custom_data SET cd_value=?", (json.dumps(mismatched_plan),))
    with pytest.raises(ValueError, match="actual position leverage"):
        runtime.validate_config(setup, spec)


def test_news_fast_recovery_audits_account_scoped_ten_x_plan(setup):
    spec=next(item for item in runtime.NEW_MANUAL_ACCOUNTS if item.key=="news_fast")
    path=runtime.REPORT/f"{spec.key}_trades.sqlite"
    with closing(sqlite3.connect(path)) as con,con:
        con.execute("INSERT INTO trades(id,is_open,leverage) VALUES (1,1,10)")
    with pytest.raises(RuntimeError,match="persisted protection"):
        runtime.validate_config(setup,spec)
    now=datetime.now(timezone.utc)
    plan={"side":"short","pair":"BTC/USDT:USDT","reference_rate":100.,"stake_pct":.15,
        "leverage":10.,"stop_price":102.,"take_profit_price":97.,
        "valid_until_utc":(now+timedelta(minutes=5)).isoformat(),
        "review_due_at_utc":(now+timedelta(hours=4)).isoformat()}
    with closing(sqlite3.connect(path)) as con,con:
        con.execute("INSERT INTO trade_custom_data VALUES (1,'paper_news_manual_plan',?)",(json.dumps(plan),))
    assert runtime.validate_config(setup,spec)["strategy"]=="PaperNewsManual"
    # A 5x plan would approve a 4% stop as a 20% margin loss, while the actual
    # 10x position would lose 40%; recovery must reject that mismatch.
    mismatched_plan={**plan,"leverage":5.,"stop_price":104.}
    with closing(sqlite3.connect(path)) as con,con:
        con.execute("UPDATE trade_custom_data SET cd_value=?",(json.dumps(mismatched_plan),))
    with pytest.raises(ValueError,match="actual position leverage"):
        runtime.validate_config(setup,spec)


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
    with pytest.raises(RuntimeError, match="neither exact legacy12"):
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


def test_fast_level_inverse_is_allowed_to_drain_on_its_paused_overlay(setup):
    _registry(setup, runtime.REVIEWED_16_ACCOUNT_KEYS)
    spec = next(item for item in runtime.LEGACY_ACCOUNTS if item.key == "fast_level_inverse")
    setup["process_recovery"]["account_lifecycle"][spec.key] = "DRAINING"
    setup["process_recovery"]["paused_accounts"] = sorted(
        key for key, state in setup["process_recovery"]["account_lifecycle"].items() if state == "PARKED"
    )
    modify_config(setup, spec, {"initial_state": "paused"})
    assert runtime.account_specs_for_record(setup)
    assert runtime.lifecycle_for(setup, spec) == "DRAINING"
    assert runtime.validate_config(setup, spec)["initial_state"] == "paused"
    assert spec.key in runtime.DRAIN_RESTART_ACCOUNT_KEYS


def test_fast_level_inverse_can_be_parked_without_becoming_mandatory_legacy_drain(setup):
    _registry(setup, runtime.ALL_ACCOUNT_KEYS)
    spec = next(item for item in runtime.ACCOUNTS if item.key == "fast_level_inverse")
    setup["process_recovery"]["account_lifecycle"][spec.key] = "PARKED"
    setup["process_recovery"]["paused_accounts"] = sorted(
        key for key, state in setup["process_recovery"]["account_lifecycle"].items() if state == "PARKED"
    )
    modify_config(setup, spec, {"initial_state": "paused"})
    assert runtime.lifecycle_for(setup, spec) == "PARKED"
    assert runtime.validate_config(setup, spec)["initial_state"] == "paused"


REPAIR_REASON = "Activate reviewed partial fill and persisted stop repair"


@pytest.fixture
def active_setup(setup, monkeypatch):
    policy = setup["process_recovery"]
    policy["allowed_accounts"] = sorted(runtime.ALL_ACCOUNT_KEYS)
    policy["account_lifecycle"] = {s.key: "DRAINING" if s.key in runtime.DRAINING_ACCOUNT_KEYS else "ACTIVE"
                                   for s in runtime.REVIEWED_17_ACCOUNTS}
    for spec in runtime.REVIEWED_17_ACCOUNTS:
        if spec.key in runtime.DRAINING_ACCOUNT_KEYS:
            modify_config(setup, spec, {"initial_state": "paused"})
        if spec.new_identity:
            runtime.account_record(setup, spec)["database_initialized_at_utc"] = runtime.utc_now()
        with closing(sqlite3.connect(runtime.REPORT / f"{spec.key}_trades.sqlite")) as con, con:
            columns = ["ALTER TABLE trades ADD COLUMN pair TEXT DEFAULT 'BTC/USDT:USDT';",
                "ALTER TABLE trades ADD COLUMN is_short INTEGER DEFAULT 0;",
                "ALTER TABLE trades ADD COLUMN open_rate REAL DEFAULT 100;",
                "ALTER TABLE trades ADD COLUMN stop_loss REAL DEFAULT 95;"]
            if spec.key not in {"news_manual", "news_lab", "news_fast", "swing_level_bounce", "swing_level_break_hold"}:
                columns.append("ALTER TABLE trades ADD COLUMN leverage REAL DEFAULT 1;")
            columns.extend(["ALTER TABLE trades ADD COLUMN amount REAL DEFAULT 1;",
                "ALTER TABLE trades ADD COLUMN stake_amount REAL DEFAULT 100;",
                "CREATE TABLE orders (ft_is_open INTEGER);"])
            con.executescript("".join(columns))
    wrapper = runtime.ROOT / runtime.ACTIVE_REPAIR_STRATEGY
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_bytes(b"isolated test reviewed wrapper")
    monkeypatch.setattr(runtime, "ACTIVE_REPAIR_SHA256", hashlib.sha256(wrapper.read_bytes()).hexdigest())
    monkeypatch.setattr(runtime, "_validate_new_strategy_bootstrap", lambda *a: None)
    monkeypatch.setattr(runtime, "_resource_reserve_available", lambda: True)
    specs = [s for s in runtime.NEW_SIEVE_ACCOUNTS if s.key in runtime.ACTIVE_REPAIR_ACCOUNT_KEYS]
    trees = {s.key: (process(s, pid=500 + index, parent=42), None) for index, s in enumerate(specs)}
    for tree in trees.values():
        tree[0]["create_time"] = (datetime.now(timezone.utc) - timedelta(minutes=1)).timestamp()
    monkeypatch.setattr(runtime, "process_inventory", lambda: [item for tree in trees.values() for item in tree if item])
    def heartbeat(spec, tree, now=None):
        return {"state": "RUNNING", "fresh": True, "at_utc": runtime.utc_now(), "pid": tree[0]["pid"]} if tree else {
            "state": None, "fresh": False, "at_utc": None}
    monkeypatch.setattr(runtime, "worker_heartbeat_state", heartbeat)
    stopped, started = [], []
    def stop(spec, tree):
        assert trees[spec.key] == tree
        stopped.append(spec.key)
        trees.pop(spec.key)
    def start(record, spec, **kwargs):
        assert spec.key not in trees
        started.append(spec.key)
        parent = process(spec, pid=700 + len(started), parent=42)
        parent["create_time"] = datetime.now(timezone.utc).timestamp()
        trees[spec.key] = (parent, None)
        row = runtime.account_record(record, spec)
        runtime.note_running(row, trees[spec.key])
        row["restarted_at_utc"] = runtime.utc_now()
        row["last_recovery"] = {"status": "running", "at_utc": row["restarted_at_utc"]}
        return {"account": spec.key, "status": "running", "parent_pid": parent["pid"], "worker_pid": None}
    monkeypatch.setattr(runtime, "_stop_exact_tree", stop)
    real_start = runtime.start_missing
    monkeypatch.setattr(runtime, "start_missing", start)
    runtime.save_record(setup)
    return SimpleNamespace(record=setup, specs=specs, trees=trees, stopped=stopped, started=started,
                           start=start, real_start=real_start)


def focused_state():
    return {"version": 1, "contract": "target_partial_invalidation_remainder", "plan": "touch_partial",
        "side": "long", "entry_rate": 100, "entry_filled_at": "2026-10-01T12:00:00+00:00",
        "entry_snapshot_candle": "2026-10-01T11:00:00+00:00", "levels": {"target_1": 110, "invalidation": 90},
        "initial_stake": 100, "stages": {"stage_1": {"fraction": .1, "tag": "focused_partial_stage_1",
        "status": "pending", "credited_stake": 0}}, "phase": "stage_1_pending"}


def add_position(spec, trade_id=1, state=None):
    with closing(sqlite3.connect(runtime.REPORT / f"{spec.key}_trades.sqlite")) as con, con:
        con.execute("INSERT INTO trades(id,is_open) VALUES (?,1)", (trade_id,))
        con.execute("INSERT INTO trade_custom_data VALUES (?, 'sieve3_v2_focused:test', ?)",
                    (trade_id, json.dumps(state or focused_state())))


def test_active_restart_dry_preflight_has_no_writes_or_actions(active_setup):
    fixture = active_setup
    before = runtime.RECORD.read_bytes()
    keys = [spec.key for spec in fixture.specs]
    rows = runtime.restart_active(fixture.record, keys, apply=False, repair_reason=REPAIR_REASON)
    assert [row["status"] for row in rows] == ["would_restart_active"] * 3
    assert runtime.RECORD.read_bytes() == before
    assert fixture.stopped == fixture.started == []
    assert "events" not in fixture.record["process_recovery"]


def test_active_restart_success_preserves_db_and_immutable_state(active_setup, monkeypatch):
    fixture = active_setup
    spec = fixture.specs[0]
    add_position(spec)
    original_start = fixture.start
    def reconciled_start(*args, **kwargs):
        result = original_start(*args, **kwargs)
        state = focused_state()
        state["stages"]["stage_1"].update(status="filled", credited_stake=10)
        state["phase"] = "remainder"
        with closing(sqlite3.connect(runtime.REPORT / f"{spec.key}_trades.sqlite")) as con, con:
            con.execute("UPDATE trade_custom_data SET cd_value=?", (json.dumps(state),))
            con.execute("UPDATE trades SET stop_loss=98")
        return result
    monkeypatch.setattr(runtime, "start_missing", reconciled_start)
    rows = runtime.restart_active(fixture.record, [spec.key], apply=True, repair_reason=REPAIR_REASON)
    assert rows[0]["status"] == "active_restart_completed"
    assert rows[0]["reviewed_strategy_sha256"] == runtime.ACTIVE_REPAIR_SHA256
    assert rows[0]["repair_reason"] == REPAIR_REASON
    assert rows[0]["protection_gap_to_heartbeat_seconds"] >= 0
    assert rows[0]["open_trade_ids_before"] == rows[0]["open_trade_ids_after"] == [1]
    assert fixture.record["process_recovery"]["account_lifecycle"][spec.key] == "ACTIVE"
    assert len(fixture.record["process_recovery"]["events"]) == 2
    assert fixture.stopped == fixture.started == [spec.key]


@pytest.mark.parametrize("block", ["order", "protection", "stale", "duplicate", "capacity", "config",
                                  "hash", "bootstrap", "initialization", "missing_db", "pending"])
def test_active_restart_any_target_preflight_blocks_entire_batch(active_setup, monkeypatch, block):
    fixture = active_setup
    spec = fixture.specs[-1]
    db = runtime.REPORT / f"{spec.key}_trades.sqlite"
    if block == "order":
        with closing(sqlite3.connect(db)) as con, con:
            con.execute("INSERT INTO orders VALUES (1)")
    elif block == "protection":
        add_position(spec)
        with closing(sqlite3.connect(db)) as con, con:
            con.execute("UPDATE trades SET stop_loss=NULL")
    elif block == "stale":
        monkeypatch.setattr(runtime, "worker_heartbeat_state", lambda *a: {"state": "RUNNING", "fresh": False})
    elif block == "duplicate":
        inventory = runtime.process_inventory()
        monkeypatch.setattr(runtime, "process_inventory", lambda: inventory + [process(spec, pid=999)])
    elif block == "capacity":
        monkeypatch.setattr(runtime, "_resource_reserve_available", lambda: False)
    elif block == "config":
        (runtime.ROOT / spec.config).write_text("{}")
    elif block == "hash":
        (runtime.ROOT / runtime.ACTIVE_REPAIR_STRATEGY).write_bytes(b"unapproved")
    elif block == "bootstrap":
        def broken(*args):
            raise RuntimeError("invalid locked parameters")
        monkeypatch.setattr(runtime, "_validate_new_strategy_bootstrap", broken)
    elif block == "initialization":
        runtime.account_record(fixture.record, spec).pop("database_initialized_at_utc")
    elif block == "missing_db":
        db.unlink()
    elif block == "pending":
        runtime.account_record(fixture.record, spec)["last_recovery"] = {"status": "failed"}
    before = runtime.RECORD.read_bytes()
    rows = runtime.restart_active(fixture.record, [s.key for s in fixture.specs], apply=True, repair_reason=REPAIR_REASON)
    assert any(row["status"] == "blocked_active_restart_preflight" for row in rows)
    assert fixture.stopped == fixture.started == []
    assert runtime.RECORD.read_bytes() == before


@pytest.mark.parametrize("kind", ["raise", "starting"])
def test_active_restart_failed_launch_stops_batch_and_fences_retry(active_setup, monkeypatch, kind):
    fixture = active_setup
    def fail(*args, **kwargs):
        if kind == "raise":
            raise RuntimeError("diagnosed launch failure")
        return {"status": "starting"}
    monkeypatch.setattr(runtime, "start_missing", fail)
    rows = runtime.restart_active(fixture.record, [s.key for s in fixture.specs], apply=True, repair_reason=REPAIR_REASON)
    assert rows[0]["status"] == "active_restart_pending"
    assert len(rows) == len(fixture.stopped) == 1
    persisted = json.loads(runtime.RECORD.read_text())
    assert runtime.account_record(persisted, fixture.specs[0])["last_recovery"]["status"] == "active_restart_pending"
    assert persisted["process_recovery"]["events"][-1]["repair_reason"] == REPAIR_REASON


@pytest.mark.parametrize("change", ["looser_stop", "frozen_level", "frozen_entry", "frozen_tag", "new_trade", "closed_trade", "quantity"])
def test_active_restart_changed_contract_or_trade_activity_requires_review(active_setup, monkeypatch, change):
    fixture = active_setup
    spec = fixture.specs[0]
    add_position(spec)
    def changed_start(*args, **kwargs):
        result = fixture.start(*args, **kwargs)
        with closing(sqlite3.connect(runtime.REPORT / f"{spec.key}_trades.sqlite")) as con, con:
            if change == "looser_stop":
                con.execute("UPDATE trades SET stop_loss=90")
            elif change == "closed_trade":
                con.execute("UPDATE trades SET is_open=0")
            elif change == "quantity":
                con.execute("UPDATE trades SET amount=.9")
            elif change.startswith("frozen"):
                state = focused_state()
                if change == "frozen_level":
                    state["levels"]["target_1"] = 120
                elif change == "frozen_entry":
                    state["entry_rate"] = 101
                else:
                    state["stages"]["stage_1"]["tag"] = "other"
                con.execute("UPDATE trade_custom_data SET cd_value=?", (json.dumps(state),))
        if change == "new_trade":
            add_position(spec, 2)
        return result
    monkeypatch.setattr(runtime, "start_missing", changed_start)
    rows = runtime.restart_active(fixture.record, [s.key for s in fixture.specs], apply=True, repair_reason=REPAIR_REASON)
    assert rows[0]["status"] == "active_restart_continuity_review"
    assert rows[0]["position_continuity_verified"] is False
    assert len(fixture.stopped) == 1
    assert spec.key in fixture.trees  # Never stop/kill the new protective worker on a review issue.


@pytest.mark.parametrize("option", ["--restart-draining", "--initialize-new", "--acknowledge-fixed-startup", "--accounts"])
def test_active_restart_cli_conflicting_actions_rejected(active_setup, option):
    value = "auto" if option in {"--restart-draining", "--accounts"} else (
        "fast_auto" if option == "--acknowledge-fixed-startup" else "sieve_pivot_partial")
    with pytest.raises(ValueError):
        runtime.main(["--restart-active", "sieve_pivot_partial", option, value, "--repair-reason", REPAIR_REASON])
    assert active_setup.stopped == []


@pytest.mark.parametrize("keys,reason", [([], REPAIR_REASON), (["sieve_h4_vp_lvn_long"], REPAIR_REASON),
    (["sieve_pivot_partial"] * 2, REPAIR_REASON), (["sieve_pivot_partial"], None),
    (["sieve_pivot_partial"], "   "), (["sieve_pivot_partial"], "repair"),
    (["sieve_pivot_partial"], "reason has\nnewline")])
def test_active_restart_explicit_names_and_meaningful_reason_required(active_setup, keys, reason):
    with pytest.raises(ValueError):
        runtime.restart_active(active_setup.record, keys, apply=True, repair_reason=reason)
    assert active_setup.stopped == active_setup.started == []


def test_active_restart_cli_readonly_and_unknown_keys(active_setup, monkeypatch, capsys):
    monkeypatch.setattr(runtime, "recovery_lock", nullcontext)
    before = runtime.RECORD.read_bytes()
    assert runtime.main(["--restart-active", "sieve_pivot_partial", "--repair-reason", REPAIR_REASON]) == 0
    packet = json.loads(capsys.readouterr().out)
    assert packet["applied"] is False
    assert runtime.RECORD.read_bytes() == before
    assert active_setup.stopped == []
    with pytest.raises(SystemExit):
        runtime.main(["--restart-active", "sieve_h4_vp_lvn_long", "--repair-reason", REPAIR_REASON])


@pytest.mark.parametrize("status", ["active_restart_pending", "active_restart_continuity_review", "blocked_active_restart_preflight"])
def test_active_restart_cli_unresolved_is_nonzero(active_setup, monkeypatch, status):
    monkeypatch.setattr(runtime, "recovery_lock", nullcontext)
    monkeypatch.setattr(runtime, "restart_active", lambda *a, **k: [{"account": "sieve_pivot_partial", "status": status}])
    assert runtime.main(["--apply", "--restart-active", "sieve_pivot_partial", "--repair-reason", REPAIR_REASON]) == 2


def test_active_restart_crash_after_stop_leaves_durable_no_retry_fence(active_setup, monkeypatch):
    fixture = active_setup
    spec = fixture.specs[0]
    def crash(spec, tree):
        fixture.trees.pop(spec.key)
        raise SystemExit("simulated attended controller crash after stop")
    monkeypatch.setattr(runtime, "_stop_exact_tree", crash)
    with pytest.raises(SystemExit):
        runtime.restart_active(fixture.record, [spec.key], apply=True, repair_reason=REPAIR_REASON)
    persisted = json.loads(runtime.RECORD.read_text())
    assert runtime.account_record(persisted, spec)["last_recovery"]["status"] == "active_restart_in_progress"
    with pytest.raises(RuntimeError, match="no automatic retry"):
        fixture.real_start(persisted, spec)
    assert runtime.run_check(persisted, apply=True, accounts=[spec.key])[0]["status"] == "blocked_attended_restart_unresolved"
    assert fixture.started == []


@pytest.mark.parametrize("scope_args", [["--accounts=auto"], ["--acc", "auto"]])
def test_active_restart_cli_explicit_account_scope_forms_conflict(active_setup, scope_args):
    with pytest.raises(ValueError, match="own exact list"):
        runtime.main(["--restart-active", "sieve_pivot_partial", "--repair-reason", REPAIR_REASON, *scope_args])


def test_active_restart_crash_after_launch_running_keeps_outstanding_fence(active_setup, monkeypatch):
    fixture = active_setup
    spec = fixture.specs[0]
    heartbeat = runtime.worker_heartbeat_state
    def crash(spec, tree, now=None):
        if tree is not None and tree[0]["pid"] >= 700:
            # Actual start_missing persists running before outer continuity checks.
            runtime.save_record(fixture.record)
            raise SystemExit("simulated controller crash before continuity acceptance")
        return heartbeat(spec, tree, now)
    monkeypatch.setattr(runtime, "worker_heartbeat_state", crash)
    with pytest.raises(SystemExit):
        runtime.restart_active(fixture.record, [spec.key], apply=True, repair_reason=REPAIR_REASON)
    persisted = json.loads(runtime.RECORD.read_text())
    row = runtime.account_record(persisted, spec)
    assert row["last_recovery"]["status"] == "running"
    assert row["attended_active_restart"]["status"] == "active_restart_requested"
    with pytest.raises(RuntimeError, match="no automatic retry"):
        fixture.real_start(persisted, spec)
    assert runtime.run_check(persisted, apply=True, accounts=[spec.key])[0]["status"] == "blocked_attended_restart_unresolved"


@pytest.mark.parametrize("clock_fault", ["prior_heartbeat", "old_process_creation"])
def test_active_restart_requires_new_launch_process_and_clock(active_setup, monkeypatch, clock_fault):
    fixture = active_setup
    spec = fixture.specs[0]
    if clock_fault == "prior_heartbeat":
        heartbeat = runtime.worker_heartbeat_state
        def stale_clock(spec, tree, now=None):
            result = heartbeat(spec, tree, now)
            if tree and tree[0]["pid"] >= 700:
                result["at_utc"] = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
            return result
        monkeypatch.setattr(runtime, "worker_heartbeat_state", stale_clock)
    else:
        def old_process(*args, **kwargs):
            result = fixture.start(*args, **kwargs)
            fixture.trees[spec.key][0]["create_time"] -= 60
            return result
        monkeypatch.setattr(runtime, "start_missing", old_process)
    rows = runtime.restart_active(fixture.record, [spec.key], apply=True, repair_reason=REPAIR_REASON)
    assert rows[0]["status"] == "active_restart_continuity_review"
    assert runtime.account_record(fixture.record, spec).get("attended_active_restart")


def test_normal_apply_never_restarts_existing_active_repair_worker(active_setup):
    spec = active_setup.specs[0]
    assert runtime.run_check(active_setup.record, apply=True, accounts=[spec.key])[0]["status"] == "already_running"
    assert active_setup.stopped == active_setup.started == []


def test_shared_path_preserves_attended_draining_restart(active_setup, monkeypatch):
    fixture = active_setup
    spec = next(spec for spec in runtime.LEGACY_ACCOUNTS if spec.key == "auto")
    fixture.trees[spec.key] = (process(spec), None)
    heartbeat = runtime.worker_heartbeat_state
    def paused(spec, tree, now=None):
        result = heartbeat(spec, tree, now)
        if tree and tree[0]["pid"] >= 700:
            result["state"] = "PAUSED"
        return result
    def start_paused(*args, **kwargs):
        result = fixture.start(*args, **kwargs)
        result["status"] = "draining_paused"
        return result
    monkeypatch.setattr(runtime, "worker_heartbeat_state", paused)
    monkeypatch.setattr(runtime, "start_missing", start_paused)
    rows = runtime.restart_draining(fixture.record, [spec.key])
    assert rows[0]["status"] == "drain_restart_completed"
    assert "repair_reason" not in rows[0]
    assert runtime.account_record(fixture.record, spec).get("attended_active_restart") is None
