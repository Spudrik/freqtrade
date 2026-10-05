"""One bounded, allowlisted PAPER process recovery check; never places orders.

Default is read-only. --apply may start missing approved workers and complete
flat/orderless closeout only for the fixed reviewed DRAINING identities. It does
not replace databases or execute recorded shell text. The Windows mutex prevents
Luna and the main review launching duplicate workers.
"""
from __future__ import annotations

import argparse
from contextlib import closing, contextmanager
import ctypes
from ctypes import wintypes
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import time

import psutil
from zoneinfo import ZoneInfo

from freqtrade.configuration.load_config import load_from_files
from user_data.Custom_Launcher.launcher_v2.services.collector_service import utf8_subprocess_env
from user_data.Custom_Launcher.research.paper_trial_control import ALLOWED_PAIRS
from user_data.strategies.paper_news_manual import decision_rows, validate_manual_plan

ROOT = Path(__file__).resolve().parents[3]
REPORT = ROOT / "user_data/research_news_data/context_features/integrated_paper_20260926"
RECORD = REPORT / "run_record.json"
BASE = "user_data/configs/integrated_paper_20260926_base.json"
THREADS = {key: "1" for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_MAX_THREADS", "MKL_NUM_THREADS")}
SIEVE_PAIRS = frozenset({"BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"})


@dataclass(frozen=True)
class Account:
    key: str
    config: str
    strategy: str
    bot_name: str
    record_keys: tuple[str, ...]
    port: int | None = None
    pairs: frozenset[str] = frozenset(ALLOWED_PAIRS)
    new_identity: bool = False


LEGACY_ACCOUNTS = (
    Account("auto", "user_data/configs/integrated_paper_20260926_auto.json", "IntegratedPaper", "integrated_paper_auto", ("accounts", "auto")),
    Account("manual", "user_data/configs/integrated_paper_20260926_manual.json", "IntegratedPaper", "integrated_paper_manual", ("accounts", "manual"), 8096),
    Account("v01", "user_data/configs/integrated_paper_variants/v01.json", "PaperVariant01", "integrated_paper_v01", ("variant_suite", "running", "01")),
    Account("v10", "user_data/configs/integrated_paper_variants/v10.json", "PaperVariant10", "integrated_paper_v10", ("variant_suite", "running", "10")),
    Account("fast_pivot", "user_data/configs/integrated_paper_fast_pivot.json", "PaperFastPivot", "paper_fast_pivot", ("active_directional_arms", "fast_sieve_comparator")),
    Account("leader_impulse", "user_data/configs/integrated_paper_leader_impulse.json", "PaperLeaderImpulse", "paper_leader_impulse", ("active_directional_arms", "high_risk_directional")),
    Account("news_manual", "user_data/configs/paper_news_manual.json", "PaperNewsManual", "paper_news_manual", ("accounts", "news_manual"), 8097),
    Account("news_lab", "user_data/configs/paper_news_lab.json", "PaperNewsManual", "paper_news_lab", ("accounts", "news_lab"), 8098),
    Account("fast_auto", "user_data/configs/paper_fast_auto.json", "PaperFastAuto", "paper_fast_auto", ("fast_reaction_pair", "automatic")),
    Account("fast_context", "user_data/configs/paper_fast_context.json", "PaperFastContext", "paper_fast_context", ("fast_reaction_pair", "contextual")),
    Account("leader_inverse", "user_data/configs/paper_leader_inverse.json", "PaperLeaderInverse", "paper_leader_inverse", ("inverse_signal_pair", "leader")),
    Account("fast_level_inverse", "user_data/configs/paper_fast_level_inverse.json", "PaperFastLevelInverse", "paper_fast_level_inverse", ("inverse_signal_pair", "fast_level")),
)
NEW_SIEVE_ACCOUNTS = (
    Account("sieve_pivot_partial", "user_data/configs/paper_sieve_pivot_partial.json", "PaperSievePivotPartial", "paper_sieve_pivot_partial", ("entry_refresh", "accounts", "sieve_pivot_partial"), pairs=SIEVE_PAIRS, new_identity=True),
    Account("sieve_d1_vp_bos_short", "user_data/configs/paper_sieve_d1_vp_bos_short.json", "PaperSieveD1VpBosShort", "paper_sieve_d1_vp_bos_short", ("entry_refresh", "accounts", "sieve_d1_vp_bos_short"), pairs=SIEVE_PAIRS, new_identity=True),
    Account("sieve_d1_support_break_long", "user_data/configs/paper_sieve_d1_support_break_long.json", "PaperSieveD1SupportBreakLong", "paper_sieve_d1_support_break_long", ("entry_refresh", "accounts", "sieve_d1_support_break_long"), pairs=SIEVE_PAIRS, new_identity=True),
    Account("sieve_h4_vp_lvn_long", "user_data/configs/paper_sieve_h4_vp_lvn_long.json", "PaperSieveH4VpLvnLong", "paper_sieve_h4_vp_lvn_long", ("entry_refresh", "accounts", "sieve_h4_vp_lvn_long"), pairs=SIEVE_PAIRS, new_identity=True),
)
ACCOUNTS = LEGACY_ACCOUNTS + NEW_SIEVE_ACCOUNTS
LEGACY_ACCOUNT_KEYS = frozenset(spec.key for spec in LEGACY_ACCOUNTS)
ALL_ACCOUNT_KEYS = frozenset(spec.key for spec in ACCOUNTS)
DRAINING_ACCOUNT_KEYS = frozenset({"auto", "manual", "v01", "v10", "leader_impulse", "fast_auto", "fast_context"})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def account_record(record: dict, spec: Account) -> dict:
    row = record
    for key in spec.record_keys:
        row = row[key]
    return row


def account_specs_for_record(record: dict) -> tuple[Account, ...]:
    """Accept only the exact pre-refresh or reviewed 16-account registry."""
    allowed = frozenset(record["process_recovery"]["allowed_accounts"])
    if allowed == LEGACY_ACCOUNT_KEYS:
        return LEGACY_ACCOUNTS
    if allowed != ALL_ACCOUNT_KEYS:
        raise RuntimeError("Recorded account allowlist is neither the exact legacy set nor reviewed 16-account set")
    lifecycle = record["process_recovery"].get("account_lifecycle")
    if not isinstance(lifecycle, dict) or set(lifecycle) != ALL_ACCOUNT_KEYS:
        raise RuntimeError("Reviewed account lifecycle map must cover all 16 fixed identities")
    if set(lifecycle.values()) - {"ACTIVE", "DRAINING", "PARKED"}:
        raise RuntimeError("Unknown account lifecycle state")
    if any(lifecycle[key] not in {"DRAINING", "PARKED"} for key in DRAINING_ACCOUNT_KEYS):
        raise RuntimeError("Only the seven reviewed legacy identities may be draining or parked")
    if any(lifecycle[key] != "ACTIVE" for key in ALL_ACCOUNT_KEYS - DRAINING_ACCOUNT_KEYS):
        raise RuntimeError("New and retained paper identities must remain active")
    for spec in ACCOUNTS:
        account_record(record, spec)
    return ACCOUNTS


def lifecycle_for(record: dict, spec: Account) -> str:
    policy = record["process_recovery"]
    if frozenset(policy["allowed_accounts"]) == LEGACY_ACCOUNT_KEYS:
        return "ACTIVE"
    account_specs_for_record(record)
    return policy["account_lifecycle"][spec.key]


def save_record(record: dict) -> None:
    # Reuse the existing atomic writer, under the shared recovery mutex.
    from user_data.Custom_Launcher.research.context_features.global_context_source_preflight import write_text_atomic
    write_text_atomic(RECORD, json.dumps(record, indent=2, allow_nan=False) + "\n")


@contextmanager
def recovery_lock():
    if os.name != "nt":
        raise RuntimeError("This recovery command is configured for Windows only")
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.CreateMutexW.restype = wintypes.HANDLE
    kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateMutexW(None, True, "Local\\FreqtradePaperTrialRecovery20260926")
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    if ctypes.get_last_error() == 183:
        kernel.CloseHandle(handle)
        raise RuntimeError("Another paper recovery check owns the lock; do not launch concurrently")
    try:
        yield
    finally:
        kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


def command_for(spec: Account) -> list[str]:
    return [str(ROOT / ".venv/Scripts/python.exe"), "-m", "freqtrade", "trade",
            "--config", str(ROOT / BASE), "--config", str(ROOT / spec.config),
            "--strategy-path", str(ROOT / "user_data/strategies"),
            "--strategy", spec.strategy, "--logfile", str(REPORT / f"{spec.key}.log")]


def normalize_args(args: list[str], cwd: str) -> tuple[str, ...]:
    normalized = list(args)
    for index, token in enumerate(args[:-1]):
        if token in {"--config", "--strategy-path", "--logfile"}:
            normalized[index + 1] = str((Path(cwd) / args[index + 1]).resolve()).casefold()
    return tuple(normalized)


def process_inventory() -> list[dict]:
    rows = []
    for process in psutil.process_iter(["pid", "ppid", "name", "cmdline", "cwd", "create_time"]):
        info = process.info
        if str(info["name"]).casefold() not in {"python.exe", "pythonw.exe"}:
            continue
        if info["cmdline"] is None:
            raise RuntimeError(f"Cannot identify Python process {info['pid']}; refusing duplicate-risk startup")
        if info["cmdline"][1:4] == ["-m", "freqtrade", "trade"]:
            if info["cwd"] is None:
                raise RuntimeError(f"Cannot identify working directory of Freqtrade process {info['pid']}")
            rows.append(info)
    return rows


def matching_processes(spec: Account, rows: list[dict]) -> tuple[dict, dict | None] | None:
    overlay = str((ROOT / spec.config).resolve()).casefold()
    logfile = str((REPORT / f"{spec.key}.log").resolve()).casefold()
    related = []
    expected = normalize_args(command_for(spec)[1:], str(ROOT))
    for row in rows:
        args = normalize_args(row["cmdline"][1:], row["cwd"])
        claims_account = any(
            args[i] == flag and args[i + 1] == value
            for flag, value in (("--config", overlay), ("--logfile", logfile))
            for i in range(len(args) - 1)
        )
        if not claims_account:
            continue
        if args != expected or Path(row["cwd"]).resolve() != ROOT.resolve():
            raise RuntimeError(f"{spec.key}: conflicting process {row['pid']}; do not duplicate or kill it")
        related.append(row)
    if not related:
        return None
    pids = {r["pid"] for r in related}
    roots = [r for r in related if r["ppid"] not in pids]
    if len(roots) != 1 or len(related) > 2:
        raise RuntimeError(f"{spec.key}: multiple process trees; main-agent attention required")
    parent = roots[0]
    # Windows venv Python normally creates exactly one base-interpreter child.
    children = [r for r in related if r["ppid"] == parent["pid"]]
    return parent, children[0] if children else None


def worker_heartbeat_state(spec: Account, tree: tuple[dict, dict | None] | None,
                           now: datetime | None = None) -> dict:
    """Return only a fresh heartbeat emitted by this exact process tree."""
    now = now or datetime.now(timezone.utc)
    if tree is None:
        return {"state": None, "at_utc": None, "fresh": False}
    path = REPORT / f"{spec.key}.log"
    if not path.is_file():
        return {"state": None, "at_utc": None, "fresh": False}
    process_rows = [item for item in tree if item is not None]
    created = {item["pid"]: datetime.fromtimestamp(item["create_time"], timezone.utc) for item in process_rows}
    path_size = path.stat().st_size
    with path.open("rb") as stream:
        stream.seek(max(0, path_size - 65536))
        lines = stream.read(65536).decode("utf-8", errors="replace").splitlines()
    found = []
    for line in lines:
        match = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - .*Bot heartbeat\. PID=(\d+),", line)
        if match is None:
            continue
        pid = int(match[2])
        if pid not in created:
            continue
        local = datetime.strptime(match[1], "%Y-%m-%d %H:%M:%S,%f").replace(tzinfo=ZoneInfo("Europe/London"))
        if local.replace(fold=0).utcoffset() != local.replace(fold=1).utcoffset():
            continue
        clock = local.astimezone(timezone.utc)
        if clock < created[pid] - timedelta(seconds=2):
            continue
        state = "PAUSED" if "state='PAUSED'" in line else "RUNNING" if "state='RUNNING'" in line else "UNKNOWN"
        found.append((clock, state, pid))
    if not found:
        return {"state": None, "at_utc": None, "fresh": False}
    clock, state, pid = max(found)
    fresh = timedelta(minutes=-2) <= now - clock <= timedelta(minutes=3)
    return {"state": state, "at_utc": clock.isoformat(), "fresh": fresh, "pid": pid}


def validate_config(record: dict, spec: Account) -> dict:
    policy = record["process_recovery"]
    row = account_record(record, spec)
    if (record.get("trial") != "integrated_paper_20260926"
            or Path(record["working_directory"]).resolve() != ROOT.resolve()
            or Path(record["python_exe"]).resolve() != (ROOT / ".venv/Scripts/python.exe").resolve()
            or row["config"] != spec.config or row["db"] != f"{spec.key}_trades.sqlite"
            or row["log"] != f"{spec.key}.log" or row["strategy"] != spec.strategy):
        raise RuntimeError(f"{spec.key}: recorded account identity drifted")
    for filename in (BASE, spec.config):
        digest = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
        if digest != policy["config_sha256"][filename]:
            raise RuntimeError(f"{spec.key}: approved config changed; main-agent review required")
    config = load_from_files([str(ROOT / BASE), str(ROOT / spec.config)])
    expected_files = {str((ROOT / p).resolve()) for p in (BASE, spec.config)}
    if {str(Path(p).resolve()) for p in config["config_files"]} != expected_files:
        raise RuntimeError("Additional configuration includes are not approved for recovery")
    exchange = config["exchange"]
    expected_pairs = spec.pairs
    expected_initial_state = "paused" if lifecycle_for(record, spec) in {"DRAINING", "PARKED"} else "running"
    if (config.get("dry_run") is not True or config.get("trading_mode") != "futures"
            or config.get("margin_mode") != "isolated" or exchange.get("name") != "binance"
            or any(exchange.get(k) for k in ("key", "secret", "password", "privateKey", "private_key"))
            or set(exchange["pair_whitelist"]) != expected_pairs
            or config.get("bot_name") != spec.bot_name or config.get("strategy") != spec.strategy
            or (spec.new_identity and config.get("max_open_trades") != 3)
            or config.get("initial_state", "running") != expected_initial_state
            or config.get("db_url") != f"sqlite:///user_data/research_news_data/context_features/integrated_paper_20260926/{spec.key}_trades.sqlite"):
        raise RuntimeError(f"{spec.key}: not the approved isolated paper configuration")
    api = config.get("api_server", {})
    if spec.port is None:
        if api.get("enabled") or config.get("force_entry_enable"):
            raise RuntimeError(f"{spec.key}: unexpected manual API or forced-entry authority")
    elif (api.get("enabled") is not True or api.get("listen_ip_address") != "127.0.0.1"
          or api.get("listen_port") != spec.port or config.get("force_entry_enable") is not True):
        raise RuntimeError(f"{spec.key}: manual API identity drifted")
    db = REPORT / f"{spec.key}_trades.sqlite"
    if not db.is_file():
        row = account_record(record, spec)
        if (not spec.new_identity or row.get("database_initialized_at_utc")
                or row.get("database_create_attempted_at_utc")):
            raise RuntimeError(f"{spec.key}: existing database missing; refuse a silent new balance")
        return config
    with closing(sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.execute("SELECT id FROM trades LIMIT 1").fetchall()
        if spec.key in {"news_manual", "news_lab"}:
            plans = connection.execute("SELECT t.id, c.cd_value FROM trades t LEFT JOIN trade_custom_data c ON c.ft_trade_id=t.id AND c.cd_key='paper_news_manual_plan' WHERE t.is_open=1").fetchall()
            for trade_id, encoded in plans:
                if encoded is None:
                    raise RuntimeError(f"{spec.key}: open trade {trade_id} has no persisted protection")
                plan = json.loads(encoded)
                validate_manual_plan(plan, float(plan["reference_rate"]))
        if spec.key in {"fast_auto", "fast_context", "fast_level_inverse"}:
            from user_data.strategies.paper_fast_reaction import validate_fast_plan
            plans = connection.execute("SELECT t.id, c.cd_value FROM trades t LEFT JOIN trade_custom_data c ON c.ft_trade_id=t.id AND c.cd_key='paper_fast_plan' WHERE t.is_open=1").fetchall()
            for trade_id, encoded in plans:
                if encoded is None:
                    raise RuntimeError(f"{spec.key}: open trade {trade_id} has no persisted protection")
                validate_fast_plan(json.loads(encoded))
    return config


def validate_journal(spec: Account) -> None:
    if spec.port is None:
        return
    journal = REPORT / ("manual_interventions.jsonl" if spec.key == "manual" else f"{spec.key}_decisions.jsonl")
    latest = {r["decision_id"]: r for r in decision_rows(journal)}
    if any(r.get("status") in {"proposed", "uncertain_manual_check_required"} for r in latest.values()):
        raise RuntimeError(f"{spec.key}: unresolved manual API decision; inspect without restarting/retrying")


def _preflight_write_environment() -> None:
    if any(key.startswith("FREQTRADE__") for key in os.environ):
        raise RuntimeError("Unexpected Freqtrade environment overrides; resolve before any process action")
    if not (ROOT / ".venv/Scripts/python.exe").is_file():
        raise RuntimeError("Configured controller Python is missing; no interpreter substitution")


def _validate_new_strategy_bootstrap(record: dict, spec: Account, config: dict | None = None) -> None:
    """Load the exact wrapper and assert locked parameters before its first process launch."""
    if not spec.new_identity:
        raise ValueError("Bootstrap validation is restricted to new Sieve identities")
    from freqtrade.enums import RunMode
    from freqtrade.resolvers import StrategyResolver
    from freqtrade.configuration import validate_config_consistency
    config = config or validate_config(record, spec)
    config.update(runmode=RunMode.DRY_RUN, strategy_path=str(ROOT / "user_data/strategies"),
                  user_data_dir=ROOT / "user_data")
    strategy = StrategyResolver.load_strategy(config)
    # Match Freqtrade's startup order: bot_start runs before hyperparameters load.
    strategy.bot_start()
    strategy.ft_load_hyper_params()
    validate_config_consistency(config)


def _resource_reserve_available() -> bool:
    logical = psutil.cpu_count()
    if logical is None or logical <= 4:
        return False
    return psutil.cpu_percent(interval=1.) <= 100. * (logical - 4) / logical


def note_running(row: dict, tree: tuple[dict, dict | None]) -> None:
    parent, worker = tree
    row.update(parent_pid=parent["pid"], worker_pid=None if worker is None else worker["pid"],
               parent_created_at_utc=datetime.fromtimestamp(parent["create_time"], timezone.utc).isoformat())


def start_missing(record: dict, spec: Account, *, resource_prechecked: bool = False,
                  allow_initial_new_identity: bool = False) -> dict:
    row = account_record(record, spec)
    if row.get("last_recovery", {}).get("status") in {"launching", "failed", "starting"}:
        raise RuntimeError(f"{spec.key}: prior launch unresolved/failed; no automatic retry")
    config = validate_config(record, spec)
    if spec.new_identity:
        db = REPORT / f"{spec.key}_trades.sqlite"
        if not db.is_file() and not allow_initial_new_identity:
            raise RuntimeError(f"{spec.key}: first database creation requires attended --initialize-new")
        if db.is_file() and not row.get("database_initialized_at_utc"):
            raise RuntimeError(f"{spec.key}: database exists without a successful registered initialization; review required")
        _validate_new_strategy_bootstrap(record, spec, config)
    validate_journal(spec)
    if spec.port is not None and any(c.status == psutil.CONN_LISTEN and c.laddr.port == spec.port
                                     for c in psutil.net_connections(kind="tcp")):
        raise RuntimeError(f"{spec.key}: API port is occupied by an unidentified process")
    if not resource_prechecked and not _resource_reserve_available():
        return {"account": spec.key, "status": "deferred_resource_reserve"}
    command = command_for(spec)
    launched_at = utc_now()
    log_path = REPORT / f"{spec.key}.log"
    log_offset = log_path.stat().st_size if log_path.exists() else 0
    if spec.new_identity and not (REPORT / f"{spec.key}_trades.sqlite").is_file():
        row["database_create_attempted_at_utc"] = launched_at
    row["last_recovery"] = {"status": "launching", "at_utc": launched_at, "log_offset": log_offset}
    save_record(record)  # Persist attempt first; a crash must not invite another launch.
    env = utf8_subprocess_env()
    env.update(THREADS)
    try:
        process = subprocess.Popen(command, cwd=str(ROOT), env=env, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                   close_fds=True, creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP)
        row.update(parent_pid=process.pid, worker_pid=None, restarted_at_utc=launched_at,
                   command=subprocess.list2cmdline(command))
        save_record(record)
        deadline = time.monotonic() + 30.
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(f"{spec.key}: worker exited with code {process.returncode}; inspect its existing log")
            tree = matching_processes(spec, process_inventory())
            if tree:
                note_running(row, tree)
            text = ""
            if log_path.exists():
                with log_path.open(encoding="utf-8", errors="replace") as handle:
                    handle.seek(log_offset)
                    text = handle.read()
            if " - ERROR - " in text or "Traceback" in text:
                raise RuntimeError(f"{spec.key}: startup error; do not kill or retry the process automatically")
            lifecycle = lifecycle_for(record, spec)
            expected_log_state = "PAUSED" if lifecycle == "DRAINING" else "RUNNING"
            if f"state='{expected_log_state}'" in text or f"Changing state to: {expected_log_state}" in text:
                row["last_recovery"]["status"] = "draining_paused" if lifecycle == "DRAINING" else "running"
                db = REPORT / f"{spec.key}_trades.sqlite"
                if db.is_file() and spec.new_identity and not row.get("database_initialized_at_utc"):
                    row["database_initialized_at_utc"] = utc_now()
                break
            time.sleep(2.)
        else:
            row["last_recovery"]["status"] = "starting"
    except (OSError, RuntimeError) as error:
        row["last_recovery"].update(status="failed", error=str(error))
        save_record(record)
        raise
    record["process_recovery"].setdefault("events", []).append({"account": spec.key, **row["last_recovery"],
                                                               "parent_pid": row["parent_pid"], "worker_pid": row["worker_pid"]})
    save_record(record)
    return {"account": spec.key, "status": row["last_recovery"]["status"],
            "parent_pid": row["parent_pid"], "worker_pid": row["worker_pid"]}


def _database_activity(spec: Account) -> tuple[int, int]:
    db = REPORT / f"{spec.key}_trades.sqlite"
    with closing(sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        open_trades = connection.execute("SELECT COUNT(*) FROM trades WHERE is_open=1").fetchone()[0]
        open_orders = connection.execute("SELECT COUNT(*) FROM orders WHERE ft_is_open=1").fetchone()[0]
    return int(open_trades), int(open_orders)


def _unprotected_open_trade(spec: Account) -> int | None:
    db = REPORT / f"{spec.key}_trades.sqlite"
    with closing(sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        row = connection.execute(
            "SELECT id FROM trades WHERE is_open=1 AND (stop_loss IS NULL OR stop_loss<=0) LIMIT 1").fetchone()
    return None if row is None else int(row[0])


def _open_position_snapshot(spec: Account) -> dict[int, dict]:
    db = REPORT / f"{spec.key}_trades.sqlite"
    with closing(sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        trades = connection.execute(
            "SELECT id,pair,is_short,open_rate,stop_loss,leverage FROM trades WHERE is_open=1 ORDER BY id").fetchall()
        result = {}
        for trade in trades:
            custom = connection.execute(
                "SELECT cd_key,cd_value FROM trade_custom_data WHERE ft_trade_id=? ORDER BY cd_key",
                (trade["id"],)).fetchall()
            result[int(trade["id"])] = {**dict(trade),
                "custom_data": {row["cd_key"]: row["cd_value"] for row in custom}}
    return result


def _position_continuity(before: dict[int, dict], after: dict[int, dict]) -> tuple[bool, list[int], list[int]]:
    """Check surviving open IDs keep identity, a no-looser stop, and saved plan keys."""
    if set(after) - set(before):
        return False, [], sorted(set(after) - set(before))
    closed = sorted(set(before) - set(after))
    for trade_id in set(before) & set(after):
        old, new = before[trade_id], after[trade_id]
        if any(old[key] != new[key] for key in ("pair", "is_short", "open_rate", "leverage")):
            return False, closed, []
        old_stop, new_stop = old["stop_loss"], new["stop_loss"]
        if old_stop is None or new_stop is None:
            return False, closed, []
        if (not bool(old["is_short"]) and float(new_stop) < float(old_stop)) or (
                bool(old["is_short"]) and float(new_stop) > float(old_stop)):
            return False, closed, []
        before_custom, after_custom = old["custom_data"], new["custom_data"]
        for key, value in before_custom.items():
            if "plan" in key.casefold() and after_custom.get(key) != value:
                return False, closed, []
            if "sieve3_v2_focused" in key.casefold() and key not in after_custom:
                return False, closed, []
    return True, closed, []


def _stop_exact_tree(spec: Account, tree: tuple[dict, dict | None]) -> None:
    """Target only the exact matched parent/worker pair; never force-kill or claim graceful shutdown."""
    current = matching_processes(spec, process_inventory())
    if current is None or tuple((item["pid"], item["create_time"]) if item else None for item in current) != tuple(
            (item["pid"], item["create_time"]) if item else None for item in tree):
        raise RuntimeError(f"{spec.key}: exact process tree changed before the requested lifecycle action")
    processes = []
    for item in (tree[1], tree[0]):
        if item is None:
            continue
        try:
            process = psutil.Process(item["pid"])
            if abs(process.create_time() - item["create_time"]) > 0.01:
                raise RuntimeError(f"{spec.key}: PID identity changed immediately before targeted stop")
            processes.append(process)
        except psutil.NoSuchProcess:
            # This exact, previously verified PID exited during the closeout window.
            continue
    for process in processes:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            continue
    _, alive = psutil.wait_procs(processes, timeout=10)
    if alive:
        raise RuntimeError(f"{spec.key}: exact worker did not stop cleanly; no force-kill attempted")
    if matching_processes(spec, process_inventory()) is not None:
        raise RuntimeError(f"{spec.key}: a matching process remains after the bounded stop; no retry attempted")


def _stop_exact_flat_tree(spec: Account, tree: tuple[dict, dict | None]) -> None:
    """Require fresh PAUSED evidence and a second DB-flat check before closeout."""
    observed = worker_heartbeat_state(spec, tree)
    if observed["state"] != "PAUSED" or not observed["fresh"]:
        raise RuntimeError(f"{spec.key}: refusing flat closeout without a fresh PAUSED heartbeat from the matched tree")
    open_trades, open_orders = _database_activity(spec)
    if open_trades or open_orders:
        raise RuntimeError(f"{spec.key}: DB activity changed before flat closeout ({open_trades} trades, {open_orders} orders)")
    _stop_exact_tree(spec, tree)
    open_trades, open_orders = _database_activity(spec)
    if open_trades or open_orders:
        raise RuntimeError(f"{spec.key}: DB activity appeared during closeout; lifecycle remains DRAINING")


def _park_flat_draining(record: dict, spec: Account, tree: tuple[dict, dict | None] | None) -> dict:
    open_trades, open_orders = _database_activity(spec)
    if open_trades or open_orders:
        return {"account": spec.key, "status": "draining_protection_continues",
                "open_trades": open_trades, "open_orders": open_orders}
    if tree is not None:
        _stop_exact_flat_tree(spec, tree)
    else:
        open_trades, open_orders = _database_activity(spec)
        if open_trades or open_orders:
            raise RuntimeError(f"{spec.key}: DB activity appeared before flat closeout")
    policy = record["process_recovery"]
    policy["account_lifecycle"][spec.key] = "PARKED"
    policy["paused_accounts"] = sorted(set(policy.get("paused_accounts", [])) | {spec.key})
    row = account_record(record, spec)
    at = utc_now()
    row["lifecycle_changed_at_utc"] = at
    row["last_recovery"] = {"status": "parked_flat", "at_utc": at, "open_trades": 0, "open_orders": 0}
    policy.setdefault("events", []).append({"account": spec.key, **row["last_recovery"]})
    save_record(record)
    return {"account": spec.key, "status": "parked_flat", "stopped_exact_tree": tree is not None}


def restart_draining(record: dict, account_keys: list[str]) -> list[dict]:
    """Attended exact-list restart onto paused overlays, preserving every DB."""
    _preflight_write_environment()
    specs = account_specs_for_record(record)
    if len(specs) != len(ACCOUNTS) or not account_keys or not set(account_keys) <= DRAINING_ACCOUNT_KEYS:
        raise ValueError("Drain restart is limited to named reviewed DRAINING accounts in the 16-account registry")
    results = []
    for key in account_keys:
        spec = next(spec for spec in LEGACY_ACCOUNTS if spec.key == key)
        if lifecycle_for(record, spec) != "DRAINING":
            raise RuntimeError(f"{key}: only the exact DRAINING lifecycle may be restarted paused")
        validate_config(record, spec)
        tree = matching_processes(spec, process_inventory())
        open_trades, open_orders = _database_activity(spec)
        if tree is None and not open_trades and not open_orders:
            results.append(_park_flat_draining(record, spec, None))
            continue
        unprotected = _unprotected_open_trade(spec)
        if unprotected is not None:
            results.append({"account": key, "status": "blocked_missing_persisted_stop",
                            "trade_id": unprotected, "open_trades": open_trades})
            continue
        before = _open_position_snapshot(spec)
        observed = {"state": "MISSING", "fresh": True} if tree is None else worker_heartbeat_state(spec, tree)
        if tree is not None and (not observed["fresh"] or observed["state"] not in {"RUNNING", "PAUSED"}):
            results.append({"account": key, "status": "blocked_worker_state_unverified", "heartbeat": observed})
            continue
        if tree is not None:
            if not _resource_reserve_available():
                results.append({"account": key, "status": "deferred_resource_reserve_before_restart",
                                "open_trades": open_trades, "worker_left_running": True})
                continue
            latest_before_stop = _open_position_snapshot(spec)
            if set(latest_before_stop) - set(before):
                results.append({"account": key, "status": "blocked_new_position_before_stop",
                                "new_trade_ids": sorted(set(latest_before_stop) - set(before))})
                continue
            before = latest_before_stop
            _, orders_before_stop = _database_activity(spec)
            if orders_before_stop:
                results.append({"account": key, "status": "blocked_open_orders_before_stop",
                                "open_orders": orders_before_stop})
                continue
            _stop_exact_tree(spec, tree)
        try:
            result = start_missing(record, spec, resource_prechecked=tree is not None)
        except (OSError, RuntimeError, ValueError) as error:
            event = {"account": key, "status": "drain_restart_pending", "at_utc": utc_now(),
                     "preserved_database": f"{key}_trades.sqlite", "open_trade_ids_before": sorted(before),
                     "prior_worker_state": observed["state"], "blocker": str(error)}
            record["process_recovery"].setdefault("events", []).append(event)
            results.append(event)
            save_record(record)
            break
        if result.get("status") != "draining_paused":
            event = {"account": key, "status": "drain_restart_pending", "at_utc": utc_now(),
                     "preserved_database": f"{key}_trades.sqlite", "open_trade_ids_before": sorted(before),
                     "prior_worker_state": observed["state"], "startup": result}
            record["process_recovery"].setdefault("events", []).append(event)
            results.append(event)
            save_record(record)
            break
        new_tree = matching_processes(spec, process_inventory())
        heartbeat_deadline = time.monotonic() + 30.
        new_heartbeat = worker_heartbeat_state(spec, new_tree)
        while True:
            row = account_record(record, spec)
            expected_parent = row.get("parent_pid")
            expected_worker = row.get("worker_pid")
            if (new_tree is None or new_tree[0]["pid"] != expected_parent
                    or (expected_worker is not None
                        and (new_tree[1] is None or new_tree[1]["pid"] != expected_worker))):
                break
            if new_heartbeat["state"] == "PAUSED" and new_heartbeat["fresh"]:
                break
            if time.monotonic() >= heartbeat_deadline:
                break
            time.sleep(2.)
            new_tree = matching_processes(spec, process_inventory())
            new_heartbeat = worker_heartbeat_state(spec, new_tree)
        after = _open_position_snapshot(spec)
        continuous, closed_ids, unexpected_ids = _position_continuity(before, after)
        current_row = account_record(record, spec)
        exact_tree_matches_record = bool(new_tree is not None
            and new_tree[0]["pid"] == current_row.get("parent_pid")
            and (current_row.get("worker_pid") is None
                 or (new_tree[1] is not None and new_tree[1]["pid"] == current_row.get("worker_pid"))))
        if (not exact_tree_matches_record or new_heartbeat["state"] != "PAUSED"
                or not new_heartbeat["fresh"] or not continuous):
            event = {"account": key, "status": "drain_restart_continuity_review", "at_utc": utc_now(),
                     "open_trade_ids_before": sorted(before), "open_trade_ids_after": sorted(after),
                     "closed_ids_during_restart": closed_ids, "unexpected_open_ids": unexpected_ids,
                     "startup_heartbeat": new_heartbeat,
                     "position_continuity_verified": continuous,
                     "exact_tree_matches_record": exact_tree_matches_record}
            record["process_recovery"].setdefault("events", []).append(event)
            results.append(event)
            save_record(record)
            break
        event = {"account": key, "status": "drain_restart_completed", "at_utc": utc_now(),
                 "preserved_database": f"{key}_trades.sqlite", "open_trade_ids_before": sorted(before),
                 "open_trade_ids_after": sorted(after), "closed_ids_during_restart": closed_ids,
                 "prior_worker_state": observed["state"], "startup": result,
                 "startup_heartbeat_at_utc": new_heartbeat["at_utc"]}
        record["process_recovery"].setdefault("events", []).append(event)
        results.append({"account": key, **event})
        save_record(record)
    return results


def initialize_new_accounts(record: dict, account_keys: list[str]) -> list[dict]:
    """Attended first launch for only the four new, isolated PAPER identities."""
    _preflight_write_environment()
    specs = account_specs_for_record(record)
    new_keys = {spec.key for spec in NEW_SIEVE_ACCOUNTS}
    if len(specs) != len(ACCOUNTS) or not account_keys or not set(account_keys) <= new_keys:
        raise ValueError("Initialization is limited to explicitly named new Sieve PAPER identities")
    results = []
    for key in account_keys:
        spec = next(spec for spec in NEW_SIEVE_ACCOUNTS if spec.key == key)
        row = account_record(record, spec)
        if lifecycle_for(record, spec) != "ACTIVE" or row.get("database_initialized_at_utc"):
            raise RuntimeError(f"{key}: identity is not an uninitialized ACTIVE new account")
        if matching_processes(spec, process_inventory()) is not None:
            raise RuntimeError(f"{key}: matching worker already exists; no initialization or replacement")
        if (REPORT / f"{key}_trades.sqlite").exists():
            raise RuntimeError(f"{key}: database already exists but is not registered initialized; review required")
        validate_config(record, spec)
        result = start_missing(record, spec, allow_initial_new_identity=True)
        results.append(result)
        if result.get("status") != "running":
            break
    return results


def run_check(record: dict, *, apply: bool, accounts: list[str]) -> list[dict]:
    policy = record["process_recovery"]
    specs = account_specs_for_record(record)
    expected_keys = {s.key for s in specs}
    if policy.get("enabled") is not True or set(policy["allowed_accounts"]) != expected_keys:
        raise RuntimeError("Paper process recovery is not explicitly enabled for the fixed account set")
    if apply:
        _preflight_write_environment()
    paused = set(policy["paused_accounts"])
    if not paused <= expected_keys:
        raise ValueError("Unknown paused account")
    if len(specs) == len(ACCOUNTS):
        parked = {key for key, state in policy["account_lifecycle"].items() if state == "PARKED"}
        if parked != paused:
            raise RuntimeError("paused_accounts must equal the exact PARKED lifecycle set")
    results = []
    for spec in specs:
        if spec.key not in accounts:
            continue
        lifecycle = lifecycle_for(record, spec)
        if lifecycle == "PARKED" or spec.key in paused:
            results.append({"account": spec.key, "status": "paused_do_not_restart"})
            continue
        try:
            validate_config(record, spec)
            tree = matching_processes(spec, process_inventory())
            if lifecycle == "DRAINING":
                if apply:
                    result = _park_flat_draining(record, spec, tree)
                    if result["status"] == "draining_protection_continues" and tree is None:
                        unprotected = _unprotected_open_trade(spec)
                        if unprotected is not None:
                            result = {"account": spec.key, "status": "blocked_missing_persisted_stop",
                                      "trade_id": unprotected}
                        else:
                            result = start_missing(record, spec)
                    results.append(result)
                elif tree is None:
                    open_trades, open_orders = _database_activity(spec)
                    results.append({"account": spec.key,
                        "status": "missing_draining_worker" if open_trades or open_orders else "would_park_flat",
                        "open_trades": open_trades, "open_orders": open_orders})
                else:
                    open_trades, open_orders = _database_activity(spec)
                    results.append({"account": spec.key,
                        "status": "draining_expected_paused" if open_trades or open_orders else "would_park_flat",
                        "open_trades": open_trades, "open_orders": open_orders,
                        "parent_pid": tree[0]["pid"], "worker_pid": None if tree[1] is None else tree[1]["pid"]})
                continue
            if tree:
                if apply:
                    note_running(account_record(record, spec), tree)
                results.append({"account": spec.key, "status": "already_running",
                                "parent_pid": tree[0]["pid"], "worker_pid": None if tree[1] is None else tree[1]["pid"]})
            elif not apply:
                results.append({"account": spec.key, "status": "missing_would_start"})
            elif (spec.new_identity
                  and not account_record(record, spec).get("database_create_attempted_at_utc")
                  and not account_record(record, spec).get("database_initialized_at_utc")
                  and not (REPORT / f"{spec.key}_trades.sqlite").is_file()):
                results.append({"account": spec.key, "status": "blocked_requires_attended_initialization"})
            else:
                results.append(start_missing(record, spec))
        except (OSError, RuntimeError, ValueError, KeyError, sqlite3.Error, psutil.AccessDenied) as error:
            results.append({"account": spec.key, "status": "blocked", "error": str(error)})
    if apply:
        policy.update(last_check_at_utc=utc_now(), last_results=results)
        save_record(record)
    return results


def acknowledge_fixed_startup(record: dict, accounts: list[str], reason: str) -> None:
    """Explicit main-agent repair acknowledgement, only for untraded new fast bots.

    Never used by scheduled recovery. Preserves the failed attempt and requires
    an exited worker, unchanged approved config, empty DB and passing strategy
    startup contract before permitting one diagnosed implementation relaunch.
    """
    from freqtrade.enums import RunMode
    from freqtrade.resolvers import StrategyResolver
    from freqtrade.configuration import validate_config_consistency
    if not reason or not set(accounts) <= {"fast_auto", "fast_context"}:
        raise ValueError("A named root-cause repair for the new fast bots is required")
    for key in accounts:
        spec = next(s for s in ACCOUNTS if s.key == key)
        row = account_record(record, spec)
        prior = row.get("last_recovery", {})
        if prior.get("status") != "failed" or "startup error" not in prior.get("error", ""):
            raise RuntimeError("Only a diagnosed failed initial startup can be acknowledged")
        config = validate_config(record, spec)
        if matching_processes(spec, process_inventory()) is not None:
            raise RuntimeError("Existing worker: no retry or replacement")
        with closing(sqlite3.connect((REPORT/row["db"]).resolve().as_uri()+"?mode=ro", uri=True)) as con:
            if con.execute("SELECT COUNT(*) FROM trades").fetchone()[0]:
                raise RuntimeError("Account has traded; initial-startup acknowledgement is not permitted")
        config.update(runmode=RunMode.DRY_RUN, strategy_path=str(ROOT/"user_data/strategies"),
                      user_data_dir=ROOT/"user_data")
        strategy = StrategyResolver.load_strategy(config)
        validate_config_consistency(config)
        strategy.bot_start()
        record["process_recovery"].setdefault("events", []).append({"account": key, **prior,
            "acknowledged_at_utc": utc_now(), "root_cause_repair": reason})
        row["last_recovery"] = {"status": "fixed_startup_verified", "at_utc": utc_now(), "reason": reason}
    save_record(record)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Start approved missing workers; finalize only reviewed flat/orderless drains")
    parser.add_argument("--accounts", nargs="+", choices=sorted(ALL_ACCOUNT_KEYS), default=sorted(ALL_ACCOUNT_KEYS))
    parser.add_argument("--restart-draining", nargs="+", choices=sorted(DRAINING_ACCOUNT_KEYS),
                        help="Attended, exact-list restart onto paused overlays; requires --apply")
    parser.add_argument("--initialize-new", nargs="+", choices=sorted(spec.key for spec in NEW_SIEVE_ACCOUNTS),
                        help="Attended first launch for isolated new Sieve accounts; requires --apply")
    parser.add_argument("--acknowledge-fixed-startup", nargs="+", choices=("fast_auto", "fast_context"))
    parser.add_argument("--repair-reason")
    args = parser.parse_args(argv)
    if args.restart_draining and args.initialize_new:
        raise ValueError("Choose one attended lifecycle action per lock acquisition")
    if (args.restart_draining or args.initialize_new) and args.acknowledge_fixed_startup:
        raise ValueError("Choose one attended lifecycle/repair action per lock acquisition")
    with recovery_lock():
        record = json.loads(RECORD.read_text(encoding="utf-8"))
        if args.restart_draining or args.initialize_new:
            if not args.apply:
                raise ValueError("Attended lifecycle actions require --apply")
            if args.restart_draining:
                results = restart_draining(record, args.restart_draining)
            else:
                results = initialize_new_accounts(record, args.initialize_new)
            save_record(record)
            applied = True
        else:
            applied = args.apply
            results = None
        if args.acknowledge_fixed_startup:
            if not args.apply:
                raise ValueError("Repair acknowledgement requires an explicit apply run")
            acknowledge_fixed_startup(record, args.acknowledge_fixed_startup, args.repair_reason)
        if results is None:
            results = run_check(record, apply=args.apply, accounts=args.accounts)
    print(json.dumps({"applied": applied, "accounts": results}, indent=2))
    blocked_statuses = {"blocked", "starting", "deferred_resource_reserve", "deferred_resource_reserve_before_restart",
                        "drain_restart_pending", "drain_restart_continuity_review", "missing_draining_worker"}
    return 2 if any(r["status"] in blocked_statuses or r["status"].startswith("blocked_") for r in results) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(json.dumps({"error": str(error)}))
        raise SystemExit(2)
