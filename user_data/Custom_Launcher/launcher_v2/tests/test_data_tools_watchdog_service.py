from pathlib import Path
import json
import sys
from types import SimpleNamespace
import subprocess

import pytest

import launcher_v2.services.data_tools_watchdog_service as watchdog_module
from launcher_v2.services.data_tools_watchdog_service import (
    DataToolsWatchdogService,
    _scheduled_task_xml,
    _scheduled_task_xml_matches,
)
from data_tools_watchdog import _state_from_args, parse_args


def _windows_normalized_task_xml(wrapper: Path, user_sid: str, user_account: str) -> str:
    return f'''<Task version="1.4" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <Triggers>
    <LogonTrigger><UserId>{user_account}</UserId></LogonTrigger>
    <TimeTrigger><Repetition><Interval>PT1H</Interval><StopAtDurationEnd>false</StopAtDurationEnd></Repetition></TimeTrigger>
  </Triggers>
  <Principals><Principal id="Author"><UserId>{user_sid}</UserId><LogonType>InteractiveToken</LogonType></Principal></Principals>
  <Settings><MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy></Settings>
  <Actions Context="Author"><Exec><Command>C:\\Windows\\System32\\wscript.exe</Command><Arguments>//B //NoLogo "{wrapper}"</Arguments></Exec></Actions>
</Task>'''


def test_scheduled_task_runs_for_current_user_on_logon_and_hourly() -> None:
    xml = _scheduled_task_xml(Path(r"C:\watchdog\runner.vbs"), 60, "S-1-5-21-123-456-789-1001")

    assert "<LogonTrigger>" in xml
    assert "<BootTrigger>" not in xml
    assert "<Interval>PT60M</Interval>" in xml
    assert "<UserId>S-1-5-21-123-456-789-1001</UserId>" in xml
    assert "<LogonType>InteractiveToken</LogonType>" in xml
    assert "<MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>" in xml
    assert "<StartWhenAvailable>true</StartWhenAvailable>" in xml
    assert "<DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>" in xml
    assert "<StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>" in xml


def test_scheduled_task_verification_requires_current_user_logon_trigger() -> None:
    wrapper = Path(r"C:\watchdog\runner.vbs")
    user_sid = "S-1-5-21-123-456-789-1001"
    other_sid = "S-1-5-21-123-456-789-1002"
    xml = _scheduled_task_xml(wrapper, 60, user_sid)
    logon_trigger = f"<LogonTrigger><Enabled>true</Enabled><UserId>{user_sid}</UserId></LogonTrigger>"

    assert _scheduled_task_xml_matches(xml, wrapper, 60, user_sid)
    assert f"<Principal id=\"Author\"><UserId>{user_sid}</UserId>" in xml
    assert not _scheduled_task_xml_matches(
        xml.replace(logon_trigger, "<LogonTrigger><Enabled>true</Enabled></LogonTrigger>"),
        wrapper,
        60,
        user_sid,
    )
    assert not _scheduled_task_xml_matches(
        xml.replace(logon_trigger, f"<LogonTrigger><Enabled>true</Enabled><UserId>{other_sid}</UserId></LogonTrigger>"),
        wrapper,
        60,
        user_sid,
    )


def test_scheduled_task_escapes_wrapper_path() -> None:
    xml = _scheduled_task_xml(Path(r'C:\watchdog & tools\runner.vbs'), 15, "S-1-5-21-123-456-789-1001")

    assert "<Interval>PT15M</Interval>" in xml
    assert "watchdog &amp; tools" in xml


def test_scheduled_task_verifies_windows_normalized_identity_interval_and_defaults() -> None:
    wrapper = Path(r"C:\watchdog\runner.vbs")
    user_sid = "S-1-5-21-2001307574-2720987186-41185893-1001"
    user_account = r"DESKTOP-VQFCI0P\engin"
    xml = _windows_normalized_task_xml(wrapper, user_sid, user_account)

    assert _scheduled_task_xml_matches(xml, wrapper, 60, user_sid, user_account.lower())
    assert not _scheduled_task_xml_matches(xml, wrapper, 60, user_sid)
    assert _scheduled_task_xml_matches(xml.replace(user_sid, user_account), wrapper, 60, user_sid, user_account)
    for wrong_account in (r"DESKTOP-VQFCI0P\other", r"OTHER-DOMAIN\engin", "engin"):
        assert not _scheduled_task_xml_matches(xml.replace(user_account, wrong_account), wrapper, 60, user_sid, user_account)
    assert not _scheduled_task_xml_matches(xml.replace(f"<UserId>{user_account}</UserId>", ""), wrapper, 60, user_sid, user_account)
    assert not _scheduled_task_xml_matches(xml.replace(user_sid, "S-1-5-21-2001307574-2720987186-41185893-1002"), wrapper, 60, user_sid, user_account)
    assert not _scheduled_task_xml_matches(xml.replace(f"<UserId>{user_sid}</UserId>", ""), wrapper, 60, user_sid, user_account)
    assert not _scheduled_task_xml_matches(xml, wrapper, 60, "", user_account)
    assert not _scheduled_task_xml_matches(xml, wrapper, 60, user_sid, "engin")


def test_scheduled_task_verifies_equivalent_durations_and_rejects_wrong_or_invalid_intervals() -> None:
    wrapper = Path(r"C:\watchdog\runner.vbs")
    user_sid = "S-1-5-21-123-456-789-1001"
    xml = _scheduled_task_xml(wrapper, 60, user_sid)

    for duration in ("PT1H", "PT3600S", "PT0H60M0S"):
        assert _scheduled_task_xml_matches(xml.replace("PT60M", duration), wrapper, 60, user_sid)
    assert _scheduled_task_xml_matches(xml.replace("PT60M", "P1D"), wrapper, 1440, user_sid)
    assert _scheduled_task_xml_matches(xml.replace("PT60M", "P1DT1H"), wrapper, 1500, user_sid)
    for duration in ("PT59M", "PT61M", "PT1M", "P1D", "", "P", "PT", "P1DT", "P1M", "P1W", "PT-1H", "PT1Hjunk", "PT\u0666\u0660M"):
        assert not _scheduled_task_xml_matches(xml.replace("PT60M", duration), wrapper, 60, user_sid)
    assert not _scheduled_task_xml_matches(xml.replace("<Interval>PT60M</Interval>", ""), wrapper, 60, user_sid)


def test_scheduled_task_rejects_explicit_elevation_and_disabled_defaults() -> None:
    wrapper = Path(r"C:\watchdog\runner.vbs")
    user_sid = "S-1-5-21-123-456-789-1001"
    user_account = r"DESKTOP-VQFCI0P\engin"
    xml = _windows_normalized_task_xml(wrapper, user_sid, user_account)

    assert not _scheduled_task_xml_matches(xml.replace("</Principal>", "<RunLevel>HighestAvailable</RunLevel></Principal>"), wrapper, 60, user_sid, user_account)
    for parent in ("LogonTrigger", "TimeTrigger", "Settings"):
        assert not _scheduled_task_xml_matches(xml.replace(f"</{parent}>", f"<Enabled>false</Enabled></{parent}>"), wrapper, 60, user_sid, user_account)


def test_current_user_identity_rejects_unattested_or_incomplete_rows(monkeypatch) -> None:
    for output in ('"engin","S-1-5-21-123-456-789-1001"', '"DOMAIN\\engin",""', '"DOMAIN\\engin","S-1-5-21-123-456-789-1001"\n"OTHER\\user","S-1-5-21-123-456-789-1002"'):
        monkeypatch.setattr(watchdog_module.subprocess, "run", lambda args, **kwargs: subprocess.CompletedProcess(args, 0, output, ""))
        with pytest.raises(RuntimeError, match="unexpected CSV output"):
            watchdog_module._current_user_identity()


def test_install_verifies_xml_and_retains_legacy_startup_entry(tmp_path: Path, monkeypatch) -> None:
    user_sid = "S-1-5-21-123-456-789-1001"
    user_account = r"DESKTOP-VQFCI0P\engin"
    wrapper = Path(r"C:\watchdog\runner.vbs")
    startup_entry = tmp_path / "FreqtradeDataToolsWatchdog_OnLogon.vbs"
    startup_entry.write_text("legacy entry", encoding="ascii")
    service = DataToolsWatchdogService(tmp_path)
    monkeypatch.setattr(watchdog_module, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(service, "write_hidden_task_wrapper", lambda state: wrapper)
    monkeypatch.setattr(service, "startup_entry_path", lambda task_name: startup_entry)
    queried_xml = _windows_normalized_task_xml(wrapper, user_sid, user_account)
    calls: list[list[str]] = []

    def fake_run(args, **kwargs):
        calls.append(args)
        if args[0] == "whoami":
            return subprocess.CompletedProcess(args, 0, f'"{user_account}","{user_sid}"\n', "")
        if args[1] == "/Create":
            return subprocess.CompletedProcess(args, 0, "created", "")
        return subprocess.CompletedProcess(args, 0, queried_xml, "")

    monkeypatch.setattr(watchdog_module.subprocess, "run", fake_run)

    result = service.install_scheduled_task()

    assert result.returncode == 0
    assert len(calls) == 3
    assert calls[0] == ["whoami", "/user", "/fo", "csv", "/nh"]
    assert calls[2][1:4] == ["/Query", "/TN", "FreqtradeDataToolsWatchdog"]
    assert startup_entry.read_text(encoding="ascii") == "legacy entry"
    assert f"Retained {startup_entry} | exists=True" in result.stdout
    assert "Removed" not in result.stdout


def test_install_keeps_startup_entry_when_scheduler_xml_verification_fails(tmp_path: Path, monkeypatch) -> None:
    user_sid = "S-1-5-21-123-456-789-1001"
    startup_entry = tmp_path / "FreqtradeDataToolsWatchdog_OnLogon.vbs"
    startup_entry.write_text("legacy entry", encoding="ascii")
    service = DataToolsWatchdogService(tmp_path)
    monkeypatch.setattr(watchdog_module, "os", SimpleNamespace(name="nt"))
    monkeypatch.setattr(watchdog_module, "_current_user_identity", lambda: (r"DESKTOP-VQFCI0P\engin", user_sid))
    monkeypatch.setattr(service, "write_hidden_task_wrapper", lambda state: Path(r"C:\watchdog\runner.vbs"))
    monkeypatch.setattr(service, "startup_entry_path", lambda task_name: startup_entry)
    monkeypatch.setattr(
        watchdog_module.subprocess,
        "run",
        lambda args, **kwargs: subprocess.CompletedProcess(args, 0, "<Task />", ""),
    )

    result = service.install_scheduled_task()

    assert result.returncode == 1
    assert startup_entry.read_text(encoding="ascii") == "legacy entry"
    assert "did not match" in result.stdout
    assert "Removed" not in result.stdout


def test_runner_command_leaves_policy_to_fresh_preset(tmp_path: Path) -> None:
    service = DataToolsWatchdogService(tmp_path)
    command = service.build_runner_command(
        {"services": [], "heartbeat_stale_minutes": "2", "restart_dead": False},
        preset_path=tmp_path / "custom-presets.json",
    )

    assert command.endswith(f'--once --preset "{tmp_path / "custom-presets.json"}"')
    assert "--services" not in command
    assert "--heartbeat-stale-minutes" not in command
    assert "--no-restart-dead" not in command


def test_cli_policy_options_are_omitted_or_explicit(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["data_tools_watchdog.py", "--once"])
    omitted = _state_from_args(parse_args())
    assert omitted == {}

    monkeypatch.setattr(sys, "argv", ["data_tools_watchdog.py", "--once", "--services", ""])
    explicit_empty = _state_from_args(parse_args())
    assert explicit_empty == {"services": []}

    monkeypatch.setattr(
        sys,
        "argv",
        ["data_tools_watchdog.py", "--once", "--services", "news,web", "--heartbeat-stale-minutes", "3", "--no-restart-dead"],
    )
    explicit = _state_from_args(parse_args())
    assert explicit == {"heartbeat_stale_minutes": "3", "services": ["news", "web"], "restart_dead": False}


def test_fresh_auto_preset_controls_state_and_durable_stop(tmp_path: Path, monkeypatch) -> None:
    app_dir = tmp_path / "app"
    preset_path = app_dir / "launcher_v2" / "config" / "presets.json"
    preset_path.parent.mkdir(parents=True)
    preset_path.write_text(
        json.dumps(
            {
                "LauncherV2-auto": {
                    "data_watchdog_services": ["news"],
                    "data_watchdog_heartbeat_stale_minutes": "7",
                    "data_watchdog_restart_dead": True,
                    "data_watchdog_operator_note": "preset note",
                    "collector_desired": {"news": False},
                },
                "other": {"data_watchdog_services": ["web"]},
            }
        ),
        encoding="utf-8",
    )
    service = DataToolsWatchdogService(app_dir)
    monkeypatch.setattr(service, "_read_status", lambda key, state: {"verified_running": False, "status": "stopped"})
    monkeypatch.setattr(service, "_start_tool", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not start")))

    payload = service.run_once(preset_path=preset_path)

    assert payload["heartbeat_stale_minutes"] == 7
    assert payload["operator_note"] == "preset note"
    assert payload["services"] == ["news"]
    assert payload["rows"][0]["action"] == "stopped_by_user"
    assert payload["rows"][0]["desired"] is False


def test_explicit_empty_services_runs_no_checks(tmp_path: Path, monkeypatch) -> None:
    service = DataToolsWatchdogService(tmp_path)
    monkeypatch.setattr(service, "_check_tool", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("must not check")))

    payload = service.run_once({"services": []})

    assert payload["services"] == []
    assert payload["rows"] == []
