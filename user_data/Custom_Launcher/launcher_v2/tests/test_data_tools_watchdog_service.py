from pathlib import Path

from launcher_v2.services.data_tools_watchdog_service import _scheduled_task_xml


def test_scheduled_task_runs_at_boot_without_login() -> None:
    xml = _scheduled_task_xml(Path(r"C:\watchdog\runner.vbs"), 60)

    assert "<BootTrigger>" in xml
    assert "<Interval>PT60M</Interval>" in xml
    assert "<UserId>S-1-5-18</UserId>" in xml
    assert "<StartWhenAvailable>true</StartWhenAvailable>" in xml
    assert "<DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>" in xml
    assert "<StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>" in xml


def test_scheduled_task_escapes_wrapper_path() -> None:
    xml = _scheduled_task_xml(Path(r'C:\watchdog & tools\runner.vbs'), 15)

    assert "<Interval>PT15M</Interval>" in xml
    assert "watchdog &amp; tools" in xml
