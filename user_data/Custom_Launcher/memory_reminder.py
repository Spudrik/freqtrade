from __future__ import annotations

from ctypes import Structure, byref, sizeof, c_uint, c_ulonglong, windll
import argparse
from datetime import datetime, timezone
from pathlib import Path
import shutil
import sys


APP_DIR = Path(__file__).resolve().parent
RUNTIME_DIR = APP_DIR / "launcher_v2" / "runtime"
LOG_PATH = RUNTIME_DIR / "memory_reminder.log"


class MEMORYSTATUSEX(Structure):
    _fields_ = [
        ("dwLength", c_uint),
        ("dwMemoryLoad", c_uint),
        ("ullTotalPhys", c_ulonglong),
        ("ullAvailPhys", c_ulonglong),
        ("ullTotalPageFile", c_ulonglong),
        ("ullAvailPageFile", c_ulonglong),
        ("ullTotalVirtual", c_ulonglong),
        ("ullAvailVirtual", c_ulonglong),
        ("ullAvailExtendedVirtual", c_ulonglong),
    ]


def _memory_status() -> tuple[float, float, float]:
    stats = MEMORYSTATUSEX()
    stats.dwLength = sizeof(MEMORYSTATUSEX)
    if not windll.kernel32.GlobalMemoryStatusEx(byref(stats)):
        raise OSError("GlobalMemoryStatusEx failed")
    total_gb = stats.ullTotalPhys / (1024 ** 3)
    free_gb = stats.ullAvailPhys / (1024 ** 3)
    used_gb = total_gb - free_gb
    return total_gb, free_gb, used_gb


def _drive_lines() -> list[str]:
    lines: list[str] = []
    for letter in ("C:", "D:", "E:", "F:", "H:", "I:", "J:", "L:"):
        root = Path(letter + "\\")
        try:
            usage = shutil.disk_usage(str(root))
        except Exception:
            continue
        free_gb = usage.free / (1024 ** 3)
        free_pct = (usage.free / usage.total * 100.0) if usage.total else 0.0
        if free_pct <= 15.0:
            lines.append(f"{letter} free {free_gb:.2f} GB ({free_pct:.2f}%)")
    return lines


def _build_message() -> str:
    total_gb, free_gb, used_gb = _memory_status()
    drive_lines = _drive_lines()
    parts = [
        "Memory reminder: take action on storage/memory pressure soon.",
        f"RAM free: {free_gb:.2f} GB / {total_gb:.2f} GB total ({used_gb:.2f} GB used).",
    ]
    if drive_lines:
        parts.append("Low drives: " + "; ".join(drive_lines))
    else:
        parts.append("No drive is currently below the warning threshold.")
    return "\n".join(parts)


def _log(message: str) -> None:
    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp}\n{message}\n\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-popup", action="store_true", help="Write the reminder log without showing a dialog.")
    args = parser.parse_args()
    message = _build_message()
    _log(message)
    if args.no_popup:
        print(message)
        return 0
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        messagebox.showwarning("Memory reminder", message, parent=root)
        root.destroy()
    except Exception:
        print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
