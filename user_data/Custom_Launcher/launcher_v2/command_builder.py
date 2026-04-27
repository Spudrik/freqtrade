from __future__ import annotations

from dataclasses import dataclass
import shlex
from typing import Iterable


@dataclass(frozen=True)
class BuildResult:
    command_args: list[str]
    preview_command: list[str]
    temp_config_path: str | None = None


def command_text(command: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in command)


def python_module_command(python_exe: str, module: str, args: Iterable[str]) -> list[str]:
    return [str(python_exe), "-m", str(module), *[str(arg) for arg in args]]


def freqtrade_command(python_exe: str, freqtrade_args: Iterable[str]) -> BuildResult:
    args = [str(arg) for arg in freqtrade_args]
    preview = python_module_command(python_exe, "freqtrade", args)
    return BuildResult(command_args=args, preview_command=preview)
