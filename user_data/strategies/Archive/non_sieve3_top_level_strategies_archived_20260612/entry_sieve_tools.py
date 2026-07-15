from __future__ import annotations

import os
from typing import Any


HYPEROPT_PARAM_ENV = "HYBRID_RECOVERY_HYPEROPT_PARAMS"
ENTRY_SIEVE_TAKE_PROFIT_ENV = "ENTRY_SIEVE_TAKE_PROFIT_PCT"
ENTRY_SIEVE_STOPLOSS_ENV = "ENTRY_SIEVE_STOPLOSS_PCT"
ENTRY_SIEVE_CONTROL_EXITS_ENV = "ENTRY_SIEVE_CONTROL_EXITS"


def split_hyperopt_tokens(value: str | None) -> set[str]:
    if not value:
        return set()
    normalized = value.replace(";", ",").replace("|", ",").replace(" ", ",")
    return {token.strip() for token in normalized.split(",") if token.strip()}


def is_parameter_object(value: Any) -> bool:
    return bool(value is not None and value.__class__.__name__.endswith("Parameter"))


def apply_explicit_hyperopt_surface(strategy_cls: type) -> None:
    """Restrict Freqtrade HyperOpt to Explorer-selected params when requested."""
    selected = split_hyperopt_tokens(os.environ.get(HYPEROPT_PARAM_ENV))
    if not selected:
        return
    for name in dir(strategy_cls):
        value = getattr(strategy_cls, name, None)
        if is_parameter_object(value):
            value.optimize = str(name) in selected


def _pct_env(name: str, default_ratio: float) -> float:
    raw = str(os.environ.get(name) or "").strip()
    if not raw:
        return float(default_ratio)
    try:
        return max(0.0, float(raw)) / 100.0
    except ValueError:
        return float(default_ratio)


def _entry_sieve_controls_exits() -> bool:
    raw = str(os.environ.get(ENTRY_SIEVE_CONTROL_EXITS_ENV, "1") or "").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def entry_sieve_minimal_roi(default: float = 0.02) -> dict[str, float]:
    if not _entry_sieve_controls_exits():
        return {"0": float(default)}
    return {"0": _pct_env(ENTRY_SIEVE_TAKE_PROFIT_ENV, default)}


def entry_sieve_stoploss(default: float = -0.02) -> float:
    if not _entry_sieve_controls_exits():
        return float(default)
    return -_pct_env(ENTRY_SIEVE_STOPLOSS_ENV, abs(default))
