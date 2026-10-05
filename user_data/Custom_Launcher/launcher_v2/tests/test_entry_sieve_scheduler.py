from __future__ import annotations

import sys
from pathlib import Path

import pytest

CUSTOM_LAUNCHER_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CUSTOM_LAUNCHER_ROOT))

from launcher_v2.services.entry_sieve_runner import (  # noqa: E402
    _adaptive_hyperopt_jobs,
    _adaptive_pipelined_backtest_worker_limit,
)


def _allocation(
    backlog: int,
    *,
    max_cores: int = 8,
    backtest_workers: int = 8,
) -> tuple[int, int]:
    jobs = _adaptive_hyperopt_jobs(
        max_hyperopt_jobs=max_cores,
        max_cores_allowed=max_cores,
        backtest_worker_count=backtest_workers,
        waiting_backtests=backlog,
        running_backtests=0,
    )
    lanes = _adaptive_pipelined_backtest_worker_limit(
        backtest_lane_count=backtest_workers,
        split_venv_pipeline=True,
        max_cores_allowed=max_cores,
        effective_hyperopt_jobs=jobs,
    )
    return jobs, lanes


def test_initial_hyperopt_uses_all_assigned_cores_without_backtest_work() -> None:
    assert _allocation(0) == (8, 0)


def test_first_queued_backtest_starts_before_core_count_hyperopts_complete() -> None:
    assert _allocation(1) == (7, 1)


@pytest.mark.parametrize(
    ("backlog", "expected"),
    [
        (0, (8, 0)),
        (1, (7, 1)),
        (3, (5, 3)),
        (7, (1, 7)),
        (20, (1, 7)),
    ],
)
def test_queued_backtests_take_priority_without_idling_cores(
    backlog: int,
    expected: tuple[int, int],
) -> None:
    allocation = _allocation(backlog)
    assert allocation == expected
    assert sum(allocation) == 8


def test_configured_backtest_lane_cap_is_respected() -> None:
    assert _allocation(20, backtest_workers=4) == (4, 4)


def test_running_and_waiting_backtests_share_the_same_priority_budget() -> None:
    jobs = _adaptive_hyperopt_jobs(
        max_hyperopt_jobs=8,
        max_cores_allowed=8,
        backtest_worker_count=8,
        waiting_backtests=1,
        running_backtests=2,
    )
    assert jobs == 5
