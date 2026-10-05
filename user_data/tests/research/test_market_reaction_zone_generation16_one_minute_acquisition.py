# ruff: noqa: S101

from __future__ import annotations

import pytest

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation16_one_minute_acquisition as g16a,
)


def test_frozen_paths_are_hash_validated() -> None:
    run_dir, acquisition, coverage, record = g16a.frozen_paths(
        g16a.g16z.DEFAULT_RUN_ID
    )

    assert run_dir.is_dir()
    assert acquisition.is_file()
    assert coverage.name == "g16_one_minute_coverage_audit.csv"
    assert record.name == "g16_one_minute_coverage_record.json"


def test_download_worker_cap_is_one() -> None:
    with pytest.raises(ValueError, match="between 1 and 1"):
        g16a.acquire_missing(g16a.g16z.DEFAULT_RUN_ID, max_workers=2)
