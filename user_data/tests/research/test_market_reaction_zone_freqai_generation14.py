# ruff: noqa: S101

from __future__ import annotations

import json

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation14 as g14f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation14_freeze as g14z,
)


def test_active_registry_is_stage_and_cohort_exact() -> None:
    frozen = json.loads(g14z.FREEZE_PATH.read_text(encoding="utf-8"))

    profiles, comparisons = g14f.active_registry(
        frozen, stage=g14z.MTF_STAGE, cohort="normal"
    )

    assert profiles
    assert comparisons
    assert all(item["stage"] == g14z.MTF_STAGE for item in comparisons)
    assert all(item["cohort"] == "normal" for item in comparisons)
    assert all(item["candidate"] in profiles for item in comparisons)
    assert all(item["baseline"] in profiles for item in comparisons)
    assert all(item["route_type"] for item in comparisons)


def test_profile_config_moves_contract_to_generation14() -> None:
    frozen = json.loads(g14z.FREEZE_PATH.read_text(encoding="utf-8"))
    profile = next(
        item
        for item in frozen["profiles"].values()
        if item["stage"] == g14z.MTF_STAGE and item["cohort"] == "normal"
    )
    base = json.loads(g14f.DEFAULT_CONFIG.read_text(encoding="utf-8"))

    config = g14f.profile_config(
        base,
        identifier="g14-test",
        pairs=["BTC/USDT:USDT"],
        feature_dir=g14f.RECORD_ROOT,
        event_dir=g14f.RECORD_ROOT,
        profile=profile,
        train_days=30,
        backtest_days=10,
        technical_smoke=True,
        maximum_target_horizon_hours=8,
    )

    assert "market_reaction_zone_g13" not in config
    assert config["market_reaction_zone_g14"]["feature_columns"] == profile[
        "feature_columns"
    ]
    assert config["market_reaction_zone_g14"]["maximum_target_horizon_hours"] == 8


def test_stage_inventory_exposes_all_preflight_contracts() -> None:
    _, cache, _ = g14f.load_sources(cohort="normal", stage=g14z.MTF_STAGE)

    inventory = g14f.stage_inventory(cache, stage=g14z.MTF_STAGE)

    assert inventory
    for item in inventory:
        for key in ("feature", "support", "event", "evaluation"):
            assert item[f"{key}_path"]
            assert item[f"{key}_sha256"]


def test_common_support_gate_excludes_a_sparse_pair(tmp_path) -> None:
    dates = pd.date_range("2025-12-01", periods=80, freq="D", tz="UTC")
    profiles = {
        "p": {
            "required_ready_blocks": ["block"],
            "targets": ["&-target"],
        }
    }
    for pair, ready_rows in (("GOOD/USDT:USDT", 80), ("SPARSE/USDT:USDT", 35)):
        frame = pd.DataFrame(
            {
                "date": dates,
                "ready__block": [True] * ready_rows + [False] * (80 - ready_rows),
                "&-target": 1.0,
            }
        )
        frame.to_parquet(tmp_path / f"{g14f.g0.pair_file_stem(pair)}.parquet")

    # Add four more supported pairs so the production minimum-five guard remains active.
    candidates = ["GOOD/USDT:USDT"]
    for number in range(4):
        pair = f"GOOD{number}/USDT:USDT"
        candidates.append(pair)
        pd.DataFrame(
            {"date": dates, "ready__block": True, "&-target": 1.0}
        ).to_parquet(tmp_path / f"{g14f.g0.pair_file_stem(pair)}.parquet")
    candidates.append("SPARSE/USDT:USDT")

    supported, audit = g14f.coverage_supported_pairs(
        candidate_pairs=candidates,
        event_dir=tmp_path,
        profiles=profiles,
        timerange="20260101-20260220",
        train_days=30,
    )

    assert "SPARSE/USDT:USDT" not in supported
    assert len(supported) == 5
    sparse = next(item for item in audit if item["pair"] == "SPARSE/USDT:USDT")
    assert not sparse["accepted"]
