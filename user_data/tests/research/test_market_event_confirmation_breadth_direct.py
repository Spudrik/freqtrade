# ruff: noqa: S101

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from user_data.Custom_Launcher.research.context_features import (
    market_event_confirmation_breadth_direct as direct,
)


PAIRS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "BNB/USDT:USDT",
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
    "ADA/USDT:USDT",
    "TRX/USDT:USDT",
    "AVAX/USDT:USDT",
    "LINK/USDT:USDT",
]
ESTABLISHED = PAIRS[2:]
MEME_PAIRS = ["DOGE/USDT:USDT", "SHIB/USDT:USDT", "PEPE/USDT:USDT"]


def _freeze() -> dict[str, object]:
    return {
        "analysis_groups": {
            "recent_live_media": {
                "partitions": [
                    "development_2026_06_01_to_07_15",
                    "validation_2026_07_16_to_08_30",
                ]
            },
            "historical_context": {
                "partitions": ["development_2021_2023", "internal_validation_2024_2025"]
            },
        },
        "cohorts": {
            "established_alts": ESTABLISHED,
            "top_ten_traded_memes": MEME_PAIRS,
        },
    }


def _leader_metrics(
    sample_id: str,
    *,
    returns: dict[str, float] | None = None,
    activities: dict[str, float] | None = None,
) -> pd.DataFrame:
    returns = returns or {pair: 0.01 for pair in PAIRS}
    activities = activities or {pair: 1.30 for pair in PAIRS}
    rows = []
    for pair in PAIRS:
        rows.append(
            {
                "sample_id": sample_id,
                "event_id": sample_id,
                "event_source": "synthetic_route",
                "event_family": "recent_live_media",
                "whole_event_partition": "development_2026_06_01_to_07_15",
                "sample_type": "event",
                "control_type": "event",
                "control_rank": 0,
                "sample_anchor_utc": pd.Timestamp("2026-06-10T12:00:00Z"),
                "pair": pair,
                "horizon_hours": 1,
                "activity_score": activities[pair],
                "signed_return": returns[pair],
                "pre_return_30d": 0.02,
            }
        )
    return pd.DataFrame.from_records(rows)


def _response_metrics(
    sample_id: str,
    group: str,
    pairs: list[str],
    *,
    one_hour: float = 0.10,
    total: float = 0.32,
    include_one_hour: bool = True,
) -> pd.DataFrame:
    rows = []
    for pair in pairs:
        base = {
            "sample_id": sample_id,
            "event_id": sample_id,
            "event_source": "synthetic_route",
            "event_family": group,
            "whole_event_partition": (
                "development_2026_06_01_to_07_15"
                if group == "recent_live_media"
                else "development_2021_2023"
            ),
            "sample_type": "event",
            "control_type": "event",
            "control_rank": 0,
            "sample_anchor_utc": pd.Timestamp("2026-06-10T12:00:00Z"),
            "pair": pair,
            "activity_score": 1.30,
            "pre_return_30d": 0.02,
        }
        if include_one_hour:
            rows.append({**base, "horizon_hours": 1, "signed_return": one_hour})
        rows.append({**base, "horizon_hours": 2, "signed_return": total})
    return pd.DataFrame.from_records(rows)


def test_sample_table_creates_unique_event_and_control_sample_ids() -> None:
    routes = pd.DataFrame(
        [
            {
                "route_event_id": "route_event_001",
                "analysis_group": "recent_live_media",
                "route_id": "live_news_activity",
                "whole_event_partition": "development_2026_06_01_to_07_15",
                "anchor_utc": pd.Timestamp("2026-06-10T12:00:00Z"),
            },
            {
                "route_event_id": "route_event_002",
                "analysis_group": "recent_live_media",
                "route_id": "live_news_activity",
                "whole_event_partition": "validation_2026_07_16_to_08_30",
                "anchor_utc": pd.Timestamp("2026-07-20T12:00:00Z"),
            },
        ]
    )
    controls = pd.DataFrame(
        [
            {
                "route_event_id": "route_event_001",
                "analysis_group": "recent_live_media",
                "route_id": "live_news_activity",
                "whole_event_partition": "development_2026_06_01_to_07_15",
                "event_anchor_utc": pd.Timestamp("2026-06-10T12:00:00Z"),
                "control_anchor_utc": pd.Timestamp("2026-06-03T12:00:00Z"),
                "control_type": direct.PRIMARY_CONTROL,
                "control_rank": 1,
            },
            {
                "route_event_id": "route_event_001",
                "analysis_group": "recent_live_media",
                "route_id": "live_news_activity",
                "whole_event_partition": "development_2026_06_01_to_07_15",
                "event_anchor_utc": pd.Timestamp("2026-06-10T12:00:00Z"),
                "control_anchor_utc": pd.Timestamp("2026-05-27T12:00:00Z"),
                "control_type": direct.PRIMARY_CONTROL,
                "control_rank": 2,
            },
            {
                "route_event_id": "route_event_002",
                "analysis_group": "recent_live_media",
                "route_id": "live_news_activity",
                "whole_event_partition": "validation_2026_07_16_to_08_30",
                "event_anchor_utc": pd.Timestamp("2026-07-20T12:00:00Z"),
                "control_anchor_utc": pd.Timestamp("2026-07-13T12:00:00Z"),
                "control_type": direct.PRIMARY_CONTROL,
                "control_rank": 1,
            },
        ]
    )
    samples = direct.sample_table(routes, controls)
    assert samples["sample_id"].is_unique
    assert set(samples["sample_type"]) == {"event", "control"}
    assert samples.loc[samples["sample_type"].eq("event"), "sample_id"].str.endswith(
        "|event|event|0"
    ).all()


def test_leader_confirmation_definitions_cover_btc_eth_agreement_and_breadth() -> None:
    low_btc = {pair: 1.30 for pair in PAIRS}
    low_btc["BTC/USDT:USDT"] = 1.24
    mixed_returns = {
        pair: value
        for pair, value in zip(
            PAIRS,
            [0.01, -0.01, 0.01, 0.01, 0.01, -0.01, -0.01, -0.01, -0.01],
            strict=True,
        )
    }
    metrics = pd.concat(
        [
            _leader_metrics("pass", activities={pair: 1.30 for pair in PAIRS}),
            _leader_metrics("low_btc", activities=low_btc),
            _leader_metrics("mixed_breadth", returns=mixed_returns),
        ],
        ignore_index=True,
    )
    leaders = direct.build_leader_rows(metrics, _freeze())

    passed = leaders.loc[leaders["sample_id"].eq("pass")].set_index("leader_type")
    assert passed.loc["btc", "confirmed"]
    assert passed.loc["eth", "confirmed"]
    assert passed.loc["btc_eth_agreement", "confirmed"]
    assert passed.loc["broad_market", "confirmed"]

    low = leaders.loc[leaders["sample_id"].eq("low_btc")].set_index("leader_type")
    assert not low.loc["btc", "confirmed"]
    assert low.loc["eth", "confirmed"]
    assert not low.loc["btc_eth_agreement", "confirmed"]

    mixed = leaders.loc[leaders["sample_id"].eq("mixed_breadth")].set_index(
        "leader_type"
    )
    assert not mixed.loc["btc_eth_agreement", "confirmed"]
    assert mixed.loc["broad_market", "confirmation_asset_count"] == 9
    assert mixed.loc["broad_market", "directional_agreement"] < 0.60
    assert not mixed.loc["broad_market", "confirmed"]


def test_missing_first_hour_becomes_nonusable_abstention(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        direct.layer2,
        "cohort_map",
        lambda _freeze: {"BTC/USDT:USDT": "btc"},
    )
    metrics = _response_metrics(
        "missing_first", "recent_live_media", ["BTC/USDT:USDT"], include_one_hour=False
    )
    response = direct.build_response_rows(metrics, _freeze())
    assert len(response) == 1
    assert response.loc[0, "response_asset_count"] == 0
    assert not response.loc[0, "response_usable"]
    assert pd.isna(response.loc[0, "median_post_first_hour_return"])


def test_post_first_hour_return_uses_compounded_total_and_first_hour() -> None:
    pair = "BTC/USDT:USDT"
    metrics = _response_metrics("formula", "recent_live_media", [pair])
    original = direct.layer2.cohort_map
    direct.layer2.cohort_map = lambda _freeze: {pair: "btc"}
    try:
        response = direct.build_response_rows(metrics, _freeze())
    finally:
        direct.layer2.cohort_map = original
    expected = (1.32 / 1.10) - 1.0
    assert response.loc[0, "median_post_first_hour_return"] == pytest.approx(expected)
    assert response.loc[0, "response_usable"]


def test_recent_meme_scope_is_allowed_but_historical_meme_scope_is_excluded() -> None:
    pair = "DOGE/USDT:USDT"
    metrics = pd.concat(
        [
            _response_metrics("recent_meme", "recent_live_media", [pair]),
            _response_metrics("historical_meme", "historical_context", [pair]),
        ],
        ignore_index=True,
    )
    original = direct.layer2.cohort_map
    direct.layer2.cohort_map = lambda _freeze: {pair: "memes"}
    try:
        response = direct.build_response_rows(metrics, _freeze())
    finally:
        direct.layer2.cohort_map = original
    assert set(response["event_id"]) == {"recent_meme"}
    assert set(response["scope"]) == {"memes"}


@pytest.mark.parametrize(
    ("scope", "pairs", "minimum"),
    [
        ("established_alts", ESTABLISHED[:3], 4),
        ("memes", MEME_PAIRS[:2], 3),
    ],
)
def test_response_member_minimum_gate_abstains_below_threshold(
    scope: str,
    pairs: list[str],
    minimum: int,
) -> None:
    metrics = _response_metrics("too_few", "recent_live_media", pairs)
    original = direct.layer2.cohort_map
    direct.layer2.cohort_map = lambda _freeze: {pair: scope for pair in pairs}
    try:
        response = direct.build_response_rows(metrics, _freeze())
    finally:
        direct.layer2.cohort_map = original
    assert response.loc[0, "response_asset_count"] == len(pairs)
    assert response.loc[0, "minimum_response_assets"] == minimum
    assert not response.loc[0, "response_usable"]


def test_collapsed_control_rate_averages_each_event_before_pooling() -> None:
    controls = pd.DataFrame(
        {
            "event_id": ["control_a", "control_a", "control_b", "control_b"],
            "continued": [True, False, True, True],
        }
    )
    count, rate = direct._collapsed_control_rate(controls, "continued")
    assert count == 2
    assert rate == pytest.approx(0.75)


def _cell_rows(
    *,
    route_id: str,
    confirmed_pattern: list[bool],
    continuation_pattern: list[bool],
    control_pattern: list[bool],
    partitions: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    event_count = len(confirmed_pattern)
    for index, (confirmed, continued) in enumerate(
        zip(confirmed_pattern, continuation_pattern, strict=True)
    ):
        partition = partitions[index % len(partitions)]
        direction = 1
        rows.append(
            {
                "sample_id": f"event_sample_{index}",
                "event_id": f"event_{index}",
                "event_source": route_id,
                "event_family": "recent_live_media",
                "whole_event_partition": partition,
                "sample_type": "event",
                "control_type": "event",
                "control_rank": 0,
                "sample_anchor_utc": pd.Timestamp("2026-06-01T00:00:00Z")
                + pd.Timedelta(hours=index),
                "leader_type": "btc",
                "scope": "btc",
                "horizon_hours": 2,
                "response_usable": True,
                "direction_usable": True,
                "confirmed": confirmed,
                "initial_direction": direction,
                "median_post_first_hour_return": 0.02 if continued else -0.02,
                "continued": continued,
                "prior_30d_trend_continued": continued,
                "response_pre_return_30d": 0.02,
            }
        )
    for index, continued in enumerate(control_pattern):
        partition = partitions[index % len(partitions)]
        rows.append(
            {
                "sample_id": f"control_sample_{index}",
                "event_id": f"control_event_{index}",
                "event_source": route_id,
                "event_family": "recent_live_media",
                "whole_event_partition": partition,
                "sample_type": "control",
                "control_type": direct.PRIMARY_CONTROL,
                "control_rank": 1,
                "sample_anchor_utc": pd.Timestamp("2026-05-01T00:00:00Z")
                + pd.Timedelta(hours=index),
                "leader_type": "btc",
                "scope": "btc",
                "horizon_hours": 2,
                "response_usable": True,
                "direction_usable": True,
                "confirmed": True,
                "initial_direction": 1,
                "median_post_first_hour_return": 0.02 if continued else -0.02,
                "continued": continued,
                "prior_30d_trend_continued": continued,
                "response_pre_return_30d": 0.02,
            }
        )
    assert len(rows) == event_count + len(control_pattern)
    return pd.DataFrame.from_records(rows)


def test_partition_and_shuffle_conditions_have_passing_and_failing_cells() -> None:
    partitions = [
        "development_2026_06_01_to_07_15",
        "validation_2026_07_16_to_08_30",
    ]
    passing = _cell_rows(
        route_id="passing_route",
        confirmed_pattern=[True] * 12 + [False] * 12,
        continuation_pattern=[True] * 12 + [False] * 12,
        control_pattern=[False] * 6 + [True] * 6,
        partitions=partitions,
    )
    failing = _cell_rows(
        route_id="failing_route",
        confirmed_pattern=[True] * 6 + [False] * 6,
        continuation_pattern=[True, False] * 6,
        control_pattern=[False] * 6 + [True] * 6,
        partitions=partitions,
    )
    summary = direct.summarize_cells(
        pd.concat([passing, failing], ignore_index=True), _freeze()
    )
    pass_row = summary.loc[summary["route_id"].eq("passing_route")].iloc[0]
    fail_row = summary.loc[summary["route_id"].eq("failing_route")].iloc[0]
    assert pass_row["confirmed_event_count"] == 12
    assert pass_row["development-2026-06-01-to-07-15_count"] >= 3
    assert pass_row["validation-2026-07-16-to-08-30_count"] >= 3
    assert pass_row["beats_shuffled"]
    assert pass_row["meets_conditional_direction_rule"]
    assert not fail_row["meets_conditional_direction_rule"]
    assert not fail_row["beats_shuffled"] or fail_row["pair_continuation_rate"] < 0.55


def test_route_decisions_classify_zero_one_and_repeated_passing_cells() -> None:
    rows = []
    for route_id, passing_count in (
        ("none", 0),
        ("one", 1),
        ("repeated", 2),
    ):
        for index in range(max(1, passing_count)):
            rows.append(
                {
                    "analysis_group": "recent_live_media",
                    "route_id": route_id,
                    "meets_conditional_direction_rule": index < passing_count,
                    "leader_type": "btc" if index == 0 else "eth",
                    "response_scope": "btc",
                    "total_horizon_hours": 2 + index,
                    "meets_65_target": index == 0,
                    "pair_continuation_rate": 0.70 - index * 0.01,
                }
            )
    decisions = direct.route_decisions(pd.DataFrame.from_records(rows))
    verdicts = decisions.set_index("route_id")["verdict"].to_dict()
    assert verdicts == {
        "none": "no_incremental_direction_lead",
        "one": "provisional_single_cell_lead",
        "repeated": "repeated_conditional_direction_lead",
    }


def _write_frozen_input_fixture(
    tmp_path: Path,
    *,
    result_status: str = "completed_event_confirmation_breadth_freeze",
    tamper_routes_hash: bool = False,
) -> dict[str, Path]:
    freeze_path = tmp_path / "freeze.json"
    routes_path = tmp_path / "routes.csv"
    controls_path = tmp_path / "controls.csv"
    coverage_path = tmp_path / "coverage.csv"
    result_path = tmp_path / "result.json"
    parent_path = tmp_path / "parent.json"

    routes = pd.DataFrame(
        [
            {
                "route_event_id": "route_001",
                "analysis_group": "recent_live_media",
                "route_id": "live_news_activity",
                "whole_event_partition": "development_2026_06_01_to_07_15",
                "anchor_utc": "2026-06-10T12:00:00Z",
            }
        ]
    )
    controls = pd.DataFrame(
        [
            {
                "route_event_id": "route_001",
                "analysis_group": "recent_live_media",
                "route_id": "live_news_activity",
                "whole_event_partition": "development_2026_06_01_to_07_15",
                "event_anchor_utc": "2026-06-10T12:00:00Z",
                "control_anchor_utc": "2026-06-03T12:00:00Z",
            }
        ]
    )
    coverage = pd.DataFrame(
        [
            {
                "analysis_group": "recent_live_media",
                "route_id": "live_news_activity",
                "coverage_eligible": True,
            }
        ]
    )
    routes.to_csv(routes_path, index=False)
    controls.to_csv(controls_path, index=False)
    coverage.to_csv(coverage_path, index=False)
    parent_path.write_text("{}\n", encoding="utf-8")

    def _hash(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    freeze = {
        "status": "frozen_event_confirmation_breadth_before_market_outcomes",
        "outcomes_read": False,
        "parent_freezes": {"parent": {"path": str(parent_path), "sha256": _hash(parent_path)}},
        "parent_catalogues": {"parent": {"path": str(parent_path), "sha256": _hash(parent_path)}},
        "market_input_contracts": {
            "parent": {"path": str(parent_path), "sha256": _hash(parent_path)}
        },
    }
    freeze_path.write_text(json.dumps(freeze), encoding="utf-8")
    result = {
        "status": result_status,
        "outcomes_read": False,
        "artifacts": {
            "freeze": {"sha256": _hash(freeze_path)},
            "routes": {
                "sha256": "stale" if tamper_routes_hash else _hash(routes_path)
            },
            "controls": {"sha256": _hash(controls_path)},
            "coverage": {"sha256": _hash(coverage_path)},
        },
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return {
        "result": result_path,
        "freeze": freeze_path,
        "routes": routes_path,
        "controls": controls_path,
        "coverage": coverage_path,
    }


def _patch_frozen_paths(monkeypatch: pytest.MonkeyPatch, paths: dict[str, Path]) -> None:
    monkeypatch.setattr(direct, "FREEZE_RESULT", paths["result"])
    monkeypatch.setattr(direct, "FREEZE_PATH", paths["freeze"])
    monkeypatch.setattr(direct, "ROUTES_PATH", paths["routes"])
    monkeypatch.setattr(direct, "CONTROLS_PATH", paths["controls"])
    monkeypatch.setattr(direct, "COVERAGE_PATH", paths["coverage"])


def test_load_frozen_inputs_rejects_nonterminal_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_frozen_input_fixture(
        tmp_path, result_status="incomplete_event_confirmation_breadth_freeze"
    )
    _patch_frozen_paths(monkeypatch, paths)
    with pytest.raises(ValueError, match="not terminal"):
        direct.load_frozen_inputs()


def test_load_frozen_inputs_rejects_changed_route_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _write_frozen_input_fixture(tmp_path, tamper_routes_hash=True)
    _patch_frozen_paths(monkeypatch, paths)
    with pytest.raises(ValueError, match="Frozen breadth artifact changed"):
        direct.load_frozen_inputs()
