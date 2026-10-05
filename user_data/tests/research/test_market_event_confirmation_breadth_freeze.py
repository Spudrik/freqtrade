from __future__ import annotations

# ruff: noqa: S101
import json
from pathlib import Path

import pandas as pd
import pytest

from user_data.Custom_Launcher.research.context_features import (
    market_event_confirmation_breadth_freeze as freeze,
)


@pytest.fixture(scope="module")
def frozen_inputs() -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load only the three outcome-blind parent freeze catalogues."""

    return freeze.load_source_catalogues()


def _recent_routes(
    independent: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    news = freeze._normalise_recent_family(
        independent,
        family=freeze.NEWS_FAMILY,
        route_id="live_news_activity",
    )
    web = freeze._normalise_recent_family(
        independent,
        family=freeze.WEB_FAMILY,
        route_id="live_web_activity",
    )
    union = freeze.build_recent_union(independent)
    _, _, overlap = freeze.choose_overlap_window(independent)
    return news, web, union, overlap


def _all_routes(
    independent: pd.DataFrame,
    confluence: pd.DataFrame,
    cross_asset: pd.DataFrame,
) -> pd.DataFrame:
    news, web, union, overlap = _recent_routes(independent)
    historical = freeze.build_historical_routes(confluence, cross_asset)
    return freeze.add_route_event_ids(
        pd.concat([news, web, union, overlap, historical], ignore_index=True)
    )


def test_ready_media_filter_has_expected_recent_partition_counts(
    frozen_inputs: tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    _, independent, _, _ = frozen_inputs

    ready = freeze.ready_media_events(independent)
    counts = ready.groupby(["family", "whole_event_partition"]).size().to_dict()

    assert counts == {
        (freeze.NEWS_FAMILY, "development_2026_06_01_to_07_15"): 27,
        (freeze.NEWS_FAMILY, "validation_2026_07_16_to_08_30"): 33,
        (freeze.WEB_FAMILY, "development_2026_06_01_to_07_15"): 17,
        (freeze.WEB_FAMILY, "validation_2026_07_16_to_08_30"): 30,
    }
    assert len(independent) == 359
    assert len(ready) == 107
    media = independent["family"].isin([freeze.NEWS_FAMILY, freeze.WEB_FAMILY])
    not_ready_media = independent.loc[
        media & ~independent.index.isin(ready.index), "coverage_ready"
    ]
    assert len(not_ready_media) == 5
    assert not_ready_media.astype(str).str.lower().eq("false").all()


def test_ready_media_filter_is_case_insensitive_and_requires_the_field() -> None:
    source = pd.DataFrame(
        {
            "family": [freeze.NEWS_FAMILY, freeze.WEB_FAMILY, "other"],
            "coverage_ready": ["TRUE", "false", True],
        }
    )

    ready = freeze.ready_media_events(source)

    assert ready["family"].tolist() == [freeze.NEWS_FAMILY]
    with pytest.raises(ValueError, match="coverage-ready field"):
        freeze.ready_media_events(source.drop(columns="coverage_ready"))


def test_cooldown_keeps_exact_24_hour_boundary_and_suppresses_inside_boundary() -> None:
    anchor = pd.Timestamp("2026-07-01T00:00:00Z")
    frame = pd.DataFrame(
        {
            "anchor_utc": [
                anchor + pd.Timedelta(hours=24),
                anchor,
                anchor + pd.Timedelta(hours=24, seconds=1),
            ],
            "value": ["exact-boundary", "first", "inside-boundary"],
        }
    )

    selected = freeze._apply_cooldown(frame, hours=24)

    assert selected["value"].tolist() == ["first", "exact-boundary"]
    assert selected["anchor_utc"].diff().dropna().min() >= pd.Timedelta(hours=24)


def test_overlap_window_selection_is_source_only_and_chooses_smallest_eligible_window(
    frozen_inputs: tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    _, independent, _, _ = frozen_inputs

    selected_window, diagnostics, selected = freeze.choose_overlap_window(independent)
    diagnostic_counts = {
        int(row.overlap_window_hours): {
            row.whole_event_partition: int(row.event_count_after_24h_cooldown)
            for row in diagnostics.loc[
                diagnostics["overlap_window_hours"].eq(row.overlap_window_hours)
            ].itertuples()
        }
        for row in diagnostics.itertuples()
    }

    assert selected_window == 8
    assert diagnostic_counts == {
        1: {
            "development_2026_06_01_to_07_15": 1,
            "validation_2026_07_16_to_08_30": 8,
        },
        2: {
            "development_2026_06_01_to_07_15": 1,
            "validation_2026_07_16_to_08_30": 11,
        },
        4: {
            "development_2026_06_01_to_07_15": 9,
            "validation_2026_07_16_to_08_30": 19,
        },
        8: {
            "development_2026_06_01_to_07_15": 15,
            "validation_2026_07_16_to_08_30": 27,
        },
        24: {
            "development_2026_06_01_to_07_15": 17,
            "validation_2026_07_16_to_08_30": 30,
        },
    }
    assert len(selected) == 42
    assert selected["whole_event_partition"].value_counts().to_dict() == {
        "development_2026_06_01_to_07_15": 15,
        "validation_2026_07_16_to_08_30": 27,
    }
    assert "outcomes_used_for_selection" not in diagnostics.columns


def test_ready_union_has_expected_counts_and_unique_component_timestamps(
    frozen_inputs: tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    _, independent, _, _ = frozen_inputs

    union = freeze.build_recent_union(independent)

    assert len(union) == 61
    assert union["whole_event_partition"].value_counts().to_dict() == {
        "development_2026_06_01_to_07_15": 27,
        "validation_2026_07_16_to_08_30": 34,
    }
    assert union["anchor_utc"].is_unique
    assert union["component_event_ids"].str.len().gt(0).all()
    assert union["component_routes"].str.contains("live_").all()


def test_historical_route_counts_exclude_current_diagnostic_partition(
    frozen_inputs: tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    _, _, confluence, cross_asset = frozen_inputs

    historical = freeze.build_historical_routes(confluence, cross_asset)
    counts = historical.groupby(["route_id", "whole_event_partition"]).size().to_dict()

    assert counts == {
        ("two_plus_families", "development_2021_2023"): 74,
        ("two_plus_families", "internal_validation_2024_2025"): 53,
        ("three_plus_families", "development_2021_2023"): 22,
        ("three_plus_families", "internal_validation_2024_2025"): 19,
        ("two_plus_source_groups", "development_2021_2023"): 61,
        ("two_plus_source_groups", "internal_validation_2024_2025"): 40,
        ("market_fear", "development_2021_2023"): 26,
        ("market_fear", "internal_validation_2024_2025"): 20,
        ("technology_equities", "development_2021_2023"): 23,
        ("technology_equities", "internal_validation_2024_2025"): 25,
        ("broad_us_dollar", "development_2021_2023"): 38,
        ("broad_us_dollar", "internal_validation_2024_2025"): 16,
    }
    assert not historical["whole_event_partition"].eq(
        "current_diagnostic_2026_pre_freeze"
    ).any()


def test_route_event_ids_are_unique_across_all_active_and_parked_routes(
    frozen_inputs: tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    _, independent, confluence, cross_asset = frozen_inputs

    routes = _all_routes(independent, confluence, cross_asset)

    assert routes["route_event_id"].is_unique
    assert routes["route_event_id"].str.len().gt(0).all()
    assert routes.groupby("route_id")["route_event_id"].nunique().eq(
        routes.groupby("route_id").size()
    ).all()


def test_coverage_gates_routes_missing_ten_events_in_each_partition() -> None:
    rows = []
    for route_id, development_count, validation_count in (
        ("ready_route", 10, 10),
        ("sparse_route", 10, 9),
    ):
        for index in range(development_count):
            rows.append(
                {
                    "analysis_group": "recent_live_media",
                    "route_id": route_id,
                    "route_label": route_id,
                    "whole_event_partition": "development_2026_06_01_to_07_15",
                    "anchor_utc": pd.Timestamp("2026-06-01T00:00:00Z")
                    + pd.Timedelta(hours=index),
                }
            )
        for index in range(validation_count):
            rows.append(
                {
                    "analysis_group": "recent_live_media",
                    "route_id": route_id,
                    "route_label": route_id,
                    "whole_event_partition": "validation_2026_07_16_to_08_30",
                    "anchor_utc": pd.Timestamp("2026-07-16T00:00:00Z")
                    + pd.Timedelta(hours=index),
                }
            )

    routes = freeze.add_route_event_ids(pd.DataFrame(rows))
    coverage = freeze.route_coverage(routes)
    eligible = coverage.set_index("route_id")["coverage_eligible"].to_dict()
    statuses = coverage.set_index("route_id")["status"].to_dict()

    assert eligible == {"ready_route": True, "sparse_route": False}
    assert statuses == {"ready_route": "frozen_ready", "sparse_route": "coverage_parked"}


def test_active_routes_receive_exactly_twelve_same_clock_controls_with_group_exclusions(
    frozen_inputs: tuple[dict[str, object], pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> None:
    _, independent, confluence, cross_asset = frozen_inputs

    routes = _all_routes(independent, confluence, cross_asset)
    coverage = freeze.route_coverage(routes)
    controls = freeze.prior_week_controls(routes, coverage)
    active_route_ids = set(
        coverage.loc[coverage["coverage_eligible"], "route_id"].astype(str)
    )
    active_event_ids = set(
        routes.loc[routes["route_id"].isin(active_route_ids), "route_event_id"]
    )

    assert set(controls["route_event_id"]) == active_event_ids
    assert controls.groupby("route_event_id").size().eq(12).all()
    assert controls.groupby("route_event_id")["control_rank"].apply(
        lambda values: values.tolist() == list(range(1, 13))
    ).all()
    assert controls["control_type"].eq("same_weekday_hour_prior_week").all()

    recent_anchors = tuple(
        pd.Timestamp(value)
        for value in routes.loc[
            routes["analysis_group"].eq("recent_live_media"), "anchor_utc"
        ].unique()
    )
    for row in controls.itertuples(index=False):
        event_anchor = pd.Timestamp(row.event_anchor_utc)
        control_anchor = pd.Timestamp(row.control_anchor_utc)
        assert event_anchor.weekday() == control_anchor.weekday()
        assert event_anchor.hour == control_anchor.hour
        assert control_anchor < event_anchor
        exclusion = (
            freeze.RECENT_CONTROL_EXCLUSION_HOURS
            if row.analysis_group == "recent_live_media"
            else freeze.HISTORICAL_CONTROL_EXCLUSION_HOURS
        )
        blocked = (
            recent_anchors
            if row.analysis_group == "recent_live_media"
            else tuple(
                pd.Timestamp(value)
                for value in routes.loc[
                    routes["route_id"].eq(row.route_id)
                    & routes["analysis_group"].eq("historical_context"),
                    "anchor_utc",
                ].unique()
            )
        )
        assert all(
            abs(control_anchor - blocked_anchor) > pd.Timedelta(hours=exclusion)
            for blocked_anchor in blocked
        )
        assert row.exclusion_hours == exclusion


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("status", "not_terminal", "Parent freeze is not terminal"),
        ("outcomes_read", True, "outcomes"),
    ],
)
def test_load_parent_rejects_bad_status_or_outcome_flag(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text("event_id\nsynthetic\n", encoding="utf-8")
    result_path = tmp_path / "result.json"
    payload: dict[str, object] = {
        "status": "expected_status",
        "outcomes_read": False,
        "artifacts": {
            "catalog": {
                "path": str(catalog_path.resolve()),
                "sha256": freeze.g0.sha256_file(catalog_path),
            }
        },
    }
    payload[field] = value
    result_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        freeze._load_parent(
            result_path,
            expected_status="expected_status",
            artifact_name="catalog",
            artifact_path=catalog_path,
        )


def test_load_parent_rejects_catalogue_hash_drift(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.csv"
    catalog_path.write_text("event_id\nsynthetic\n", encoding="utf-8")
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "status": "expected_status",
                "outcomes_read": False,
                "artifacts": {
                    "catalog": {
                        "path": str(catalog_path.resolve()),
                        "sha256": "not-the-current-hash",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="hash changed"):
        freeze._load_parent(
            result_path,
            expected_status="expected_status",
            artifact_name="catalog",
            artifact_path=catalog_path,
        )


def test_execute_writes_terminal_outcome_blind_freeze_only_under_tmp_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "event_confirmation_breadth"
    monkeypatch.setattr(freeze, "OUTPUT_ROOT", output_root)

    result = freeze.execute(overwrite=True)

    assert result["status"] == "completed_event_confirmation_breadth_freeze"
    assert result["outcomes_read"] is False
    assert result["selected_media_overlap_hours"] == 8
    assert Path(result["artifacts"]["routes"]["path"]).is_relative_to(output_root)

    freeze_path = Path(result["artifacts"]["freeze"]["path"])
    freeze_document = json.loads(freeze_path.read_text(encoding="utf-8"))
    assert freeze_document["status"] == (
        "frozen_event_confirmation_breadth_before_market_outcomes"
    )
    assert freeze_document["outcomes_read"] is False

    for artifact in result["artifacts"].values():
        artifact_path = Path(artifact["path"])
        assert artifact_path.is_file()
        assert freeze.g0.sha256_file(artifact_path) == artifact["sha256"]
        assert artifact_path.is_relative_to(output_root)
