# ruff: noqa: S101

from __future__ import annotations

import json

import pandas as pd

from user_data.Custom_Launcher.research.context_features import (
    market_event_hierarchy_layer1 as layer1,
)


def sample_calendar() -> str:
    rows = []
    for year in layer1.EXPECTED_FOMC_YEARS:
        for month, day in (
            ("January", 25),
            ("March", 15),
            ("May", 3),
            ("June", 14),
            ("July", 26),
            ("September", 20),
            ("November", 1),
            ("December", 13),
        ):
            anchor = f"{year}{pd.Timestamp(f'{year}-{month}-{day}').month:02d}{day:02d}"
            link = (
                f'<a href="/newsevents/pressreleases/monetary{anchor}a.htm">HTML</a>'
                if pd.Timestamp(f"{year}-{month}-{day}", tz="America/New_York")
                < pd.Timestamp(layer1.INFORMATION_CUTOFF_UTC)
                else ""
            )
            rows.append(
                '<div class="row fomc-meeting">'
                f'<div class="fomc-meeting__month"><strong>{month}</strong></div>'
                f'<div class="fomc-meeting__date">{day}</div><div>{link}</div></div>'
            )
        rows[-3] = rows[-3].replace(">20</div>", ">20*</div>")
    panels = []
    for offset, year in enumerate(layer1.EXPECTED_FOMC_YEARS):
        year_rows = rows[offset * 8 : (offset + 1) * 8]
        panels.append(f"<div><h4>{year} FOMC Meetings</h4>{''.join(year_rows)}</div>")
    return "<html><body>" + "".join(panels) + "</body></html>"


def test_parser_keeps_complete_family_and_historical_dst() -> None:
    events = layer1.parse_fomc_calendar(
        sample_calendar(), cutoff_utc=layer1.INFORMATION_CUTOFF_UTC
    )
    assert len(events) == 56
    january = next(item for item in events if item["event_id"] == "fomc_2021_01_25")
    june = next(item for item in events if item["event_id"] == "fomc_2021_06_14")
    september = next(item for item in events if item["event_id"] == "fomc_2021_09_20")
    assert january["anchor_utc"].endswith("19:00:00Z")
    assert june["anchor_utc"].endswith("18:00:00Z")
    assert september["summary_of_economic_projections"] is True


def test_parser_rejects_an_incomplete_family() -> None:
    html = sample_calendar().replace(
        "<h4>2027 FOMC Meetings</h4>", "<h4>2030 FOMC Meetings</h4>"
    )
    try:
        layer1.parse_fomc_calendar(html, cutoff_utc=layer1.INFORMATION_CUTOFF_UTC)
    except ValueError as exc:
        assert "Incomplete regular FOMC family" in str(exc)
    else:
        raise AssertionError("Incomplete FOMC family was accepted.")


def test_cross_month_meeting_uses_the_decision_month() -> None:
    assert layer1._month_number("Apr/May") == 5


def test_legacy_remembered_events_are_never_evaluation_rows() -> None:
    legacy = json.loads(layer1.LEGACY_EVENTS_PATH.read_text(encoding="utf-8"))
    events = layer1.parse_fomc_calendar(
        sample_calendar(), cutoff_utc=layer1.INFORMATION_CUTOFF_UTC
    )
    audit = layer1.audit_legacy_events(legacy, events)
    assert len(audit) == len(legacy["events"])
    assert set(audit["evaluation_status"]) == {"discovery_only_hindsight_selected"}
    assert audit["contains_hindsight_expected_shape"].all()


def test_freeze_contains_every_layer1_contract_without_outcomes() -> None:
    freeze, events, legacy = layer1.build_freeze_document(
        calendar_html=sample_calendar(),
        source_sha256="fixture-sha256",
        fetched_at_utc="2026-09-03T00:00:00Z",
    )
    assert freeze["status"] == "frozen_before_event_market_outcomes"
    assert freeze["outcomes_read"] is False
    assert len(events) == 56
    assert len(legacy) > 0
    assert len(freeze["definitions"]["controls"]) >= 5
    assert freeze["definitions"]["ordinary_meme_sensitivity"]["primary_history_days"] == 90
    assert freeze["cohorts"]["leader_candidates"] == [
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        "broad_participation",
    ]
    queue = {item["id"]: item["status"] for item in freeze["layer2_sibling_queue"]}
    assert queue["event_to_signed_direction"].startswith("coverage_parked")
    assert queue["post_event_range_behaviour"].startswith("ready")
