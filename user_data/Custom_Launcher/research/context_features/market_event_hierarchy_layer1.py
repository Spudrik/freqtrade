"""Freeze honest event-history definitions before reading market outcomes.

The first event-hierarchy layer is deliberately outcome blind.  It extracts the
complete regular FOMC family from the Federal Reserve calendar, audits the old
hand-picked event list, and freezes the clocks, cohorts, controls, and evidence
boundaries needed by later market-response tests.
"""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import hashlib
import json
import re
import sys
from collections.abc import Sequence
from datetime import datetime
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
USER_DATA_DIR = REPO_ROOT / "user_data"
CONFIG_DIR = USER_DATA_DIR / "Custom_Launcher" / "research" / "config"
LEGACY_EVENTS_PATH = CONFIG_DIR / "known_btc_event_windows.json"
SOURCE_COVERAGE_RESULT = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "reports"
    / "source_coverage_audit_event_hierarchy_layer1_20260903a.csv"
)
GENERATION0_MANIFEST = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "market_reaction_zones"
    / "generation0_manifest.json"
)
MEME_MANIFEST = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "market_reaction_zones"
    / "generation2_shared"
    / "g2_meme_reaction_manifest.json"
)
OUTPUT_ROOT = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "event_hierarchy"
    / "layer1_honest_history_20260903a"
)

FOMC_CALENDAR_URL = "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm"
INFORMATION_CUTOFF_UTC = "2026-09-03T00:00:00Z"
EXPECTED_FOMC_YEARS = tuple(range(2021, 2028))
EXPECTED_REGULAR_MEETINGS_PER_YEAR = 8
FED_TIMEZONE = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


class FomcCalendarParser(HTMLParser):
    """Extract calendar rows without depending on a third-party HTML parser."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.current_year: int | None = None
        self.in_heading = False
        self.heading_text: list[str] = []
        self.row_depth = 0
        self.field_depth = 0
        self.current_field: str | None = None
        self.row: dict[str, Any] | None = None
        self.rows: list[dict[str, Any]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if tag == "h4" and self.row is None:
            self.in_heading = True
            self.heading_text = []
        if tag == "div" and "fomc-meeting" in classes and self.row is None:
            self.row_depth = 1
            self.row = {"year": self.current_year, "month": [], "date": [], "links": []}
            return
        if self.row is None:
            return
        if tag == "div":
            self.row_depth += 1
            if "fomc-meeting__month" in classes:
                self.current_field = "month"
                self.field_depth = self.row_depth
            elif "fomc-meeting__date" in classes:
                self.current_field = "date"
                self.field_depth = self.row_depth
        if tag == "a" and attributes.get("href"):
            self.row["links"].append(str(attributes["href"]))

    def handle_endtag(self, tag: str) -> None:
        if tag == "h4" and self.in_heading:
            text = " ".join(self.heading_text)
            match = re.search(r"\b(20\d{2})\s+FOMC\s+Meetings\b", text)
            if match:
                self.current_year = int(match.group(1))
            self.in_heading = False
        if self.row is None or tag != "div":
            return
        if self.current_field is not None and self.row_depth == self.field_depth:
            self.current_field = None
            self.field_depth = 0
        self.row_depth -= 1
        if self.row_depth == 0:
            self.row["month"] = " ".join(self.row["month"]).strip()
            self.row["date"] = " ".join(self.row["date"]).strip()
            self.rows.append(self.row)
            self.row = None

    def handle_data(self, data: str) -> None:
        text = " ".join(data.split())
        if not text:
            return
        if self.in_heading:
            self.heading_text.append(text)
        if self.row is not None and self.current_field is not None:
            self.row[self.current_field].append(text)


def _attrs(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def _iso_utc(value: datetime | pd.Timestamp) -> str:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _month_number(value: str) -> int:
    final_month = value.strip().split("/")[-1]
    for pattern in ("%B", "%b"):
        try:
            return datetime.strptime(final_month, pattern).month
        except ValueError:
            continue
    raise ValueError(f"Unrecognized FOMC month: {value!r}")


def parse_fomc_calendar(html: str, *, cutoff_utc: str) -> list[dict[str, Any]]:
    """Return one record for every regular meeting in the official calendar."""

    parser = FomcCalendarParser()
    parser.feed(html)
    cutoff = pd.Timestamp(cutoff_utc)
    events: list[dict[str, Any]] = []
    for row in parser.rows:
        year = row["year"]
        if year not in EXPECTED_FOMC_YEARS or "notation vote" in row["date"].casefold():
            continue
        days = [int(value) for value in re.findall(r"\d+", row["date"])]
        if not days:
            raise ValueError(f"FOMC row has no decision day: {row}")
        decision_date = datetime(year, _month_number(row["month"]), days[-1])
        local_release = decision_date.replace(hour=14, tzinfo=FED_TIMEZONE)
        release_utc = pd.Timestamp(local_release.astimezone(UTC))
        statement_urls = [
            urljoin(FOMC_CALENDAR_URL, href)
            for href in row["links"]
            if re.search(r"/newsevents/pressreleases/monetary20\d{6}a\.htm$", href)
        ]
        statement_url = statement_urls[0] if statement_urls else None
        occurred_before_cutoff = release_utc < cutoff
        if occurred_before_cutoff and statement_url is None:
            raise ValueError(f"Past regular FOMC meeting lacks an official statement: {row}")
        status = (
            "historical_official_release"
            if occurred_before_cutoff
            else "prospective_schedule_frozen_before_release"
        )
        events.append(
            {
                "event_id": f"fomc_{decision_date:%Y_%m_%d}",
                "event_family": "scheduled_policy_decision",
                "event_subfamily": "fomc_regular_decision",
                "decision_date_local": decision_date.date().isoformat(),
                "release_timezone": "America/New_York",
                "release_local_rule": "regular decision at 14:00 local time",
                "anchor_utc": _iso_utc(release_utc),
                "summary_of_economic_projections": "*" in row["date"],
                "information_status": status,
                "calendar_url": FOMC_CALENDAR_URL,
                "statement_url": statement_url,
                "historical_activity_test_eligible": occurred_before_cutoff,
                "historical_direction_test_eligible": False,
                "direction_limit": (
                    "A meeting occurrence has no sign. A historically timestamped market "
                    "expectation and release surprise have not been reconstructed."
                ),
                "anticipation_test_eligible": not occurred_before_cutoff,
                "whole_event_partition": whole_event_partition(release_utc),
            }
        )
    events.sort(key=lambda item: item["anchor_utc"])
    validate_complete_fomc_family(events)
    return events


def validate_complete_fomc_family(events: Sequence[dict[str, Any]]) -> None:
    counts: dict[int, int] = {}
    for event in events:
        year = int(str(event["decision_date_local"])[:4])
        counts[year] = counts.get(year, 0) + 1
    expected = {year: EXPECTED_REGULAR_MEETINGS_PER_YEAR for year in EXPECTED_FOMC_YEARS}
    if counts != expected:
        raise ValueError(f"Incomplete regular FOMC family: expected {expected}, got {counts}")
    ids = [str(item["event_id"]) for item in events]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate FOMC event identifiers detected.")


def whole_event_partition(anchor_utc: pd.Timestamp) -> str:
    year = pd.Timestamp(anchor_utc).year
    if year <= 2023:
        return "development_2021_2023"
    if year <= 2025:
        return "internal_validation_2024_2025"
    if pd.Timestamp(anchor_utc) < pd.Timestamp(INFORMATION_CUTOFF_UTC):
        return "current_diagnostic_2026_pre_freeze"
    return "prospective_untouched_after_freeze"


def audit_legacy_events(
    legacy_document: dict[str, Any], fomc_events: Sequence[dict[str, Any]]
) -> DataFrame:
    """Make the hindsight risk in the old remembered-event list explicit."""

    fomc_dates = {str(item["decision_date_local"]) for item in fomc_events}
    records: list[dict[str, Any]] = []
    for event in legacy_document.get("events", []):
        anchor = pd.Timestamp(event["anchor_utc"])
        source_urls = [str(url) for url in event.get("source_urls", [])]
        official_urls = [
            url
            for url in source_urls
            if any(
                domain in url.casefold()
                for domain in ("federalreserve.gov", "sec.gov", "fdic.gov")
            )
        ]
        exact_intraday_anchor = not (
            anchor.hour == 0 and anchor.minute == 0 and anchor.second == 0
        )
        reasons = [
            "The file explicitly selects remembered market-moving events.",
            "The expected_shape field describes the later move and is hindsight information.",
        ]
        if not exact_intraday_anchor:
            reasons.append("The midnight anchor does not establish the first tradable timestamp.")
        if not official_urls:
            reasons.append("No official primary-source release is attached.")
        overlaps_fomc = anchor.date().isoformat() in fomc_dates
        if overlaps_fomc:
            reasons.append(
                "A complete official FOMC-family record supersedes this selected example."
            )
        records.append(
            {
                "event_id": str(event["event_id"]),
                "name": str(event["name"]),
                "archetype": str(event["archetype"]),
                "anchor_utc": _iso_utc(anchor),
                "source_url_count": len(source_urls),
                "official_primary_source_count": len(official_urls),
                "exact_intraday_anchor": exact_intraday_anchor,
                "contains_hindsight_expected_shape": bool(event.get("expected_shape")),
                "overlaps_complete_fomc_family": overlaps_fomc,
                "evaluation_status": "discovery_only_hindsight_selected",
                "permitted_use": "source discovery and qualitative case review only",
                "not_permitted_use": "pass rates, model fitting, thresholds, or confirmation",
                "reasons": " ".join(reasons),
            }
        )
    return DataFrame.from_records(records).sort_values("anchor_utc", kind="stable")


def frozen_cohorts() -> dict[str, Any]:
    normal = json.loads(GENERATION0_MANIFEST.read_text(encoding="utf-8"))["data"]["pairs"]
    memes = json.loads(MEME_MANIFEST.read_text(encoding="utf-8"))["data"]["pairs"]
    expected_normal = {
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        "BNB/USDT:USDT",
        "SOL/USDT:USDT",
        "XRP/USDT:USDT",
        "ADA/USDT:USDT",
        "DOGE/USDT:USDT",
        "TRX/USDT:USDT",
        "AVAX/USDT:USDT",
        "LINK/USDT:USDT",
    }
    if set(normal) != expected_normal or len(memes) != 10:
        raise ValueError("Previously frozen market cohorts drifted.")
    excluded_from_established = {
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        "DOGE/USDT:USDT",
    }
    return {
        "leader_candidates": ["BTC/USDT:USDT", "ETH/USDT:USDT", "broad_participation"],
        "established_alts": [
            pair for pair in normal if pair not in excluded_from_established
        ],
        "top_ten_traded_memes": memes,
        "meme_cohort_source": _attrs(MEME_MANIFEST),
        "normal_cohort_source": _attrs(GENERATION0_MANIFEST),
        "cohort_rule": (
            "Keep BTC separate. Report established alts and memes separately. A coherent "
            "smaller predeclared group may be a narrow lead; one coin is not general evidence."
        ),
    }


def layer1_definitions() -> dict[str, Any]:
    """Return the definitions that every later event layer must preserve."""

    return {
        "three_clocks": {
            "slow_background": "Trailing 30, 90, and 180 days, using only timestamp-safe sources.",
            "event_clock": (
                "Anticipation, official release/detection, confirmation, and correction."
            ),
            "local_market_clock": "First BTC, ETH, breadth, cohort, and coin-local response.",
        },
        "event_families": [
            {
                "id": "scheduled_policy_decision",
                "status": "evaluation_ready_for_activity",
                "first_family": "complete regular FOMC decisions, not selected movers",
                "signed_limit": (
                    "direction parked until expectation and surprise are reconstructable"
                ),
            },
            {
                "id": "unexpected_shock",
                "status": "activity_detection_only",
                "rule": (
                    "GDELT may locate activity spikes; independent sources must supply "
                    "story and sign."
                ),
            },
            {
                "id": "persistent_background_change",
                "status": "partly_ready",
                "rule": (
                    "OHLCV and audited broad activity are usable; semantic macro claims "
                    "need provenance."
                ),
            },
            {
                "id": "structurally_anticipated_crypto_event",
                "status": "coverage_parked",
                "rule": (
                    "Too few independent historic halvings; freeze future events prospectively."
                ),
            },
        ],
        "availability_rules": {
            "official_release": "available at its primary-source publication timestamp",
            "scheduled_anticipation": (
                "historically eligible only if schedule publication was archived; "
                "otherwise release-only"
            ),
            "aggregate_news": "activity/intensity only when story-level history is absent",
            "expectation_and_surprise": (
                "require timestamped pre-release expectation plus released value or decision"
            ),
            "market_features": (
                "each feature uses only complete candles strictly available by decision time"
            ),
            "live_collectors": "never query a running SQLite database; use an immutable snapshot",
        },
        "duplicate_handling": {
            "same_family_within_24h": (
                "one episode anchored at the first independently confirmed release"
            ),
            "different_families_with_overlapping_168h_windows": (
                "retain a multi-event flag; exclude from single-event attribution"
            ),
            "rumour_then_confirmation": (
                "keep phases inside one episode; do not count them as two successes"
            ),
            "whole_event_holdout": (
                "all rows and coins from an overlapping episode stay in one partition"
            ),
        },
        "leader_definition": {
            "candidates": ["BTC", "ETH", "broad participation"],
            "checkpoints_hours": [1, 2, 4, 8, 24],
            "primary_threshold": (
                "causal trailing-90-day 95th percentile of matching-horizon activity"
            ),
            "sensitivity_thresholds": [0.90, 0.99],
            "activity_components": ["absolute return", "high-low range", "relative volume"],
            "leader_rule": (
                "earliest candidate crossing its frozen threshold at least one completed 1h candle "
                "before the follower; equal checkpoint means simultaneous, not leadership"
            ),
            "no_final_move_rule": "the final event move cannot choose the candidate or threshold",
        },
        "ordinary_meme_sensitivity": {
            "primary_history_days": 90,
            "sensitivity_history_days": [30, 180],
            "driver": "BTC hourly return available before the event",
            "method": "causal intercept-and-slope least-squares fit, separately for each meme",
            "amplification": (
                "actual meme move minus its pre-estimated ordinary BTC-linked move, scaled by "
                "that coin's prior residual volatility"
            ),
            "minimum_primary_observations": 1080,
            "rule": "raw larger meme moves alone are not evidence of event amplification",
        },
        "event_time_grids_hours": {
            "scheduled_policy_decision": {
                "pre": [-168, -72, -24, -4, -1],
                "post": [1, 2, 4, 8, 24, 72, 168],
            },
            "unexpected_shock": {
                "pre": [-168, -72, -24, -4, -1],
                "post": [1, 4, 24, 72, 168],
                "targeted_1m_minutes": [5, 15, 30, 60],
            },
            "persistent_background_change": {"post": [24, 72, 168, 720]},
            "structurally_anticipated_crypto_event": {
                "pre": [-720, -168, -72, -24],
                "post": [24, 72, 168, 720],
            },
        },
        "post_event_ranges": {
            "scheduled_or_unexpected": {
                "formation_window": "event through event + 24h",
                "first_knowable": "event + 24h",
                "evaluation_window": "after +24h through +168h",
            },
            "structurally_anticipated": {
                "formation_window": "event +24h through +168h",
                "first_knowable": "event +168h",
                "evaluation_window": "after +168h through +720h",
            },
            "measurements": [
                "boundary reaction",
                "range residence",
                "time and distance to revisit",
                "maximum movement the wrong way",
                "permanent range failure",
            ],
        },
        "controls": [
            (
                "matched no-event time with same hour, weekday, prior-volatility band, "
                "and slow-background band"
            ),
            "high aggregate-news activity without an independently confirmed event",
            "false event times shifted by whole weeks while preserving weekday and hour",
            "source-missing periods kept separate rather than filled as quiet news",
            "ordinary market times outside every event exclusion window",
            "event, leader, group, and coin-local components alone for later combinations",
        ],
        "whole_event_partitions": [
            {"id": "development_2021_2023", "role": "definitions and candidate discovery"},
            {"id": "internal_validation_2024_2025", "role": "chronological internal validation"},
            {"id": "current_diagnostic_2026_pre_freeze", "role": "diagnostic only"},
            {"id": "prospective_untouched_after_freeze", "role": "untouched confirmation"},
        ],
        "layer2_decision_rules": {
            "acceptable_floor": 0.55,
            "main_target": 0.65,
            "minimum_claim": (
                "positive paired uplift over controls, at least 55% whole-event success, and no "
                "single event or coin supplying the result"
            ),
            "abstention": (
                "unsupported story, sign, source, or coin coverage is parked, not guessed"
            ),
            "separate_labels": [
                "activity only",
                "signed direction",
                "simultaneous description",
                "usable lead-lag",
                "transmission",
                "amplification",
                "local modifier",
                "post-event range",
            ],
        },
    }


def source_readiness() -> list[dict[str, str]]:
    return [
        {
            "source": "Federal Reserve regular FOMC calendar",
            "status": "ready_complete_family_for_release_activity",
            "allowed": "release timing, activity response, prospective anticipation after freeze",
            "not_allowed": "historical directional surprise without an expectation source",
        },
        {
            "source": "GDELT aggregate events",
            "status": "ready_for_activity_intensity_only",
            "allowed": "outcome-blind unusual-news detection and high-news controls",
            "not_allowed": "story identity, expected direction, or surprise sign",
        },
        {
            "source": "frozen global/crypto market columns",
            "status": "coverage_present_provenance_audit_required",
            "allowed": "coverage audit and later source-by-source provenance checks",
            "not_allowed": "named macro causal claims before provenance verification",
        },
        {
            "source": "historical orderbook aggregates",
            "status": "coverage_present_separate_semantics_test_required",
            "allowed": "later independently frozen disturbance tests in covered windows",
            "not_allowed": "assume orderbook predicts an event response",
        },
        {
            "source": "article, GKG, Trends, and ETF-flow history",
            "status": "coverage_parked_too_sparse",
            "allowed": "prospective collection and narrow covered-window diagnostics",
            "not_allowed": "broad historical event tests or missing-as-zero filling",
        },
        {
            "source": "CPI releases and expectations",
            "status": "prospective_family_pending_reproducible_history",
            "allowed": "freeze future official dates once captured",
            "not_allowed": (
                "historical surprise scoring until release and expectation history is frozen"
            ),
        },
    ]


def build_freeze_document(
    *, calendar_html: str, source_sha256: str, fetched_at_utc: str
) -> tuple[dict[str, Any], DataFrame, DataFrame]:
    events = parse_fomc_calendar(calendar_html, cutoff_utc=INFORMATION_CUTOFF_UTC)
    legacy = json.loads(LEGACY_EVENTS_PATH.read_text(encoding="utf-8"))
    legacy_audit = audit_legacy_events(legacy, events)
    event_frame = DataFrame.from_records(events)
    historical = int(event_frame["historical_activity_test_eligible"].sum())
    prospective = int((~event_frame["historical_activity_test_eligible"]).sum())
    freeze = {
        "schema_version": 1,
        "objective": "objective_02b_market_reaction_zone_discovery",
        "layer": 1,
        "status": "frozen_before_event_market_outcomes",
        "created_at_utc": g0.utc_now(),
        "information_cutoff_utc": INFORMATION_CUTOFF_UTC,
        "outcomes_read": False,
        "profit_used": False,
        "ordinary_timestamp_direction_authorized": False,
        "purpose": (
            "Freeze complete event families and causal definitions before testing whether events, "
            "leaders, groups, local context, or ranges relate to market behaviour."
        ),
        "official_event_catalog": {
            "family": "regular FOMC policy decisions",
            "selection_rule": "every regular calendar meeting from 2021 through 2027",
            "event_count": len(events),
            "historical_activity_eligible_count": historical,
            "prospective_untouched_count": prospective,
            "calendar_url": FOMC_CALENDAR_URL,
            "calendar_fetched_at_utc": fetched_at_utc,
            "calendar_response_sha256": source_sha256,
            "release_time_rule": "14:00 America/New_York converted with historical DST",
            "historical_direction_status": "parked_missing_timestamped_expectation_and_surprise",
        },
        "legacy_remembered_event_audit": {
            "source": _attrs(LEGACY_EVENTS_PATH),
            "events_audited": len(legacy_audit),
            "evaluation_ready_events": 0,
            "discovery_only_events": len(legacy_audit),
            "conclusion": (
                "The remembered list is useful for finding sources and examples but cannot "
                "estimate "
                "success because it was selected after knowing events moved markets."
            ),
        },
        "definitions": layer1_definitions(),
        "cohorts": frozen_cohorts(),
        "source_readiness": source_readiness(),
        "layer2_sibling_queue": [
            {"id": "event_to_market_activity", "status": "ready_fomc_and_activity_only_gdelt"},
            {"id": "event_to_signed_direction", "status": "coverage_parked_missing_surprise_sign"},
            {"id": "btc_eth_breadth_leadership", "status": "ready_for_ohlcv_test"},
            {"id": "leader_to_established_alts", "status": "ready_subject_to_coin_history"},
            {"id": "leader_to_memes", "status": "ready_subject_to_common_meme_history"},
            {"id": "meme_residual_amplification", "status": "ready_subject_to_training_history"},
            {
                "id": "slow_background_modification",
                "status": "partly_ready_ohlcv_and_activity_only",
            },
            {"id": "coin_local_context_description", "status": "ready_for_causal_ohlcv_levels"},
            {"id": "post_event_range_behaviour", "status": "ready_after_frozen_knowability_rule"},
        ],
        "source_contracts": {
            "analysis_script": _attrs(ANALYSIS_PATH),
            "source_coverage_audit": _attrs(SOURCE_COVERAGE_RESULT),
            "generation0_manifest": _attrs(GENERATION0_MANIFEST),
            "meme_manifest": _attrs(MEME_MANIFEST),
        },
    }
    return freeze, event_frame, legacy_audit


def render_report(freeze: dict[str, Any]) -> str:
    catalog = freeze["official_event_catalog"]
    legacy = freeze["legacy_remembered_event_audit"]
    rows = [
        "# Event Hierarchy Layer 1 - Honest History And Frozen Definitions",
        "",
        f"- Status: `{freeze['status']}`",
        f"- Information cutoff: `{freeze['information_cutoff_utc']}`",
        "- Market outcomes read: **No**",
        f"- Complete regular FOMC meetings: `{catalog['event_count']}`",
        f"- Historical release/activity cases: `{catalog['historical_activity_eligible_count']}`",
        f"- Prospective untouched cases: `{catalog['prospective_untouched_count']}`",
        "",
        "## What Was Closed",
        "",
        (
            f"All `{legacy['events_audited']}` events in the old remembered-mover list are now "
            "restricted to discovery and qualitative review. None may contribute to a pass rate."
        ),
        "",
        "## What Can Be Tested Next",
        "",
        "| Question | State | Plain meaning |",
        "| --- | --- | --- |",
    ]
    plain = {
        "event_to_market_activity": (
            "Do complete official events coincide with unusually busy markets?"
        ),
        "event_to_signed_direction": (
            "Direction cannot be judged without a valid expectation and surprise."
        ),
        "btc_eth_breadth_leadership": "Which part of the broad crypto market moves first, if any?",
        "leader_to_established_alts": (
            "Do established coins follow a genuinely earlier market move?"
        ),
        "leader_to_memes": "Do memes follow a genuinely earlier market move?",
        "meme_residual_amplification": "Do memes move more than their normal BTC-linked behaviour?",
        "slow_background_modification": "Does the prior market climate change event reactions?",
        "coin_local_context_description": (
            "Do nearby levels or local conditions change the response?"
        ),
        "post_event_range_behaviour": "Does price settle into a reusable range after the event?",
    }
    for item in freeze["layer2_sibling_queue"]:
        rows.append(f"| `{item['id']}` | `{item['status']}` | {plain[item['id']]} |")
    rows.extend(
        [
            "",
            "## Boundary",
            "",
            (
                "This freeze supports activity, timing, transmission, amplification, "
                "local-context, "
                "and range tests. It does not authorize profit optimization or a direction call at "
                "ordinary timestamps."
            ),
            "",
        ]
    )
    return "\n".join(rows)


def execute(*, calendar_html_path: Path | None, overwrite: bool) -> dict[str, Any]:
    result_path = OUTPUT_ROOT / "layer1_result.json"
    if result_path.is_file() and not overwrite:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed_event_hierarchy_layer1":
            raise ValueError("Existing Layer 1 result is not terminal.")
        return result

    if calendar_html_path is None:
        response = requests.get(FOMC_CALENDAR_URL, timeout=30)
        response.raise_for_status()
        calendar_html = response.text
        fetched_at = g0.utc_now()
        source_sha = hashlib.sha256(response.content).hexdigest()
    else:
        calendar_html = calendar_html_path.read_text(encoding="utf-8")
        fetched_at = g0.utc_now()
        source_sha = g0.sha256_file(calendar_html_path)

    freeze, events, legacy_audit = build_freeze_document(
        calendar_html=calendar_html,
        source_sha256=source_sha,
        fetched_at_utc=fetched_at,
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    events_path = OUTPUT_ROOT / "official_fomc_event_catalog.csv"
    legacy_path = OUTPUT_ROOT / "legacy_remembered_event_audit.csv"
    freeze_path = OUTPUT_ROOT / "layer1_freeze.json"
    report_path = OUTPUT_ROOT / "layer1_report.md"
    g0.atomic_write_csv(events, events_path)
    g0.atomic_write_csv(legacy_audit, legacy_path)
    g0.atomic_write_json(freeze, freeze_path)
    report_path.write_text(render_report(freeze), encoding="utf-8", newline="\n")
    result = {
        "schema_version": 1,
        "status": "completed_event_hierarchy_layer1",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "official_events": len(events),
        "historical_activity_events": int(events["historical_activity_test_eligible"].sum()),
        "prospective_untouched_events": int((~events["historical_activity_test_eligible"]).sum()),
        "legacy_events_restricted_to_discovery": len(legacy_audit),
        "layer2_ready_siblings": sum(
            not str(item["status"]).startswith("coverage_parked")
            for item in freeze["layer2_sibling_queue"]
        ),
        "layer2_coverage_parked_siblings": sum(
            str(item["status"]).startswith("coverage_parked")
            for item in freeze["layer2_sibling_queue"]
        ),
        "artifacts": {
            "freeze": _attrs(freeze_path),
            "event_catalog": _attrs(events_path),
            "legacy_audit": _attrs(legacy_path),
            "report": _attrs(report_path),
        },
    }
    g0.atomic_write_json(result, result_path)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--calendar-html", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.execute:
        print(
            json.dumps(
                {
                    "status": "ready_not_executed",
                    "output_root": str(OUTPUT_ROOT),
                    "information_cutoff_utc": INFORMATION_CUTOFF_UTC,
                    "outcomes_will_be_read": False,
                },
                indent=2,
            )
        )
        return 0
    print(
        json.dumps(
            execute(calendar_html_path=args.calendar_html, overwrite=args.overwrite),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
