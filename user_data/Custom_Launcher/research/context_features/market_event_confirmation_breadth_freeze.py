"""Freeze source clocks for the breadth-first early-market-confirmation batch.

This module only reads already-frozen event/source catalogues.  It deliberately does
not read OHLCV, returns, reaction outcomes, or prior direct-review summaries.
"""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
EVENT_ROOT = (
    REPO_ROOT / "user_data" / "research_news_data" / "context_features" / "event_hierarchy"
)
INDEPENDENT_ROOT = EVENT_ROOT / "independent_context_sources_20260904a"
CONFLUENCE_ROOT = EVENT_ROOT / "event_confluence_20260904a"
CROSS_ASSET_ROOT = EVENT_ROOT / "cross_asset_relevance_20260904a"
OUTPUT_ROOT = EVENT_ROOT / "event_confirmation_breadth_20260905a"

INDEPENDENT_RESULT = INDEPENDENT_ROOT / "independent_context_freeze_result.json"
INDEPENDENT_EVENTS = INDEPENDENT_ROOT / "source_event_catalog.csv"
CONFLUENCE_RESULT = CONFLUENCE_ROOT / "event_confluence_freeze_result.json"
CONFLUENCE_EVENTS = CONFLUENCE_ROOT / "event_confluence_route_catalog.csv"
CROSS_ASSET_RESULT = CROSS_ASSET_ROOT / "cross_asset_freeze_result.json"
CROSS_ASSET_EVENTS = CROSS_ASSET_ROOT / "cross_asset_event_catalog.csv"
COHORT_FREEZE = (
    EVENT_ROOT
    / "layer2_individual_links_20260903a"
    / "layer2_freeze.json"
)
MARKET_METRIC_HELPER = (
    REPO_ROOT
    / "user_data"
    / "Custom_Launcher"
    / "research"
    / "context_features"
    / "market_event_hierarchy_layer2_direct.py"
)

RECENT_PARTITIONS = (
    "development_2026_06_01_to_07_15",
    "validation_2026_07_16_to_08_30",
)
HISTORICAL_PARTITIONS = (
    "development_2021_2023",
    "internal_validation_2024_2025",
)
PARTITIONS_BY_GROUP = {
    "recent_live_media": RECENT_PARTITIONS,
    "historical_context": HISTORICAL_PARTITIONS,
}

NEWS_FAMILY = "live_news_activity_spike"
WEB_FAMILY = "live_web_activity_spike"
RECENT_COOLDOWN_HOURS = 24
RECENT_CONTROL_EXCLUSION_HOURS = 24
HISTORICAL_CONTROL_EXCLUSION_HOURS = 72
CONTROL_SEARCH_WEEKS = 52
CONTROL_COUNT = 12
MINIMUM_EVENTS_PER_PARTITION = 10
OVERLAP_WINDOWS_HOURS = (1, 2, 4, 8, 24)

CONFLUENCE_ROUTES = (
    "two_plus_families",
    "three_plus_families",
    "two_plus_source_groups",
)
CROSS_ASSET_FAMILIES = (
    "market_fear",
    "technology_equities",
    "broad_us_dollar",
)

ROUTE_LABELS = {
    "live_news_activity": "Live-news activity clock",
    "live_web_activity": "Live-web and announcement activity clock",
    "live_news_or_web_union": "Either live-news or live-web activity clock",
    "live_news_and_web_overlap": "Both live-news and live-web clocks within the selected window",
    "two_plus_families": "At least two recent historical event families",
    "three_plus_families": "At least three recent historical event families",
    "two_plus_source_groups": "At least two historical source types",
    "market_fear": "Large US market-fear change",
    "technology_equities": "Large US technology-equity change",
    "broad_us_dollar": "Large broad-US-dollar change",
}


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def parse_time(frame: DataFrame, column: str) -> None:
    frame[column] = pd.to_datetime(
        frame[column], utc=True, errors="raise", format="mixed"
    )


def _truthy(values: pd.Series) -> pd.Series:
    return values.astype(str).str.lower().eq("true")


def ready_media_events(source: DataFrame) -> DataFrame:
    selected = source.loc[source["family"].isin([NEWS_FAMILY, WEB_FAMILY])].copy()
    if "coverage_ready" not in selected:
        raise ValueError("Independent media catalogue has no coverage-ready field.")
    return selected.loc[_truthy(selected["coverage_ready"])].copy()


def _load_parent(
    result_path: Path,
    *,
    expected_status: str,
    artifact_name: str,
    artifact_path: Path,
) -> tuple[dict[str, Any], DataFrame]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != expected_status:
        raise ValueError(f"Parent freeze is not terminal: {result_path}")
    if result.get("outcomes_read"):
        raise ValueError(f"Parent freeze unexpectedly read outcomes: {result_path}")
    contract = result["artifacts"][artifact_name]
    if Path(contract["path"]).resolve() != artifact_path.resolve():
        raise ValueError(f"Parent artifact path changed: {artifact_path}")
    if contract["sha256"] != g0.sha256_file(artifact_path):
        raise ValueError(f"Parent artifact hash changed: {artifact_path}")
    return result, pd.read_csv(artifact_path)


def load_source_catalogues() -> tuple[dict[str, Any], DataFrame, DataFrame, DataFrame]:
    independent_result, independent = _load_parent(
        INDEPENDENT_RESULT,
        expected_status="completed_independent_context_source_freeze",
        artifact_name="events",
        artifact_path=INDEPENDENT_EVENTS,
    )
    confluence_result, confluence = _load_parent(
        CONFLUENCE_RESULT,
        expected_status="completed_event_confluence_freeze",
        artifact_name="catalog",
        artifact_path=CONFLUENCE_EVENTS,
    )
    cross_asset_result, cross_asset = _load_parent(
        CROSS_ASSET_RESULT,
        expected_status="completed_cross_asset_relevance_freeze",
        artifact_name="catalog",
        artifact_path=CROSS_ASSET_EVENTS,
    )
    parse_time(independent, "anchor_utc")
    parse_time(confluence, "anchor_utc")
    parse_time(cross_asset, "anchor_utc")
    parents = {
        "independent": independent_result,
        "confluence": confluence_result,
        "cross_asset": cross_asset_result,
    }
    return parents, independent, confluence, cross_asset


def _apply_cooldown(frame: DataFrame, hours: int) -> DataFrame:
    kept: list[int] = []
    last: pd.Timestamp | None = None
    cooldown = pd.Timedelta(hours=hours)
    for index, row in frame.sort_values("anchor_utc", kind="stable").iterrows():
        anchor = pd.Timestamp(row["anchor_utc"])
        if last is None or anchor - last >= cooldown:
            kept.append(index)
            last = anchor
    return frame.loc[kept].sort_values("anchor_utc", kind="stable").reset_index(drop=True)


def _normalise_recent_family(
    source: DataFrame, *, family: str, route_id: str
) -> DataFrame:
    selected = ready_media_events(source)
    selected = selected.loc[selected["family"].eq(family)].copy()
    selected["route_id"] = route_id
    selected["route_label"] = ROUTE_LABELS[route_id]
    selected["analysis_group"] = "recent_live_media"
    selected["source_route"] = family
    selected["component_event_ids"] = selected["event_id"].astype(str)
    selected["component_routes"] = family
    selected["overlap_window_hours"] = pd.NA
    return selected[
        [
            "anchor_utc",
            "whole_event_partition",
            "route_id",
            "route_label",
            "analysis_group",
            "source_route",
            "component_event_ids",
            "component_routes",
            "overlap_window_hours",
        ]
    ]


def build_recent_union(source: DataFrame) -> DataFrame:
    selected = ready_media_events(source)[[
        "anchor_utc",
        "whole_event_partition",
        "family",
        "event_id",
    ]].copy()
    collapsed = selected.groupby(
        ["anchor_utc", "whole_event_partition"], as_index=False, sort=True
    ).agg(
        component_event_ids=("event_id", lambda values: ";".join(sorted(map(str, values)))),
        component_routes=("family", lambda values: ";".join(sorted(set(map(str, values))))),
    )
    collapsed = _apply_cooldown(collapsed, RECENT_COOLDOWN_HOURS)
    collapsed["route_id"] = "live_news_or_web_union"
    collapsed["route_label"] = ROUTE_LABELS["live_news_or_web_union"]
    collapsed["analysis_group"] = "recent_live_media"
    collapsed["source_route"] = "news_or_web"
    collapsed["overlap_window_hours"] = pd.NA
    return collapsed[
        [
            "anchor_utc",
            "whole_event_partition",
            "route_id",
            "route_label",
            "analysis_group",
            "source_route",
            "component_event_ids",
            "component_routes",
            "overlap_window_hours",
        ]
    ]


def build_recent_overlap(source: DataFrame, hours: int) -> DataFrame:
    selected = ready_media_events(source)[[
        "anchor_utc",
        "whole_event_partition",
        "family",
        "event_id",
    ]].copy().sort_values("anchor_utc", kind="stable")
    records: list[dict[str, Any]] = []
    window = pd.Timedelta(hours=hours)
    for event in selected.itertuples(index=False):
        earlier_other = selected.loc[
            selected["family"].ne(event.family)
            & selected["anchor_utc"].le(event.anchor_utc)
            & selected["anchor_utc"].ge(pd.Timestamp(event.anchor_utc) - window)
        ].sort_values("anchor_utc", ascending=False, kind="stable")
        if earlier_other.empty:
            continue
        other = earlier_other.iloc[0]
        records.append(
            {
                "anchor_utc": pd.Timestamp(event.anchor_utc),
                "whole_event_partition": event.whole_event_partition,
                "component_event_ids": ";".join(
                    sorted([str(event.event_id), str(other["event_id"])])
                ),
                "component_routes": ";".join(sorted([str(event.family), str(other["family"])])),
            }
        )
    if not records:
        return DataFrame(
            columns=[
                "anchor_utc",
                "whole_event_partition",
                "route_id",
                "route_label",
                "analysis_group",
                "source_route",
                "component_event_ids",
                "component_routes",
                "overlap_window_hours",
            ]
        )
    overlap = DataFrame.from_records(records)
    overlap = overlap.sort_values("anchor_utc", kind="stable").drop_duplicates(
        ["anchor_utc", "whole_event_partition"], keep="last"
    )
    overlap = _apply_cooldown(overlap, RECENT_COOLDOWN_HOURS)
    overlap["route_id"] = "live_news_and_web_overlap"
    overlap["route_label"] = ROUTE_LABELS["live_news_and_web_overlap"]
    overlap["analysis_group"] = "recent_live_media"
    overlap["source_route"] = "news_and_web"
    overlap["overlap_window_hours"] = hours
    return overlap[
        [
            "anchor_utc",
            "whole_event_partition",
            "route_id",
            "route_label",
            "analysis_group",
            "source_route",
            "component_event_ids",
            "component_routes",
            "overlap_window_hours",
        ]
    ].reset_index(drop=True)


def _partition_counts(frame: DataFrame, route_id: str) -> dict[str, int]:
    selected = frame.loc[frame["route_id"].eq(route_id)]
    return {
        partition: int(selected["whole_event_partition"].eq(partition).sum())
        for partition in PARTITIONS_BY_GROUP[str(selected["analysis_group"].iloc[0])]
    }


def choose_overlap_window(source: DataFrame) -> tuple[int, DataFrame]:
    diagnostics: list[dict[str, Any]] = []
    selected_window: int | None = None
    selected_frame: DataFrame | None = None
    for hours in OVERLAP_WINDOWS_HOURS:
        frame = build_recent_overlap(source, hours)
        counts = {
            partition: int(frame["whole_event_partition"].eq(partition).sum())
            for partition in RECENT_PARTITIONS
        }
        eligible = all(
            count >= MINIMUM_EVENTS_PER_PARTITION for count in counts.values()
        )
        diagnostics.extend(
            {
                "route_id": "live_news_and_web_overlap",
                "overlap_window_hours": hours,
                "whole_event_partition": partition,
                "event_count_after_24h_cooldown": count,
                "coverage_eligible": eligible,
            }
            for partition, count in counts.items()
        )
        if selected_window is None and eligible:
            selected_window = hours
            selected_frame = frame
    if selected_window is None or selected_frame is None:
        selected_window = OVERLAP_WINDOWS_HOURS[-1]
        selected_frame = build_recent_overlap(source, selected_window)
    return selected_window, DataFrame.from_records(diagnostics), selected_frame


def build_historical_routes(confluence: DataFrame, cross_asset: DataFrame) -> DataFrame:
    confluence_selected = confluence.loc[
        confluence["route_id"].isin(CONFLUENCE_ROUTES)
        & confluence["whole_event_partition"].isin(HISTORICAL_PARTITIONS)
    ].copy()
    confluence_out = DataFrame(
        {
            "anchor_utc": confluence_selected["anchor_utc"],
            "whole_event_partition": confluence_selected["whole_event_partition"],
            "route_id": confluence_selected["route_id"],
            "route_label": confluence_selected["route_id"].map(ROUTE_LABELS),
            "analysis_group": "historical_context",
            "source_route": confluence_selected["route_id"],
            "component_event_ids": confluence_selected["component_event_ids"],
            "component_routes": confluence_selected["component_families"],
            "overlap_window_hours": pd.NA,
        }
    )
    cross_selected = cross_asset.loc[
        cross_asset["event_family"].isin(CROSS_ASSET_FAMILIES)
        & cross_asset["whole_event_partition"].isin(HISTORICAL_PARTITIONS)
    ].copy()
    cross_out = DataFrame(
        {
            "anchor_utc": cross_selected["anchor_utc"],
            "whole_event_partition": cross_selected["whole_event_partition"],
            "route_id": cross_selected["event_family"],
            "route_label": cross_selected["event_family"].map(ROUTE_LABELS),
            "analysis_group": "historical_context",
            "source_route": cross_selected["event_family"],
            "component_event_ids": cross_selected["event_id"],
            "component_routes": cross_selected["event_family"],
            "overlap_window_hours": pd.NA,
        }
    )
    return pd.concat([confluence_out, cross_out], ignore_index=True)


def add_route_event_ids(routes: DataFrame) -> DataFrame:
    output = routes.sort_values(
        ["analysis_group", "route_id", "anchor_utc"], kind="stable"
    ).reset_index(drop=True)
    output["route_event_id"] = [
        f"{route}_{pd.Timestamp(anchor).strftime('%Y%m%d_%H%M')}_{rank:03d}"
        for rank, (route, anchor) in enumerate(
            output[["route_id", "anchor_utc"]].itertuples(index=False, name=None),
            start=1,
        )
    ]
    if output["route_event_id"].duplicated().any():
        raise ValueError("Route event IDs are not unique.")
    return output


def route_coverage(routes: DataFrame) -> DataFrame:
    counts = routes.groupby(
        ["analysis_group", "route_id", "route_label", "whole_event_partition"],
        as_index=False,
        sort=True,
    ).agg(event_count=("route_event_id", "nunique"))
    eligible: dict[tuple[str, str], bool] = {}
    for (group, route_id), rows in counts.groupby(
        ["analysis_group", "route_id"], sort=False
    ):
        required = PARTITIONS_BY_GROUP[str(group)]
        indexed = rows.set_index("whole_event_partition")["event_count"]
        eligible[(str(group), str(route_id))] = all(
            int(indexed.get(partition, 0)) >= MINIMUM_EVENTS_PER_PARTITION
            for partition in required
        )
    counts["coverage_eligible"] = [
        eligible[(str(group), str(route))]
        for group, route in counts[["analysis_group", "route_id"]].itertuples(
            index=False, name=None
        )
    ]
    counts["status"] = counts["coverage_eligible"].map(
        {True: "frozen_ready", False: "coverage_parked"}
    )
    return counts


def prior_week_controls(routes: DataFrame, coverage: DataFrame) -> DataFrame:
    active_keys = {
        (str(row.analysis_group), str(row.route_id))
        for row in coverage.loc[coverage["coverage_eligible"]]
        .drop_duplicates(["analysis_group", "route_id"])
        .itertuples(index=False)
    }
    active = routes.loc[
        [
            (str(group), str(route)) in active_keys
            for group, route in routes[["analysis_group", "route_id"]].itertuples(
                index=False, name=None
            )
        ]
    ].copy()
    recent_blocked = tuple(
        pd.Timestamp(value)
        for value in routes.loc[
            routes["analysis_group"].eq("recent_live_media"), "anchor_utc"
        ].unique()
    )
    blocked_by_route = {
        (str(group), str(route)): tuple(
            pd.Timestamp(value) for value in rows["anchor_utc"].unique()
        )
        for (group, route), rows in active.groupby(
            ["analysis_group", "route_id"], sort=False
        )
    }
    records: list[dict[str, Any]] = []
    for event in active.itertuples(index=False):
        group = str(event.analysis_group)
        route_id = str(event.route_id)
        exclusion_hours = (
            RECENT_CONTROL_EXCLUSION_HOURS
            if group == "recent_live_media"
            else HISTORICAL_CONTROL_EXCLUSION_HOURS
        )
        blocked = (
            recent_blocked
            if group == "recent_live_media"
            else blocked_by_route[(group, route_id)]
        )
        exclusion = pd.Timedelta(hours=exclusion_hours)
        rank = 0
        for weeks in range(1, CONTROL_SEARCH_WEEKS + 1):
            anchor = pd.Timestamp(event.anchor_utc) - pd.Timedelta(weeks=weeks)
            if any(abs(anchor - other) <= exclusion for other in blocked):
                continue
            rank += 1
            records.append(
                {
                    "route_event_id": event.route_event_id,
                    "analysis_group": group,
                    "route_id": route_id,
                    "event_anchor_utc": event.anchor_utc,
                    "whole_event_partition": event.whole_event_partition,
                    "control_anchor_utc": anchor,
                    "control_rank": rank,
                    "control_type": "same_weekday_hour_prior_week",
                    "exclusion_hours": exclusion_hours,
                }
            )
            if rank >= CONTROL_COUNT:
                break
    return DataFrame.from_records(records)


def freeze_document(
    parents: Mapping[str, Any],
    routes: DataFrame,
    coverage: DataFrame,
    controls: DataFrame,
    overlap_window: int,
    overlap_diagnostics: DataFrame,
) -> dict[str, Any]:
    active_routes = coverage.loc[coverage["coverage_eligible"], "route_id"].unique()
    cohort_freeze = json.loads(COHORT_FREEZE.read_text(encoding="utf-8"))
    cohorts = cohort_freeze.get("cohorts")
    if not isinstance(cohorts, dict):
        raise ValueError("Frozen market cohorts are missing.")
    return {
        "schema_version": 1,
        "status": "frozen_event_confirmation_breadth_before_market_outcomes",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "plain_question": (
            "After an independently known news, web, wider-market, or multi-event "
            "activity clock fires, does the first hour of BTC, ETH, or broad crypto "
            "movement provide useful direction for the following hours?"
        ),
        "analysis_groups": {
            key: {"partitions": list(value)} for key, value in PARTITIONS_BY_GROUP.items()
        },
        "active_routes": sorted(map(str, active_routes)),
        "route_rows": len(routes),
        "control_rows": len(controls),
        "minimum_events_per_partition": MINIMUM_EVENTS_PER_PARTITION,
        "overlap_selection": {
            "candidate_windows_hours": list(OVERLAP_WINDOWS_HOURS),
            "selected_window_hours": overlap_window,
            "selection_rule": (
                "smallest source-only window retaining at least ten cooldown-separated "
                "events in each recent chronological partition"
            ),
            "outcomes_used_for_selection": False,
            "diagnostic_rows": len(overlap_diagnostics),
        },
        "confirmation_definitions": {
            "btc": "BTC first-hour activity >=1.25 and non-zero close direction",
            "eth": "ETH first-hour activity >=1.25 and non-zero close direction",
            "btc_eth_agreement": (
                "BTC and ETH both have first-hour activity >=1.25 and close in the same direction"
            ),
            "broad_market": (
                "at least six of BTC ETH BNB SOL XRP ADA TRX AVAX LINK are available, "
                "at least 60% move in the same direction, and median first-hour "
                "activity is at least 1.25"
            ),
        },
        "outcome_definition": (
            "Direction from the end of the first hour through total horizons 2h 4h 8h and 24h"
        ),
        "outcome_scopes_by_group": {
            "recent_live_media": ["btc", "eth", "established_alts", "top_ten_memes"],
            "historical_context": ["btc", "eth", "established_alts"],
        },
        "cohorts": cohorts,
        "component_controls": [
            "all event times using the observed first-hour sign",
            "the same first-hour confirmation at matched ordinary prior-week times",
            "recent pre-event direction",
            "confirmation state and sign rotated within each chronological partition",
        ],
        "retention_rule": [
            "at least ten confirmed event episodes and ten confirmed ordinary controls",
            "at least three confirmed events in each chronological partition",
            "continuation rate at least 55 percent and at least 50 percent in both partitions",
            "at least five percentage points above event-only and confirmation-only controls",
            "beats the 75th percentile of rotated confirmation/sign controls",
        ],
        "main_target": "65 percent is the main target; 55 percent is only a lead floor",
        "limits": [
            "These are conditional research leads, not trading rules.",
            "News and web activity clocks carry no assumed positive or negative sign.",
            "Nested union, overlap, and historical confluence routes count as dependent evidence.",
            (
                "The current top-ten meme cohort is tested only in the recent period; "
                "it is not backfilled into years when several members did not exist."
            ),
            "No level or cluster condition is opened until every sibling route is reviewed.",
            "No profit target or trade return is used as a research label.",
        ],
        "parent_freezes": {
            "independent": artifact(INDEPENDENT_RESULT),
            "confluence": artifact(CONFLUENCE_RESULT),
            "cross_asset": artifact(CROSS_ASSET_RESULT),
        },
        "parent_catalogues": {
            "independent": artifact(INDEPENDENT_EVENTS),
            "confluence": artifact(CONFLUENCE_EVENTS),
            "cross_asset": artifact(CROSS_ASSET_EVENTS),
        },
        "market_input_contracts": {
            "freeze_builder": artifact(ANALYSIS_PATH),
            "cohort_freeze": artifact(COHORT_FREEZE),
            "market_metric_helper": artifact(MARKET_METRIC_HELPER),
        },
        "parent_statuses": {
            key: value["status"] for key, value in parents.items()
        },
    }


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    result_path = OUTPUT_ROOT / "event_confirmation_breadth_freeze_result.json"
    if result_path.is_file() and not overwrite:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed_event_confirmation_breadth_freeze":
            raise ValueError("Existing breadth freeze is not terminal.")
        return result

    parents, independent, confluence, cross_asset = load_source_catalogues()
    recent_news = _normalise_recent_family(
        independent, family=NEWS_FAMILY, route_id="live_news_activity"
    )
    recent_web = _normalise_recent_family(
        independent, family=WEB_FAMILY, route_id="live_web_activity"
    )
    recent_union = build_recent_union(independent)
    selected_overlap, overlap_diagnostics, recent_overlap = choose_overlap_window(
        independent
    )
    historical = build_historical_routes(confluence, cross_asset)
    routes = add_route_event_ids(
        pd.concat(
            [recent_news, recent_web, recent_union, recent_overlap, historical],
            ignore_index=True,
        )
    )
    coverage = route_coverage(routes)
    controls = prior_week_controls(routes, coverage)
    freeze = freeze_document(
        parents,
        routes,
        coverage,
        controls,
        selected_overlap,
        overlap_diagnostics,
    )

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    paths = {
        "routes": OUTPUT_ROOT / "event_confirmation_route_catalog.csv",
        "controls": OUTPUT_ROOT / "event_confirmation_control_catalog.csv",
        "coverage": OUTPUT_ROOT / "event_confirmation_route_coverage.csv",
        "overlap_diagnostics": OUTPUT_ROOT / "media_overlap_coverage_diagnostics.csv",
        "freeze": OUTPUT_ROOT / "event_confirmation_breadth_freeze.json",
    }
    g0.atomic_write_csv(routes, paths["routes"])
    g0.atomic_write_csv(controls, paths["controls"])
    g0.atomic_write_csv(coverage, paths["coverage"])
    g0.atomic_write_csv(overlap_diagnostics, paths["overlap_diagnostics"])
    g0.atomic_write_json(freeze, paths["freeze"])
    result = {
        "schema_version": 1,
        "status": "completed_event_confirmation_breadth_freeze",
        "created_at_utc": g0.utc_now(),
        "outcomes_read": False,
        "selected_media_overlap_hours": selected_overlap,
        "route_rows": len(routes),
        "active_route_count": int(
            coverage.loc[coverage["coverage_eligible"], "route_id"].nunique()
        ),
        "artifacts": {name: artifact(path) for name, path in paths.items()},
    }
    g0.atomic_write_json(result, result_path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.execute:
        print(json.dumps({"status": "ready_not_executed", "outcomes_read": False}, indent=2))
        return 0
    print(json.dumps(execute(overwrite=args.overwrite), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
