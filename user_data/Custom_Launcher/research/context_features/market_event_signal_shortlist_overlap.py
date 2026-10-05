"""Measure overlap among five retrospective event-signal family representatives."""

from __future__ import annotations

# The repository root is inserted before local research imports.
# ruff: noqa: E402
import argparse
import itertools
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_freqai_breadth as breadth,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_freqai_breadth_freeze as breadth_freeze,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_signal_portfolio as portfolio,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
SOURCE_RUN_ID = "event_signal_portfolio_20260908b"
SOURCE_ROOT = breadth_freeze.OUTPUT_ROOT.parent / SOURCE_RUN_ID
SOURCE_FREEZE_PATH = SOURCE_ROOT / "event_signal_portfolio_freeze.json"
SOURCE_RESULT_PATH = SOURCE_ROOT / "event_signal_portfolio_result.json"
SOURCE_CALLS_PATH = SOURCE_ROOT / "event_signal_calls.csv"
SOURCE_BREADTH_MANIFEST = (
    breadth.RECORD_ROOT
    / portfolio.SOURCE_RUN_ID
    / "event_freqai_run_manifest.json"
)

RUN_ID = "event_signal_shortlist_20260909b"
SUPERSEDES_RUN_ID = "event_signal_shortlist_20260909a"
OUTPUT_ROOT = breadth_freeze.OUTPUT_ROOT.parent / RUN_ID
BULKY_ROOT = Path(
    "D:/FreqTradeStuffLargeData/research_outputs/event_hierarchy/"
    "event_signal_shortlist_20260909b"
)
FREEZE_PATH = OUTPUT_ROOT / "event_signal_shortlist_freeze.json"
MEMBERSHIP_PATH = BULKY_ROOT / "event_signal_shortlist_membership.parquet"
ROUTE_PERIOD_PATH = OUTPUT_ROOT / "event_signal_shortlist_route_periods.csv"
OVERLAP_PERIOD_PATH = OUTPUT_ROOT / "event_signal_shortlist_overlap_periods.csv"
OVERLAP_DECISIONS_PATH = OUTPUT_ROOT / "event_signal_shortlist_overlap_decisions.csv"
RESULT_PATH = OUTPUT_ROOT / "event_signal_shortlist_result.json"

ACTIVITY_TARGET = "&-meb_log_volume_ratio_h1"
HORIZON_HOURS = 1
ACTIVITY_METRIC = "log_volume_ratio"
DIRECTION_METRIC_FOR_DEDUPLICATION = "close_return_atr"
VALIDATION_PERIODS = portfolio.VALIDATION_PERIODS
MIN_INTERSECTION_EPISODES = 20
MIN_UNIQUE_ONLY_EPISODES = 10
MIN_COMPLEMENTARY_UPLIFT = 0.02
NEAR_DUPLICATE_JACCARD = 0.80

REPRESENTATIVES: tuple[dict[str, str], ...] = (
    {
        "family_id": "major_event_information",
        "route_id": "event_with_recent_confirmation",
        "plain_meaning": (
            "A known event and recent market behaviour jointly warn that the next hour "
            "may be unusually busy."
        ),
    },
    {
        "family_id": "background_and_market_leadership",
        "route_id": "event_with_background",
        "plain_meaning": (
            "A known event is interpreted using the slower market background."
        ),
    },
    {
        "family_id": "calculated_reaction_areas",
        "route_id": "event_with_level_cluster",
        "plain_meaning": (
            "A known event occurs while several independently calculated areas overlap "
            "near price."
        ),
    },
    {
        "family_id": "local_participation_and_pressure",
        "route_id": "recent_local_participation",
        "plain_meaning": (
            "Recent price, range, and volume behaviour warn that activity may continue."
        ),
    },
    {
        "family_id": "cross_asset_transmission_and_amplification",
        "route_id": "event_with_cross_market_confirmation",
        "plain_meaning": (
            "A known event coincides with activity already visible across established "
            "crypto markets."
        ),
    },
)

PRESERVED_OUTSIDE_THIS_COMPARISON = (
    "CPI short-release activity and provisional direction proxy",
    "SEC Item 2.02 short activity",
    "negative-background fade after an observed positive four-hour event move",
    "Bitcoin-dominance relative BTC-versus-meme rotation",
    "single calculated levels and general level-density traffic",
    "live news and web activity clocks",
    "all parked, negative, and unresolved rows in the approved hypothesis ledger",
)


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _episode_ids(values: Series) -> set[str]:
    identifiers: set[str] = set()
    for value in values.dropna():
        identifiers.update(str(item) for item in json.loads(str(value)))
    return identifiers


def _rate(frame: DataFrame) -> float:
    return float(frame["actual_reaction"].mean()) if len(frame) else np.nan


def build_freeze() -> dict[str, Any]:
    source_freeze = _load_json(SOURCE_FREEZE_PATH)
    source_result = _load_json(SOURCE_RESULT_PATH)
    if source_freeze.get("run_id") != SOURCE_RUN_ID:
        raise ValueError("Unexpected source signal-portfolio freeze.")
    if source_result.get("status") != (
        "completed_retrospective_signal_portfolio_prototype"
    ):
        raise ValueError("Source signal portfolio is incomplete.")
    source_routes = {str(route["route_id"]) for route in source_freeze["routes"]}
    representatives = {str(route["route_id"]) for route in REPRESENTATIVES}
    if not representatives.issubset(source_routes):
        raise ValueError("A shortlist route is absent from the source portfolio.")
    families = {str(route["family_id"]) for route in REPRESENTATIVES}
    if families != set(portfolio.FAMILY_IDS):
        raise ValueError("The shortlist must contain exactly one route per family.")
    return {
        "schema_version": 1,
        "run_id": RUN_ID,
        "supersedes_run_id": SUPERSEDES_RUN_ID,
        "supersession_reason": (
            "The first overlap review compared an intersection with each route's larger "
            "unfiltered call set. This revision also compares it with the same number of "
            "each route's strongest calls, preventing selectivity alone from being called "
            "complementary information."
        ),
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_shortlist_overlap_outcomes",
        "purpose": (
            "Reduce the broad portfolio to one comparable one-hour volume signal per "
            "family and determine whether the routes are distinct, redundant, or more "
            "useful when they overlap."
        ),
        "selection_status": (
            "Retrospective usability shortlist chosen from already inspected evidence; "
            "not an untouched confirmation set."
        ),
        "representatives": list(REPRESENTATIVES),
        "preserved_outside_this_comparison": list(PRESERVED_OUTSIDE_THIS_COMPARISON),
        "common_surface": {
            "activity_target": ACTIVITY_TARGET,
            "horizon_hours": HORIZON_HOURS,
            "validation_periods": list(VALIDATION_PERIODS),
            "coins": list(breadth_freeze.NORMAL_PAIRS),
        },
        "decision_rule": {
            "near_duplicate": (
                f"Jaccard overlap is at least {NEAR_DUPLICATE_JACCARD:.0%} in both years."
            ),
            "complementary_overlap": (
                "The intersection has at least 20 independent event episodes and reacts "
                "at least two percentage points more often than the same number of each "
                "route's strongest individual calls in both years."
            ),
            "distinct_coverage": (
                "Each route has at least ten independent events without the other in both "
                "years and is not a near duplicate."
            ),
            "all_other_cases": "Retain as unresolved rather than forcing a merge.",
        },
        "interpretation_boundary": {
            "profit_used": False,
            "direction_tested": False,
            "entry_or_exit_mapping_allowed": False,
            "shortlist_does_not_delete_other_evidence": True,
        },
        "storage": {
            "compact_outputs": str(OUTPUT_ROOT.resolve()),
            "detailed_membership": str(MEMBERSHIP_PATH.resolve()),
        },
        "source_contracts": {
            "analysis_script": artifact(ANALYSIS_PATH),
            "source_freeze": artifact(SOURCE_FREEZE_PATH),
            "source_result": artifact(SOURCE_RESULT_PATH),
            "source_calls": artifact(SOURCE_CALLS_PATH),
            "source_breadth_manifest": artifact(SOURCE_BREADTH_MANIFEST),
        },
    }


def freeze(*, overwrite: bool = False) -> dict[str, Any]:
    if FREEZE_PATH.is_file() and not overwrite:
        return _load_json(FREEZE_PATH)
    result = build_freeze()
    g0.atomic_write_json(result, FREEZE_PATH)
    return result


def verify_freeze() -> dict[str, Any]:
    frozen = _load_json(FREEZE_PATH)
    if frozen.get("status") != "frozen_before_shortlist_overlap_outcomes":
        raise ValueError("Shortlist overlap batch is not frozen.")
    for item in frozen["source_contracts"].values():
        path = Path(item["path"])
        if not path.is_file() or g0.sha256_file(path) != item["sha256"]:
            raise ValueError(f"Frozen shortlist source changed: {path}")
    return frozen


def load_activity_membership() -> DataFrame:
    use_columns = [
        "date",
        "pair",
        "period",
        "parent_episode_ids_json",
        "family_id",
        "route_id",
        "horizon_hours",
        "activity_metric",
        "direction_metric",
        "activity_score",
        "activity_threshold",
        "activity_signal",
    ]
    route_ids = {str(item["route_id"]) for item in REPRESENTATIVES}
    chunks: list[DataFrame] = []
    for chunk in pd.read_csv(
        SOURCE_CALLS_PATH,
        usecols=use_columns,
        chunksize=100_000,
    ):
        selected = chunk.loc[
            chunk["route_id"].isin(route_ids)
            & pd.to_numeric(chunk["horizon_hours"], errors="coerce").eq(HORIZON_HOURS)
            & chunk["activity_metric"].eq(ACTIVITY_METRIC)
            & chunk["direction_metric"].eq(DIRECTION_METRIC_FOR_DEDUPLICATION)
            & chunk["activity_signal"].astype(str).str.lower().eq("true")
        ].copy()
        if len(selected):
            chunks.append(selected)
    if not chunks:
        raise ValueError("No shortlisted activity calls were found.")
    output = pd.concat(chunks, ignore_index=True)
    output["date"] = pd.to_datetime(output["date"], utc=True, errors="raise")
    if output.duplicated(["route_id", "pair", "date"]).any():
        raise ValueError("Shortlisted activity membership contains duplicate keys.")
    return output


def build_universe(source_manifest: Mapping[str, Any]) -> DataFrame:
    actual = breadth.load_actual(source_manifest)
    actual = actual.loc[
        actual["sample_kind"].eq("actual_event")
        & actual["period"].isin(VALIDATION_PERIODS)
    ].copy()
    reference = pd.read_csv(source_manifest["storage"]["training_reference"])
    reference = reference.loc[reference["target"].eq(ACTIVITY_TARGET), [
        "pair",
        "period",
        "training_median",
    ]]
    actual = actual.merge(
        reference,
        on=["pair", "period"],
        how="left",
        validate="many_to_one",
    )
    actual["actual_reaction"] = pd.to_numeric(
        actual[ACTIVITY_TARGET], errors="coerce"
    ).gt(pd.to_numeric(actual["training_median"], errors="coerce"))
    required = [
        "date",
        "pair",
        "period",
        "sample_id",
        "parent_episode_ids_json",
        "actual_reaction",
    ]
    if actual[required].isna().any().any():
        raise ValueError("Shortlist universe contains missing required values.")
    return actual[required]


def build_membership(universe: DataFrame, calls: DataFrame) -> DataFrame:
    outputs: list[DataFrame] = []
    for representative in REPRESENTATIVES:
        route_id = str(representative["route_id"])
        route_calls = calls.loc[calls["route_id"].eq(route_id), [
            "pair",
            "date",
            "activity_score",
            "activity_threshold",
        ]]
        frame = universe.merge(
            route_calls,
            on=["pair", "date"],
            how="left",
            validate="one_to_one",
        )
        frame["signal_issued"] = frame["activity_score"].notna()
        frame["family_id"] = str(representative["family_id"])
        frame["route_id"] = route_id
        frame["plain_meaning"] = str(representative["plain_meaning"])
        outputs.append(frame)
    return pd.concat(outputs, ignore_index=True)


def score_routes(membership: DataFrame) -> DataFrame:
    keys = ["family_id", "route_id", "plain_meaning", "pair", "period"]
    rows: list[dict[str, Any]] = []
    for key, cell in membership.groupby(keys, observed=True, sort=False):
        calls = cell.loc[cell["signal_issued"]]
        non_calls = cell.loc[~cell["signal_issued"]]
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "eligible_events": len(cell),
                "eligible_unique_episodes": len(
                    _episode_ids(cell["parent_episode_ids_json"])
                ),
                "calls": len(calls),
                "call_unique_episodes": len(
                    _episode_ids(calls["parent_episode_ids_json"])
                ),
                "call_rate": len(calls) / max(1, len(cell)),
                "reaction_rate_calls": _rate(calls),
                "reaction_rate_non_calls": _rate(non_calls),
                "reaction_enrichment": _rate(calls) - _rate(non_calls),
            }
        )
    return DataFrame.from_records(rows)


def overlap_metrics(cell: DataFrame, route_a: str, route_b: str) -> dict[str, Any]:
    index_columns = [
        "date",
        "pair",
        "period",
        "sample_id",
        "parent_episode_ids_json",
        "actual_reaction",
    ]
    pivot = cell.pivot(
        index=index_columns,
        columns="route_id",
        values="signal_issued",
    )
    scores = cell.pivot(
        index=index_columns,
        columns="route_id",
        values="activity_score",
    )
    thresholds = cell.pivot(
        index=index_columns,
        columns="route_id",
        values="activity_threshold",
    )
    a = pivot[route_a].fillna(False).astype(bool)
    b = pivot[route_b].fillna(False).astype(bool)
    both = a & b
    union = a | b
    only_a = a & ~b
    only_b = b & ~a
    neither = ~a & ~b
    actual = Series(
        pivot.index.get_level_values("actual_reaction").astype(bool),
        index=pivot.index,
    )

    def episodes(mask: Series) -> int:
        values = Series(
            pivot.index.get_level_values("parent_episode_ids_json")[mask],
            dtype="object",
        )
        return len(_episode_ids(values))

    def reaction(mask: Series) -> float:
        return float(actual.loc[mask].mean()) if mask.any() else np.nan

    intersection_count = int(both.sum())

    def matched_strength_reaction(route_id: str, issued: Series) -> float:
        if not intersection_count:
            return np.nan
        margin = scores[route_id] - thresholds[route_id]
        strongest = margin.loc[issued].sort_values(
            ascending=False, kind="stable"
        ).head(intersection_count)
        return float(actual.loc[strongest.index].mean()) if len(strongest) else np.nan

    return {
        "eligible_events": len(pivot),
        "route_a_calls": int(a.sum()),
        "route_b_calls": int(b.sum()),
        "both_calls": int(both.sum()),
        "union_calls": int(union.sum()),
        "only_a_calls": int(only_a.sum()),
        "only_b_calls": int(only_b.sum()),
        "both_unique_episodes": episodes(both),
        "only_a_unique_episodes": episodes(only_a),
        "only_b_unique_episodes": episodes(only_b),
        "jaccard": float(both.sum() / union.sum()) if union.any() else np.nan,
        "call_agreement": float(a.eq(b).mean()),
        "route_a_reaction_rate": reaction(a),
        "route_b_reaction_rate": reaction(b),
        "route_a_matched_strength_reaction_rate": matched_strength_reaction(
            route_a, a
        ),
        "route_b_matched_strength_reaction_rate": matched_strength_reaction(
            route_b, b
        ),
        "both_reaction_rate": reaction(both),
        "union_reaction_rate": reaction(union),
        "only_a_reaction_rate": reaction(only_a),
        "only_b_reaction_rate": reaction(only_b),
        "neither_reaction_rate": reaction(neither),
    }


def score_overlaps(membership: DataFrame) -> DataFrame:
    routes = [str(item["route_id"]) for item in REPRESENTATIVES]
    rows: list[dict[str, Any]] = []
    for (pair, period), cell in membership.groupby(
        ["pair", "period"], observed=True, sort=False
    ):
        for route_a, route_b in itertools.combinations(routes, 2):
            rows.append(
                {
                    "pair": pair,
                    "period": period,
                    "route_a": route_a,
                    "route_b": route_b,
                    **overlap_metrics(cell, route_a, route_b),
                }
            )
    return DataFrame.from_records(rows)


OVERLAP_RATE_COLUMNS = (
    "jaccard",
    "call_agreement",
    "route_a_reaction_rate",
    "route_b_reaction_rate",
    "route_a_matched_strength_reaction_rate",
    "route_b_matched_strength_reaction_rate",
    "both_reaction_rate",
    "union_reaction_rate",
    "only_a_reaction_rate",
    "only_b_reaction_rate",
    "neither_reaction_rate",
)


def add_overlap_scopes(pair_scores: DataFrame) -> DataFrame:
    individual = pair_scores.copy()
    individual["market_scope"] = "asset:" + individual["pair"].astype(str)
    groups = {
        "established_alts_equal_weight": {
            "ETH/USDT:USDT",
            "BNB/USDT:USDT",
            "ADA/USDT:USDT",
            "TRX/USDT:USDT",
        },
        "all_five_equal_weight": set(breadth_freeze.NORMAL_PAIRS),
    }
    count_columns = (
        "eligible_events",
        "route_a_calls",
        "route_b_calls",
        "both_calls",
        "union_calls",
        "only_a_calls",
        "only_b_calls",
    )
    episode_columns = (
        "both_unique_episodes",
        "only_a_unique_episodes",
        "only_b_unique_episodes",
    )
    rows: list[dict[str, Any]] = []
    for scope, members in groups.items():
        selected = pair_scores.loc[pair_scores["pair"].isin(members)]
        for key, cell in selected.groupby(
            ["period", "route_a", "route_b"], observed=True, sort=False
        ):
            if set(cell["pair"]) != members:
                continue
            rows.append(
                {
                    "pair": scope.upper(),
                    "market_scope": scope,
                    "period": key[0],
                    "route_a": key[1],
                    "route_b": key[2],
                    **{column: int(cell[column].sum()) for column in count_columns},
                    **{column: int(cell[column].min()) for column in episode_columns},
                    **{
                        column: float(pd.to_numeric(cell[column], errors="coerce").mean())
                        for column in OVERLAP_RATE_COLUMNS
                    },
                }
            )
    return pd.concat([individual, DataFrame.from_records(rows)], ignore_index=True)


def decide_overlaps(scope_scores: DataFrame) -> DataFrame:
    expected = set(VALIDATION_PERIODS)
    rows: list[dict[str, Any]] = []
    for key, cell in scope_scores.groupby(
        ["market_scope", "route_a", "route_b"], observed=True, sort=False
    ):
        complete = set(cell["period"]) == expected
        near_duplicate = bool(
            complete and cell["jaccard"].ge(NEAR_DUPLICATE_JACCARD).all()
        )
        complementary = bool(
            complete
            and cell["both_unique_episodes"].ge(MIN_INTERSECTION_EPISODES).all()
            and (
                cell["both_reaction_rate"]
                - cell[
                    [
                        "route_a_matched_strength_reaction_rate",
                        "route_b_matched_strength_reaction_rate",
                    ]
                ].max(axis=1)
            )
            .ge(MIN_COMPLEMENTARY_UPLIFT)
            .all()
        )
        distinct = bool(
            complete
            and not near_duplicate
            and cell["only_a_unique_episodes"].ge(MIN_UNIQUE_ONLY_EPISODES).all()
            and cell["only_b_unique_episodes"].ge(MIN_UNIQUE_ONLY_EPISODES).all()
        )
        decision = (
            "complementary_overlap"
            if complementary
            else "near_duplicate"
            if near_duplicate
            else "distinct_coverage"
            if distinct
            else "unresolved_overlap"
        )
        rows.append(
            {
                "market_scope": key[0],
                "route_a": key[1],
                "route_b": key[2],
                "complete_periods": complete,
                "decision": decision,
                "near_duplicate": near_duplicate,
                "complementary_overlap": complementary,
                "distinct_coverage": distinct,
                "minimum_jaccard": float(cell["jaccard"].min()),
                "maximum_jaccard": float(cell["jaccard"].max()),
                "minimum_both_unique_episodes": int(
                    cell["both_unique_episodes"].min()
                ),
                "minimum_only_a_unique_episodes": int(
                    cell["only_a_unique_episodes"].min()
                ),
                "minimum_only_b_unique_episodes": int(
                    cell["only_b_unique_episodes"].min()
                ),
                "minimum_overlap_uplift": float(
                    (
                        cell["both_reaction_rate"]
                        - cell[
                            [
                                "route_a_matched_strength_reaction_rate",
                                "route_b_matched_strength_reaction_rate",
                            ]
                        ].max(axis=1)
                    ).min()
                ),
            }
        )
    return DataFrame.from_records(rows)


def score() -> dict[str, Any]:
    frozen = verify_freeze()
    source_manifest = _load_json(SOURCE_BREADTH_MANIFEST)
    calls = load_activity_membership()
    universe = build_universe(source_manifest)
    membership = build_membership(universe, calls)
    route_periods = score_routes(membership)
    pair_overlaps = score_overlaps(membership)
    overlap_periods = add_overlap_scopes(pair_overlaps)
    decisions = decide_overlaps(overlap_periods)
    g0.atomic_write_parquet(membership, MEMBERSHIP_PATH)
    g0.atomic_write_csv(route_periods, ROUTE_PERIOD_PATH)
    g0.atomic_write_csv(overlap_periods, OVERLAP_PERIOD_PATH)
    g0.atomic_write_csv(decisions, OVERLAP_DECISIONS_PATH)
    result = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "supersedes_run_id": SUPERSEDES_RUN_ID,
        "created_at_utc": g0.utc_now(),
        "status": "completed_five_family_overlap_review",
        "prototype_only_not_fresh_confirmation": True,
        "profit_used": False,
        "families": len(REPRESENTATIVES),
        "representative_routes": len(REPRESENTATIVES),
        "route_period_cells": len(route_periods),
        "overlap_decision_cells": len(decisions),
        "all_five_decisions": decisions.loc[
            decisions["market_scope"].eq("all_five_equal_weight")
        ]["decision"].value_counts().to_dict(),
        "artifacts": {
            "freeze": artifact(FREEZE_PATH),
            "detailed_membership_on_d": artifact(MEMBERSHIP_PATH),
            "route_periods": artifact(ROUTE_PERIOD_PATH),
            "overlap_periods": artifact(OVERLAP_PERIOD_PATH),
            "overlap_decisions": artifact(OVERLAP_DECISIONS_PATH),
        },
        "interpretation": (
            "This batch determines whether five already-retained activity representations "
            "are operationally separate or overlapping. It does not retest direction, "
            "profit, entries, or exits and is not fresh confirmation."
        ),
        "freeze_status": frozen["status"],
    }
    g0.atomic_write_json(result, RESULT_PATH)
    return {**result, "result_path": str(RESULT_PATH.resolve())}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--overwrite-freeze", action="store_true")
    args = parser.parse_args(argv)
    result = (
        freeze(overwrite=args.overwrite_freeze)
        if args.freeze_only
        else score()
    )
    print(json.dumps(result, indent=2, default=g0.json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
