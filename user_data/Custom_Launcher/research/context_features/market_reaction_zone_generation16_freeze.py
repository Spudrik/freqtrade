"""Freeze the post-Generation-15 attribution batch and bounded one-minute replay."""

from __future__ import annotations

# Bind native pools before pandas/numpy imports.
# ruff: noqa: E402
import os


for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYARROW_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
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
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation3_one_minute_replay as g3m,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation13_direct_controls as g13d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation15_freeze as g15z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation15_joint_review as g15j,
)


DEFAULT_RUN_ID = "g16_broad_attribution_20260822a"
FREEZE_ROOT = g15j.REVIEW_ROOT.parent / "g16_freeze"
FREEZE_PATH = g15j.REVIEW_ROOT.parent / "g16_broad_attribution_freeze_20260822a.json"
ONE_MINUTE_FREEZE_RECORD_NAME = "g16_one_minute_freeze_record.json"
G15_REVIEW = g15j.REVIEW_ROOT / g15j.DEFAULT_RUN_ID / "g15_joint_review.json"
G15_LEADS = g15j.REVIEW_ROOT / g15j.DEFAULT_RUN_ID / "matched_level_leads.csv"
G9_BATCH_FREEZE = (
    REPO_ROOT
    / "user_data/research_news_data/context_features/market_reaction_zones"
    / "generation9_review"
    / "g9_frozen_limited_multisource_and_one_minute_batch_20260821a.json"
)
SELECTION_SEED = "g16-activity-location-representative-1m-v1"
ONE_MINUTE_TIMEFRAMES = ("1h", "4h")
HORIZONS = (1, 2, 4)
PERIODS = {
    "normal": (
        "validation_early",
        "validation_late",
        "recent_confirmation_early",
        "recent_confirmation_late",
    ),
    "meme": ("meme_validation_early", "meme_validation_late"),
}
EPISODE_COLUMNS = (
    "cohort",
    "pair",
    "source_timeframe",
    "level_family",
    "level_name",
    "level_column",
    "representation",
    "control",
    "base_index",
    "event_time",
    "period",
    "source_available_at",
    "source_open",
    "level_price",
    "zone_half_width",
    "zone_half_width_atr",
    "base_atr",
    "approach_state",
    "pre_distance_atr",
    "contact_close_distance_atr",
    "level_score",
    *g13d.MATCH_FEATURES,
)


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


def stable_hash(*values: Any) -> str:
    payload = "|".join(str(value) for value in values).encode()
    return hashlib.sha256(payload).hexdigest()


def validate_parent() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not G15_REVIEW.is_file():
        raise FileNotFoundError(G15_REVIEW)
    review = json.loads(G15_REVIEW.read_text(encoding="utf-8"))
    if review.get("status") != "completed_generation15_joint_review":
        raise ValueError("Generation 15 joint review is not terminal.")
    if not review.get("all_frozen_siblings_completed_or_parked_before_review"):
        raise ValueError("Generation 15 was reviewed before its sibling gate completed.")
    source = json.loads(g15z.FREEZE_PATH.read_text(encoding="utf-8"))
    if source.get("status") != "frozen_before_generation15_outcomes":
        raise ValueError("Generation 15 source freeze changed.")
    manifest_record = source["source_contracts"]["generation6_event_manifest"]
    manifest_path = Path(manifest_record["path"])
    if (
        not manifest_path.is_file()
        or g0.sha256_file(manifest_path) != manifest_record["sha256"]
    ):
        raise ValueError("Frozen Generation 6 event manifest changed.")
    return (
        review,
        json.loads(manifest_path.read_text(encoding="utf-8")),
        manifest_record,
    )


def retained_parent_surface() -> tuple[tuple[str, ...], tuple[str, ...]]:
    leads = pd.read_csv(G15_LEADS)
    strict = leads.loc[leads["strict_with_dominance"]]
    families = tuple(
        sorted(
            strict.loc[
                strict["route_id"].eq("level_family_attribution"), "scope_value"
            ].unique()
        )
    )
    timeframes = tuple(
        sorted(
            strict.loc[
                strict["route_id"].eq("source_timeframe_attribution"), "scope_value"
            ].unique(),
            key=g0.timeframe_hours,
        )
    )
    if not families or not set(ONE_MINUTE_TIMEFRAMES).issubset(timeframes):
        raise ValueError("Generation 15 did not retain the required one-minute parent surface.")
    return families, timeframes


def previous_episode_keys() -> tuple[set[tuple[str, int]], dict[str, Any]]:
    if not G9_BATCH_FREEZE.is_file():
        raise FileNotFoundError(G9_BATCH_FREEZE)
    batch = json.loads(G9_BATCH_FREEZE.read_text(encoding="utf-8"))
    if batch.get("status") != "frozen_before_generation9_sibling_outcomes":
        raise ValueError("Generation 9 batch freeze changed.")
    record_artifact = batch["one_minute_family"]["freeze_record"]
    record_path = Path(record_artifact["path"])
    if (
        not record_path.is_file()
        or g0.sha256_file(record_path) != record_artifact["sha256"]
    ):
        raise ValueError("Generation 9 one-minute freeze record changed.")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    sample_artifact = record["artifacts"]["diagnostic_sample_csv"]
    sample_path = Path(sample_artifact["path"])
    if (
        not sample_path.is_file()
        or g0.sha256_file(sample_path) != sample_artifact["sha256"]
    ):
        raise ValueError("Generation 9 one-minute sample changed.")
    frame = pd.read_csv(sample_path, usecols=["pair", "event_time"])
    times = pd.to_datetime(frame["event_time"], utc=True, errors="raise").astype("int64")
    return set(zip(frame["pair"].astype(str), times, strict=True)), sample_artifact


def build_episode_pool(
    manifest: dict[str, Any], families: Sequence[str]
) -> tuple[DataFrame, dict[str, Any], dict[str, Any]]:
    prior, prior_artifact = previous_episode_keys()
    rows: list[DataFrame] = []
    input_rows = 0
    excluded_prior = 0
    availability_violations = 0
    for task in manifest["tasks"]:
        path = Path(task["event_path"])
        if not path.is_file() or g0.sha256_file(path) != task["event_sha256"]:
            raise ValueError(f"Frozen event source changed: {path}")
        frame = pd.read_parquet(
            path,
            columns=list(EPISODE_COLUMNS),
            filters=[("control", "=", "actual")],
        )
        input_rows += len(frame)
        frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True, errors="raise")
        frame["source_available_at"] = pd.to_datetime(
            frame["source_available_at"], utc=True, errors="raise"
        )
        frame["source_open"] = pd.to_datetime(
            frame["source_open"], utc=True, errors="raise"
        )
        availability_violations += int(
            (frame["source_available_at"] > frame["event_time"]).sum()
        )
        frame = g13d.add_analysis_windows(frame, str(task["cohort"]))
        frame = frame.loc[
            frame["source_timeframe"].isin(ONE_MINUTE_TIMEFRAMES)
            & frame["level_family"].isin(families)
            & frame["analysis_period"].isin(PERIODS[str(task["cohort"])])
        ].copy()
        if frame.empty:
            continue
        old = [
            (pair, int(timestamp)) in prior
            for pair, timestamp in zip(
                frame["pair"].astype(str), frame["event_time"].astype("int64"), strict=True
            )
        ]
        excluded_prior += int(sum(old))
        frame = frame.loc[[not value for value in old]].copy()
        rows.append(frame)
    pool = pd.concat(rows, ignore_index=True, sort=False)
    pool["absolute_contact_distance_atr"] = pd.to_numeric(
        pool["contact_close_distance_atr"], errors="coerce"
    ).abs()
    pool["selection_hash"] = [
        stable_hash(
            SELECTION_SEED,
            cohort,
            pair,
            timestamp,
            timeframe,
            family,
            name,
            level,
        )
        for cohort, pair, timestamp, timeframe, family, name, level in zip(
            pool["cohort"],
            pool["pair"],
            pool["event_time"],
            pool["source_timeframe"],
            pool["level_family"],
            pool["level_name"],
            pool["level_price"],
            strict=True,
        )
    ]
    pool.sort_values(
        [
            "cohort",
            "pair",
            "analysis_period",
            "event_time",
            "source_timeframe",
            "absolute_contact_distance_atr",
            "selection_hash",
        ],
        inplace=True,
    )
    pool = pool.drop_duplicates(
        ["cohort", "pair", "analysis_period", "event_time", "source_timeframe"],
        keep="first",
    ).reset_index(drop=True)
    pool["episode_id"] = [
        "g16m_" + stable_hash(cohort, pair, timestamp, timeframe, selection)[:20]
        for cohort, pair, timestamp, timeframe, selection in zip(
            pool["cohort"],
            pool["pair"],
            pool["event_time"],
            pool["source_timeframe"],
            pool["selection_hash"],
            strict=True,
        )
    ]
    pool["cluster_causal_anchor_timeframe"] = pool["source_timeframe"]
    pool["direction_used_for_selection"] = False
    pool["future_reaction_used_for_selection"] = False
    pool["price_reference_known_at_event"] = True
    return (
        pool,
        {
            "raw_actual_rows": input_rows,
            "prior_generation_episode_rows_excluded": excluded_prior,
            "causal_availability_violations": availability_violations,
            "qualifying_rows": len(pool),
        },
        prior_artifact,
    )


def select_one_minute_sample(pool: DataFrame) -> DataFrame:
    chosen: list[str] = []
    ranks: dict[str, int] = {}
    used_pairs: set[str] = set()
    rank = 0
    for cohort in ("normal", "meme"):
        for period in PERIODS[cohort]:
            for timeframe in ONE_MINUTE_TIMEFRAMES:
                cell = pool.loc[
                    pool["cohort"].eq(cohort)
                    & pool["analysis_period"].eq(period)
                    & pool["source_timeframe"].eq(timeframe)
                    & ~pool["pair"].isin(used_pairs)
                ].sort_values("selection_hash")
                if cell.empty:
                    raise ValueError(
                        "No distinct-pair representative episode for "
                        f"{cohort}|{period}|{timeframe}."
                    )
                selected = cell.iloc[0]
                episode_id = str(selected["episode_id"])
                chosen.append(episode_id)
                ranks[episode_id] = rank
                used_pairs.add(str(selected["pair"]))
                rank += 1
    sample = pool.loc[pool["episode_id"].isin(chosen)].copy()
    sample["sample_selection_order"] = sample["episode_id"].map(ranks).astype(int)
    sample["selected_for_one_minute_replay"] = True
    sample["sample_stratum"] = [
        f"{cohort}|{period}|{timeframe}"
        for cohort, period, timeframe in zip(
            sample["cohort"],
            sample["analysis_period"],
            sample["source_timeframe"],
            strict=True,
        )
    ]
    if len(sample) != 12 or sample["pair"].nunique() != 12:
        raise ValueError("The bounded replay must retain twelve distinct-pair episodes.")
    return sample.sort_values("sample_selection_order").reset_index(drop=True)


def main_routes() -> list[dict[str, Any]]:
    return [
        {
            "route_id": "completed_contact_activity_control",
            "plain_question": (
                "Does the calculated area still add future activity information after "
                "the completed contact candle's volume, range, and pressure are matched?"
            ),
            "controls": ["matched ordinary time", "genuine near miss"],
        },
        {
            "route_id": "stale_shift_and_density_controls",
            "plain_question": (
                "Can a causal 72h-old or deterministically shifted coordinate explain the "
                "same apparent activity, once contact opportunity and candle state match?"
            ),
            "controls": ["causal 72h-old level", "price-shifted level", "ordinary time"],
        },
        {
            "route_id": "pre_contact_to_post_contact_change",
            "plain_question": (
                "Does activity begin or accelerate after contact, or is the result merely "
                "continuation of activity already visible before and during contact?"
            ),
            "controls": ["pre-contact path", "completed contact candle", "matched controls"],
        },
        {
            "route_id": "head_to_head_family_and_timeframe",
            "plain_question": (
                "Does any family or source timeframe add activity information relative to "
                "other genuine calculated-level contacts under the same market state?"
            ),
            "head_to_head_not_pass_count_ranking": True,
        },
        {
            "route_id": "level_density_and_overlap",
            "plain_question": (
                "Is the activity effect explained by how much of the chart is covered or "
                "how many related levels occupy the same area?"
            ),
            "controls": ["same-density contacts", "72h-old density", "shuffled density"],
        },
        {
            "route_id": "local_wider_and_completed_8h_activity_interaction",
            "plain_question": (
                "Does local, wider-crypto, or completed 8h activity supply the market-wide "
                "activity while a calculated area supplies only location or timing?"
            ),
            "component_ablations_required": True,
        },
        {
            "route_id": "freqai_incremental_activity_attribution",
            "plain_question": (
                "Can FreqAI predict volume, range, crossings, or unsigned reaction better "
                "when current level/density/context blocks are added to contact activity?"
            ),
            "profiles": [
                "contact_activity_only",
                "contact_plus_level_identity",
                "contact_plus_level_density",
                "contact_plus_wider_crypto_activity",
                "contact_plus_completed_8h_activity",
                "contact_plus_level_plus_wider_activity",
                "causal_72h_old_added_block",
                "within_period_shuffled_added_block",
            ],
            "complete_models_must_beat_every_immediately_simpler_component": True,
        },
    ]


def build_freeze(run_id: str) -> dict[str, Any]:
    _review, manifest, manifest_record = validate_parent()
    families, timeframes = retained_parent_surface()
    pool, pool_audit, prior_sample_record = build_episode_pool(manifest, families)
    sample = select_one_minute_sample(pool)
    intervals = g3m.acquisition_intervals(sample)
    run_dir = FREEZE_ROOT / run_id
    pool_path = run_dir / "g16_one_minute_qualifying_episode_pool.csv"
    sample_path = run_dir / "g16_one_minute_frozen_sample.csv"
    interval_path = run_dir / "g16_one_minute_acquisition_intervals.csv"
    one_minute_record_path = run_dir / ONE_MINUTE_FREEZE_RECORD_NAME
    g0.atomic_write_csv(pool, pool_path)
    g0.atomic_write_csv(sample, sample_path)
    g0.atomic_write_csv(intervals, interval_path)
    one_minute_record = {
        "schema_version": 1,
        "generation": 16,
        "run_id": run_id,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_one_minute_paths",
        "request_sha256": stable_hash(
            SELECTION_SEED,
            *sample["episode_id"].astype(str).tolist(),
            *intervals["timerange"].astype(str).tolist(),
        ),
        "directional_path_read": False,
        "episode_pool_rows": len(pool),
        "sample_rows": len(sample),
        "selection_seed": SELECTION_SEED,
        "artifacts": {
            "episode_pool_csv": artifact(pool_path),
            "frozen_sample_csv": artifact(sample_path),
            "acquisition_intervals_csv": artifact(interval_path),
        },
    }
    g0.atomic_write_json(one_minute_record, one_minute_record_path)
    routes = main_routes()
    return {
        "schema_version": 1,
        "objective": "objective_02b_market_reaction_zone_discovery",
        "generation": 16,
        "run_id": run_id,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation16_outcomes",
        "plain_objective": (
            "Challenge the retained activity-location result against contact persistence, "
            "density, genuine alternative levels, family/timeframe, and market-regime "
            "explanations while running one subordinate representative 1m replay."
        ),
        "source_contracts": {
            "generation15_joint_review": artifact(G15_REVIEW),
            "generation15_level_leads": artifact(G15_LEADS),
            "generation6_event_manifest": manifest_record,
            "generation9_prior_episode_sample": prior_sample_record,
        },
        "parent_surface": {
            "strict_level_families": list(families),
            "strict_source_timeframes": list(timeframes),
            "main_horizons_hours": list(HORIZONS),
            "retained_behaviours": ["future volume", "future range", "crossings"],
            "unretained_behaviours": [
                "broad absolute excursion",
                "hit timing",
                "dwell",
                "absolute pressure change",
            ],
        },
        "main_direction_neutral_family": {
            "routes": routes,
            "route_count": len(routes),
            "cohorts": ["normal", "meme"],
            "normal_groups": {
                key: list(value) for key, value in g15z.NORMAL_GROUPS.items()
            },
            "matching": {
                "outcome_blind": True,
                "pre_contact_state": list(g13d.MATCH_FEATURES),
                "completed_contact_state": [
                    "contact_volume_ratio",
                    "contact_range_ratio",
                    "absolute_contact_pressure_change",
                ],
                "starting_distance_caliper_atr": 0.10,
                "maximum_control_reuse": 3,
                "overlap_purge_equals_each_horizon": True,
            },
            "targets": {
                "horizons_hours": list(HORIZONS),
                "behaviours": [
                    "future_volume_ratio",
                    "future_range_ratio",
                    "future_crossings",
                    "unsigned_reaction",
                    "pre_to_post_activity_change",
                ],
            },
            "point_rule": (
                "One sign in both periods and all declared controls with adequate coin "
                "support; combinations beat every immediately simpler component."
            ),
            "strict_rule": (
                "Point rule plus weekly-block uncertainty on one side of zero and no one "
                "coin above 50% of the absolute equal-coin effect."
            ),
            "future_signed_direction_used": False,
            "profit_used": False,
        },
        "bounded_one_minute_family": {
            "status": "frozen_before_one_minute_paths",
            "parent_pattern": (
                "A causally calculated 1h or 4h area from a Generation 15 strict family "
                "is a repeated short-window activity location."
            ),
            "selection": {
                "seed": SELECTION_SEED,
                "one_episode_per_period_and_source_timeframe": True,
                "periods": {key: list(value) for key, value in PERIODS.items()},
                "source_timeframes": list(ONE_MINUTE_TIMEFRAMES),
                "distinct_pairs": True,
                "prior_generation_episode_times_excluded": True,
                "future_reaction_used": False,
                "future_signed_direction_used": False,
            },
            "pool_audit": pool_audit,
            "pool_rows": len(pool),
            "sample_rows": len(sample),
            "sample_pairs": sorted(sample["pair"].astype(str).unique()),
            "sample_family_counts": sample["level_family"].value_counts().to_dict(),
            "acquisition_intervals": len(intervals),
            "windows": {
                "1h": {"pre_hours": 24, "post_hours": 12},
                "4h": {"pre_hours": 72, "post_hours": 48},
            },
            "direct_first": True,
            "freqai_on_twelve_episode_sample": False,
            "direction_methods": [
                "pre-contact 15m/60m/240m path",
                "completed contact-candle pressure and close position",
                "rolling minute volume/pressure acceleration",
                "simple local trend comparator",
                "majority direction comparator",
                "timestamp-safe order book only where genuinely covered",
            ],
            "required_reporting": [
                "reaction success",
                "conditional direction success",
                "joint success",
                "issued-call coverage and abstentions",
                "independent episodes",
                "per-coin and per-period outcomes",
                "55% floor and 65% target comparison",
            ],
            "diagnostic_not_promotion": True,
        },
        "sequencing": {
            "all_main_routes_and_one_minute_lane_frozen_together": True,
            "all_terminal_or_honestly_parked_before_joint_review": True,
            "no_generation17_descendant_before_joint_review": True,
            "automatic_descendant_launch": False,
        },
        "coverage_parks": {
            "historical_news": "zero eligible historical rows in Generation 14",
            "historical_orderbook_newest_later_period": "not covered",
        },
        "research_boundary": {
            "broad_hourly_direction_model": False,
            "ordinary_timestamp_direction_search": False,
            "direction_only_inside_frozen_one_minute_sample": True,
            "profit_used": False,
            "entry_exit_or_trading_action": False,
        },
        "runtime": {
            "maximum_main_workers": 4,
            "maximum_one_minute_workers_while_main_active": 1,
        },
        "artifacts": {
            "one_minute_episode_pool": artifact(pool_path),
            "one_minute_frozen_sample": artifact(sample_path),
            "one_minute_acquisition_intervals": artifact(interval_path),
            "one_minute_freeze_record": artifact(one_minute_record_path),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    freeze = build_freeze(args.run_id)
    g0.atomic_write_json(freeze, FREEZE_PATH)
    print(
        json.dumps(
            {
                "status": freeze["status"],
                "main_routes": freeze["main_direction_neutral_family"]["route_count"],
                "one_minute_pool": freeze["bounded_one_minute_family"]["pool_rows"],
                "one_minute_sample": freeze["bounded_one_minute_family"]["sample_rows"],
                "one_minute_pairs": freeze["bounded_one_minute_family"]["sample_pairs"],
                "output": str(FREEZE_PATH.resolve()),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
