"""Freeze the complete Generation 17 branch layer before opening its outcomes."""

from __future__ import annotations

# Keep selection deterministic and bounded.
# ruff: noqa: E402
import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation13_cache as g13c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation16_freqai_cache as g16c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation16_joint_review as g16j,
)


DEFAULT_BATCH_ID = "g17_broad_branch_layer_20260822a"
OUTPUT_ROOT = g16j.REVIEW_ROOT.parent / "g17_broad_branch_layer"
FREEZE_PATH = OUTPUT_ROOT.parent / "g17_broad_branch_layer_freeze_20260822a.json"
SELECTION_SEED = "g17-density-balanced-one-minute-v1"
ONE_MINUTE_SAMPLE_SIZE = 40
HORIZONS_HOURS = (1, 2, 4)
ONE_MINUTE_HORIZONS = (15, 60, 240, 720)

DENSITY_PROFILES = (
    "contact_only",
    "contact_plus_density_counts",
    "contact_plus_proximity_width",
    "contact_plus_cluster_geometry",
    "contact_plus_counts_and_geometry",
    "contact_plus_full_density_geometry",
    "contact_plus_stale_full_density_geometry",
    "contact_plus_shuffled_full_density_geometry",
)
REACTION_PROFILES = (
    "contact_only",
    "contact_plus_full_density_geometry",
    "contact_plus_completed_8h",
    "contact_plus_local_activity",
    "contact_plus_wider_crypto",
    "contact_plus_density_and_completed_8h",
    "contact_plus_density_and_local",
    "contact_plus_density_and_wider",
    "contact_plus_density_local_wider_8h",
)
EXTERNAL_PROFILES = (
    "contact_plus_full_density_geometry",
    "contact_plus_density_and_orderbook",
    "contact_plus_density_and_stale_orderbook",
    "contact_plus_density_and_shuffled_orderbook",
)


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


def stable_key(*values: object) -> str:
    text = "|".join(str(value) for value in values)
    return hashlib.sha256(f"{SELECTION_SEED}|{text}".encode()).hexdigest()


def load_parent() -> dict[str, Any]:
    path = g16j.REVIEW_ROOT / g16j.DEFAULT_REVIEW_ID / "g16_joint_review.json"
    parent = json.loads(path.read_text(encoding="utf-8"))
    if parent.get("status") != "completed_generation16_joint_review":
        raise ValueError("Generation 16 joint review is not terminal.")
    if parent.get("branch_batches_queued") != 6:
        raise ValueError("Generation 16 did not queue the complete six-batch layer.")
    return parent


def branch_definitions() -> list[dict[str, Any]]:
    return [
        {
            "branch_id": "g17a_density_geometry_decomposition",
            "plain_question": (
                "Which gradual density, distance/width, or cluster-geometry components "
                "explain the robust crossing-count gain?"
            ),
            "method": "FreqAI component ladder with current, causal-72h, and shuffled controls",
            "profiles": list(DENSITY_PROFILES),
            "targets": [
                f"crossing_count_h{horizon}" for horizon in HORIZONS_HOURS
            ]
            + [f"repeated_recross_h{horizon}" for horizon in HORIZONS_HOURS]
            + [f"two_sided_traversal_h{horizon}" for horizon in HORIZONS_HOURS],
            "pass_rule": (
                "A complete component must beat contact-only and every immediately "
                "simpler component in both normal chronologies and the meme cell. The "
                "current block must also beat stale and shuffled equivalents."
            ),
        },
        {
            "branch_id": "g17b_crossing_semantics",
            "plain_question": (
                "Are calculated areas associated with useful rejection, traversal, "
                "two-sided traffic, repeated recrossing, or dwell rather than merely "
                "one mechanically available crossing?"
            ),
            "method": (
                "Direct completed-contact matching against ordinary time and "
                "genuine near miss"
            ),
            "targets": [
                "any_recross",
                "repeated_recross",
                "two_sided_traversal",
                "one_sided_rejection",
                "one_sided_breakthrough",
                "dwell_fraction",
            ],
            "horizons_hours": list(HORIZONS_HOURS),
            "pass_rule": (
                "The same behaviour sign must beat both controls in both normal "
                "chronologies or form a separate coherent meme/subgroup result."
            ),
        },
        {
            "branch_id": "g17c_reaction_probability_calibration",
            "plain_question": (
                "Can completed 8h, local, wider-crypto, and density state estimate the "
                "probability of a direction-neutral reaction better than simpler inputs?"
            ),
            "method": "FreqAI reaction-only component and combination ladder",
            "profiles": list(REACTION_PROFILES),
            "targets": [f"reaction_h{horizon}" for horizon in HORIZONS_HOURS],
            "pass_rule": (
                "Retain only if unseen balanced accuracy, rank AUC, calibration error, "
                "and weekly uncertainty beat contact-only; every combination must beat "
                "all immediately simpler components."
            ),
        },
        {
            "branch_id": "g17d_broad_level_sources",
            "plain_question": (
                "Do rational causal active-area calculations beyond the retained fixed "
                "surface localize reaction without event-specific tuning?"
            ),
            "method": "Direct source/control atlas, singles and clusters reported equally",
            "fixed_source_families": [
                "adaptive_volume_profile_nodes",
                "rolling_vwap_deviation_bands",
                "donchian_boundaries",
                "weekly_pivot_grid",
                "generic_ma_bollinger_negative_control",
            ],
            "volume_profile_variants": {
                "lookback_hours": [72, 168, 720],
                "fixed_bins": [24, 48, 96],
                "adaptive_bins": "Freedman-Diaconis clipped to 24-96 bins",
                "value_area_fraction": 0.70,
                "hvn_quantiles": [0.80, 0.90],
                "lvn_quantiles": [0.10, 0.20],
            },
            "pass_rule": (
                "No parameter may be selected from an exposed event. A family must beat "
                "ordinary, near-miss, shifted, stale, and same-coverage controls on later "
                "normal chronology or a coherent predeclared subgroup."
            ),
        },
        {
            "branch_id": "g17e_external_context_regimes",
            "plain_question": (
                "Do source-ready order book, news, or global-market regimes add reaction "
                "information beyond contact plus density?"
            ),
            "method": "Coverage gate first; FreqAI only on identical source-ready rows",
            "profiles": list(EXTERNAL_PROFILES),
            "source_routes": [
                "timestamp_safe_btc_orderbook",
                "historical_news_context_if_ready",
                "non_crypto_global_market_context_if_ready",
            ],
            "pass_rule": (
                "Current context must beat density-only, causal stale, shuffled, and "
                "source-missing controls in repeated windows. A route with inadequate "
                "coverage is parked without model training."
            ),
        },
        {
            "branch_id": "g17f_bounded_one_minute_expansion",
            "plain_question": (
                "On a larger density-balanced causal sample, can one-minute pressure, "
                "trend, momentum, order book, or wider state predict both reaction and "
                "direction?"
            ),
            "method": "Frozen 40-episode replay with source-timeframe-scaled context windows",
            "sample_size": ONE_MINUTE_SAMPLE_SIZE,
            "horizons_minutes": list(ONE_MINUTE_HORIZONS),
            "pass_rule": (
                "Abstentions count as failures. At least 55% joint reaction-and-direction "
                "success is the minimum lead and 65% is the main target on independent "
                "episodes, with reaction-only and conditional-direction rates separate."
            ),
        },
    ]


def source_items(cohort: str) -> tuple[dict[str, Any], dict[str, Any]]:
    g13_path = g13c.RECORD_ROOT / f"{cohort}_manifest.json"
    g16_path = g16c.RECORD_ROOT / f"{cohort}_manifest.json"
    g13 = json.loads(g13_path.read_text(encoding="utf-8"))
    g16 = json.loads(g16_path.read_text(encoding="utf-8"))
    if g13.get("status") != "completed_generation13_cache":
        raise ValueError(f"Generation 13 {cohort} cache is not terminal.")
    if g16.get("status") != "completed_generation16_freqai_cache":
        raise ValueError(f"Generation 16 {cohort} cache is not terminal.")
    return g13, g16


def episode_pool() -> DataFrame:
    frames: list[DataFrame] = []
    for cohort in ("normal", "meme"):
        g13, g16 = source_items(cohort)
        g13_by_pair = {item["pair"]: item for item in g13["inventory"]}
        for item in g16["inventory"]:
            pair = item["pair"]
            original = g13_by_pair[pair]
            features = pd.read_parquet(
                item["feature_path"],
                columns=[
                    "date",
                    "g11_minimal_contact__contacted_level_count",
                    "g11_minimal_contact__distinct_family_count",
                    "g11_cluster_geometry__independent_family_count",
                    "g11_cluster_geometry__any_cross_timeframe_cluster",
                ],
            )
            metadata = pd.read_parquet(
                item["evaluation_path"],
                columns=[
                    "date",
                    "period",
                    "pair",
                    "cohort",
                    "anchor_level_family",
                    "anchor_level_name",
                    "anchor_source_timeframe",
                    "anchor_representation",
                ],
            )
            for frame in (features, metadata):
                frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
            candidates = metadata.merge(features, on="date", how="inner", validate="one_to_one")
            source = pd.read_parquet(
                original["source_path"],
                columns=[
                    "event_time",
                    "level_family",
                    "level_name",
                    "source_timeframe",
                    "representation",
                    "control",
                    "level_price",
                    "zone_half_width",
                    "zone_half_width_atr",
                    "base_atr",
                    "pre_distance_atr",
                ],
                filters=[("control", "==", "actual")],
            )
            source["event_time"] = pd.to_datetime(
                source["event_time"], utc=True, errors="raise"
            )
            source.sort_values(
                [
                    "event_time",
                    "level_family",
                    "level_name",
                    "source_timeframe",
                    "representation",
                ],
                inplace=True,
            )
            source.drop_duplicates(
                [
                    "event_time",
                    "level_family",
                    "level_name",
                    "source_timeframe",
                    "representation",
                ],
                inplace=True,
            )
            candidates = candidates.merge(
                source.drop(columns="control"),
                left_on=[
                    "date",
                    "anchor_level_family",
                    "anchor_level_name",
                    "anchor_source_timeframe",
                    "anchor_representation",
                ],
                right_on=[
                    "event_time",
                    "level_family",
                    "level_name",
                    "source_timeframe",
                    "representation",
                ],
                how="inner",
                validate="one_to_one",
            )
            candidates["density_score"] = (
                pd.to_numeric(
                    candidates["g11_minimal_contact__contacted_level_count"],
                    errors="coerce",
                ).rank(pct=True)
                + pd.to_numeric(
                    candidates["g11_minimal_contact__distinct_family_count"],
                    errors="coerce",
                ).rank(pct=True)
                + pd.to_numeric(
                    candidates["g11_cluster_geometry__independent_family_count"],
                    errors="coerce",
                ).rank(pct=True)
            ) / 3.0
            candidates["density_regime"] = np.where(
                candidates["density_score"].ge(0.70),
                "high",
                np.where(candidates["density_score"].le(0.30), "low", "middle"),
            )
            candidates = candidates.loc[
                candidates["density_regime"].isin(["high", "low"])
                & pd.to_numeric(candidates["level_price"], errors="coerce").gt(0)
                & pd.to_numeric(candidates["base_atr"], errors="coerce").gt(0)
            ].copy()
            candidates["selection_key"] = [
                stable_key(
                    cohort,
                    pair,
                    row.date,
                    row.density_regime,
                    row.anchor_source_timeframe,
                )
                for row in candidates.itertuples()
            ]
            frames.append(candidates)
    return pd.concat(frames, ignore_index=True, sort=False)


def select_episodes(pool: DataFrame) -> DataFrame:
    rows: list[pd.Series] = []
    for (cohort, pair, regime), frame in pool.groupby(
        ["cohort", "pair", "density_regime"], observed=True, sort=True
    ):
        preferred = frame.copy()
        if cohort == "normal":
            pair_number = int(stable_key(pair)[:8], 16)
            desired_window = "standard" if (pair_number + (regime == "high")) % 2 else "recent"
            if desired_window == "recent":
                recent = preferred.loc[
                    preferred["date"].ge(pd.Timestamp("2026-04-01", tz="UTC"))
                ]
                if not recent.empty:
                    preferred = recent
            else:
                standard = preferred.loc[
                    preferred["period"].isin(["validation_early", "validation_late"])
                ]
                if not standard.empty:
                    preferred = standard
        selected = preferred.sort_values("selection_key", kind="stable").iloc[0].copy()
        selected["selection_stratum"] = f"{cohort}|{regime}"
        rows.append(selected)
    sample = DataFrame(rows)
    if len(sample) != ONE_MINUTE_SAMPLE_SIZE:
        raise ValueError(
            f"Expected {ONE_MINUTE_SAMPLE_SIZE} density-balanced episodes, got {len(sample)}."
        )
    cohort_pairs = sample[["cohort", "pair"]].drop_duplicates()
    if len(cohort_pairs) != 20:
        raise ValueError(
            "Generation 17 replay must retain all twenty cohort-pair units."
        )
    if not (
        sample.groupby(["cohort", "pair"], observed=True)["density_regime"]
        .nunique()
        .eq(2)
        .all()
    ):
        raise ValueError(
            "Each cohort-pair unit must contribute one high- and one low-density episode."
        )
    sample["episode_id"] = [
        "g17m_" + stable_key(row.cohort, row.pair, row.date, row.density_regime)[:20]
        for row in sample.itertuples()
    ]
    sample["pre_context_hours"] = sample["anchor_source_timeframe"].map(
        {"1h": 12, "4h": 48, "8h": 96}
    )
    sample["post_context_hours"] = sample["pre_context_hours"]
    if sample[["pre_context_hours", "post_context_hours"]].isna().any().any():
        raise ValueError("Unsupported source timeframe in Generation 17 replay sample.")
    return sample.sort_values(["cohort", "pair", "density_regime"]).reset_index(drop=True)


def acquisition_intervals(sample: DataFrame) -> DataFrame:
    output = sample[
        [
            "episode_id",
            "cohort",
            "pair",
            "date",
            "anchor_source_timeframe",
            "pre_context_hours",
            "post_context_hours",
        ]
    ].copy()
    output["interval_start"] = output["date"] - pd.to_timedelta(
        output["pre_context_hours"], unit="h"
    )
    output["interval_stop"] = output["date"] + pd.to_timedelta(
        output["post_context_hours"], unit="h"
    )
    return output


def freeze_batch(batch_id: str) -> dict[str, Any]:
    parent = load_parent()
    output_dir = OUTPUT_ROOT / batch_id
    output_dir.mkdir(parents=True, exist_ok=True)
    pool = episode_pool()
    sample = select_episodes(pool)
    intervals = acquisition_intervals(sample)
    pool_path = output_dir / "g17_one_minute_outcome_blind_pool.csv"
    sample_path = output_dir / "g17_one_minute_frozen_sample.csv"
    interval_path = output_dir / "g17_one_minute_acquisition_intervals.csv"
    g0.atomic_write_csv(pool, pool_path)
    g0.atomic_write_csv(sample, sample_path)
    g0.atomic_write_csv(intervals, interval_path)

    g13_sources = [
        artifact(g13c.RECORD_ROOT / f"{cohort}_manifest.json")
        for cohort in ("normal", "meme")
    ]
    g16_sources = [
        artifact(g16c.RECORD_ROOT / f"{cohort}_manifest.json")
        for cohort in ("normal", "meme")
    ]
    freeze = {
        "schema_version": 1,
        "generation": 17,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation17_outcomes",
        "batch_id": batch_id,
        "objective": (
            "Complete six broad sibling batches inspired by Generation 16 without "
            "letting the density lead dominate the whole investigation."
        ),
        "sequencing": {
            "all_six_frozen_together": True,
            "all_terminal_or_honestly_parked_before_joint_review": True,
            "no_generation18_descendant_before_joint_review": True,
            "automatic_descendant_launch": False,
            "authorized_branch_layer": 1,
        },
        "branches": branch_definitions(),
        "cohorts": {
            "normal": 10,
            "top10_memes": 10,
            "similar_coin_groups_retained": True,
            "btc_separate": True,
        },
        "main_horizons_hours": list(HORIZONS_HOURS),
        "research_boundary": {
            "main_profit_used": False,
            "main_future_signed_direction_used": False,
            "direction_only_in_bounded_one_minute_branch": True,
            "reaction_accuracy_not_direction_accuracy": True,
            "trading_promotion": False,
        },
        "one_minute_sample": {
            "selection_used_future_reaction_or_direction": False,
            "episodes": len(sample),
            "cohort_pair_units": len(
                sample[["cohort", "pair"]].drop_duplicates()
            ),
            "distinct_symbols": sample["pair"].nunique(),
            "normal_episodes": int(sample["cohort"].eq("normal").sum()),
            "meme_episodes": int(sample["cohort"].eq("meme").sum()),
            "high_density_episodes": int(sample["density_regime"].eq("high").sum()),
            "low_density_episodes": int(sample["density_regime"].eq("low").sum()),
            "sample": artifact(sample_path),
            "acquisition_intervals": artifact(interval_path),
            "pool": artifact(pool_path),
        },
        "coverage_parks": {
            "historical_news": (
                "Park without training if the active event cache has no source-ready "
                "rows in the required validation cells."
            ),
            "non_crypto_global_markets": (
                "Park without training if no timestamp-safe multi-period source is "
                "available."
            ),
            "new_level_source": (
                "Park an individual family when fair ordinary, near-miss, shifted, "
                "stale, and same-coverage controls cannot be constructed."
            ),
        },
        "source_contracts": {
            "generation16_joint_review": artifact(
                g16j.REVIEW_ROOT / g16j.DEFAULT_REVIEW_ID / "g16_joint_review.json"
            ),
            "generation16_branch_queue": parent["artifacts"]["branch_queue"],
            "generation13_manifests": g13_sources,
            "generation16_freqai_caches": g16_sources,
        },
    }
    g0.atomic_write_json(freeze, FREEZE_PATH)
    return freeze


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
    args = parser.parse_args(argv)
    freeze = freeze_batch(args.batch_id)
    print(
        json.dumps(
            {
                "status": freeze["status"],
                "branches": len(freeze["branches"]),
                "episodes": freeze["one_minute_sample"]["episodes"],
                "output": str(FREEZE_PATH.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
