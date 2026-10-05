"""Freeze the Generation 11 FreqAI attribution registry before model outcomes."""

from __future__ import annotations

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
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation11_freeze as g11z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation11_joint_review as g11r,
)


JOINT_REVIEW = (
    g11r.REVIEW_ROOT / g11r.DEFAULT_REVIEW_ID / "g11_joint_review.json"
)
FREEZE_PATH = (
    g11r.REVIEW_ROOT / "g11_frozen_freqai_initial_attribution_20260822a.json"
)
INITIAL_SEED = 42
TARGETS = tuple(
    [
        *(f"&-g11_reaction_h{horizon}" for horizon in g11z.HORIZONS),
        *(f"&-g11_volume_ratio_h{horizon}" for horizon in g11z.HORIZONS),
    ]
)

MINIMAL = "g11_minimal_contact"
LEVEL = "g11_level_identity_timeframe"
GEOMETRY = "g11_cluster_geometry"
TREND = "g11_local_trend_momentum"
ACTIVITY = "g11_local_activity_volatility"
MARKET = "g11_wider_crypto_market"
NEWS = "g11_news_context"
ORDERBOOK = "g11_orderbook_context"

CURRENT_BLOCKS = (
    MINIMAL,
    LEVEL,
    GEOMETRY,
    TREND,
    ACTIVITY,
    MARKET,
    NEWS,
    ORDERBOOK,
)

LEVEL_FAMILIES = (
    "confirmed_swing",
    "generic_prior_range",
    "generic_round_number",
    "tlv2_forecast_zone",
    "tlv2_ranked",
    "volume_profile_explicit_prior",
    "volume_profile_nodes",
    "volume_profile_settled",
)

FEATURE_SUFFIXES: dict[str, tuple[str, ...]] = {
    MINIMAL: (
        "contacted_level_count",
        "distinct_family_count",
        "distinct_timeframe_count",
        "distinct_mechanism_count",
        "mean_zone_half_width_atr",
        "minimum_contact_distance_atr",
        "anchor_zone_half_width_atr",
        "anchor_pre_distance_atr",
        "anchor_contact_distance_atr",
        "anchor_approach_code",
        "anchor_level_score_available",
        "anchor_level_score",
    ),
    LEVEL: (
        *(f"anchor_source_{timeframe}" for timeframe in g11z.SOURCE_TIMEFRAMES),
        *(f"anchor_family_{family}" for family in LEVEL_FAMILIES),
        *(f"contact_fraction_source_{timeframe}" for timeframe in g11z.SOURCE_TIMEFRAMES),
        *(f"contact_fraction_family_{family}" for family in LEVEL_FAMILIES),
    ),
    GEOMETRY: (
        "same_mechanism_agreement",
        "different_mechanism_agreement",
        "any_cross_timeframe_cluster",
        "opposing_side_overlap",
        "isolated_anchor",
        "independent_family_count",
    ),
    TREND: (
        "ema20_slope",
        "ma_separation",
        "return_slope",
        "return_acceleration",
        "adx14",
        "rsi14_centered",
        "rsi_change",
        "macd_histogram",
        "macd_histogram_change",
    ),
    ACTIVITY: (
        "relative_volume",
        "volume_acceleration",
        "absolute_pressure",
        "pressure_persistence",
        "atr_fraction",
        "prior_range_atr",
        "bollinger_width",
        "range_contraction",
    ),
    MARKET: (
        "btc_return_1h",
        "btc_return_4h",
        "btc_return_24h",
        "btc_relative_volume",
        "eth_return_1h",
        "eth_return_4h",
        "eth_return_24h",
        "cohort_breadth_positive",
        "cohort_dispersion",
        "cohort_absolute_activity",
        "cohort_ready_members",
        "pair_minus_btc_1h",
        "pair_minus_btc_24h",
        "pair_btc_corr_168h",
        "pair_btc_beta_168h",
    ),
    NEWS: (
        "source_ready",
        "topics_ready",
        "activity",
        "log_event_count_1h",
        "log_event_count_24h",
        "log_num_mentions_24h",
        "log_num_sources_24h",
        "average_tone_24h",
        "goldstein_24h",
        "topic_geopolitics_and_energy",
        "topic_macro_policy_and_growth",
        "topic_crypto_policy_and_institutions",
        "topic_crypto_liquidity_and_security",
        "regime_quiet",
        "regime_middle",
        "regime_shock",
    ),
    ORDERBOOK: (
        "source_ready",
        "model_ready",
        "coverage_ratio",
        "pressure_25bps",
        "absolute_pressure",
        "pair_is_btc",
        "regime_quiet",
        "regime_middle",
        "regime_active",
    ),
}

QUESTION_DEFINITIONS = (
    {
        "question_id": "level_identity_and_timeframe",
        "route_id": "level_identity_and_timeframe",
        "plain_question": (
            "Does causal level family and source timeframe improve multi-horizon reaction "
            "and volume forecasts beyond basic contact geometry?"
        ),
        "base_blocks": (MINIMAL,),
        "addition_blocks": (LEVEL,),
    },
    {
        "question_id": "isolated_and_cluster_geometry",
        "route_id": "isolated_and_cluster_geometry",
        "plain_question": (
            "Do isolated, independently clustered, or opposing calculated levels add "
            "nonlinear reaction information after identity and timeframe are known?"
        ),
        "base_blocks": (MINIMAL, LEVEL),
        "addition_blocks": (GEOMETRY,),
    },
    {
        "question_id": "local_trend_and_momentum",
        "route_id": "local_trend_and_momentum",
        "plain_question": (
            "Do causal local trend, momentum, RSI, and MACD state improve unsigned "
            "reaction forecasts at comparable calculated areas?"
        ),
        "base_blocks": (MINIMAL, LEVEL),
        "addition_blocks": (TREND,),
    },
    {
        "question_id": "local_activity_and_volatility",
        "route_id": "local_activity_and_volatility",
        "plain_question": (
            "Can nonlinear volume, pressure, volatility, and compression combinations "
            "add information despite the failed coarse activity-state direct screen?"
        ),
        "base_blocks": (MINIMAL, LEVEL),
        "addition_blocks": (ACTIVITY,),
    },
    {
        "question_id": "wider_crypto_market_alignment",
        "route_id": "wider_crypto_market_alignment",
        "plain_question": (
            "Does signed BTC, ETH, and cohort state improve reaction forecasts when local "
            "and wider-market pressure agree or conflict?"
        ),
        "base_blocks": (MINIMAL, LEVEL),
        "addition_blocks": (MARKET,),
    },
    {
        "question_id": "external_context_guard",
        "route_id": "news_context",
        "plain_question": (
            "Within genuinely covered timestamps, does GDELT activity and topic state add "
            "reaction information beyond level and OHLCV data?"
        ),
        "base_blocks": (MINIMAL, LEVEL),
        "addition_blocks": (NEWS,),
    },
    {
        "question_id": "external_context_guard",
        "route_id": "orderbook_context",
        "plain_question": (
            "Within genuinely covered timestamps, does BTC order-book pressure and regime "
            "add reaction information beyond level and OHLCV data?"
        ),
        "base_blocks": (MINIMAL, LEVEL),
        "addition_blocks": (ORDERBOOK,),
    },
)


def stale(blocks: Sequence[str]) -> tuple[str, ...]:
    return tuple(f"{block}_stale" for block in blocks)


def shuffled(blocks: Sequence[str]) -> tuple[str, ...]:
    return tuple(f"{block}_shuffled" for block in blocks)


def profile_id(cohort: str, role: str, seed: int = INITIAL_SEED) -> str:
    return f"g11__{cohort}__{role}__seed{seed}"


def register_profile(
    profiles: dict[str, dict[str, Any]],
    *,
    cohort: str,
    role: str,
    blocks: Sequence[str],
    seed: int,
) -> str:
    identifier = profile_id(cohort, role, seed)
    definition = {
        "profile_id": identifier,
        "cohort": cohort,
        "role": role,
        "seed": seed,
        "blocks": list(dict.fromkeys(blocks)),
        "required_ready_blocks": list(dict.fromkeys(blocks)),
        "targets": list(TARGETS),
    }
    existing = profiles.get(identifier)
    if existing is not None and existing != definition:
        raise ValueError(f"Profile id collision with different blocks: {identifier}")
    profiles[identifier] = definition
    return identifier


def build_registry(
    seeds: Sequence[int] = (INITIAL_SEED,),
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    profiles: dict[str, dict[str, Any]] = {}
    comparisons: list[dict[str, Any]] = []
    for cohort in g11z.COHORTS:
        for seed in seeds:
            for question in QUESTION_DEFINITIONS:
                base = tuple(question["base_blocks"])
                addition = tuple(question["addition_blocks"])
                stem = str(question["route_id"])
                baseline = register_profile(
                    profiles,
                    cohort=cohort,
                    role=f"{stem}__baseline",
                    blocks=base,
                    seed=seed,
                )
                candidate = register_profile(
                    profiles,
                    cohort=cohort,
                    role=f"{stem}__current",
                    blocks=(*base, *addition),
                    seed=seed,
                )
                stale_control = register_profile(
                    profiles,
                    cohort=cohort,
                    role=f"{stem}__stale",
                    blocks=(*base, *stale(addition)),
                    seed=seed,
                )
                shuffled_control = register_profile(
                    profiles,
                    cohort=cohort,
                    role=f"{stem}__shuffled",
                    blocks=(*base, *shuffled(addition)),
                    seed=seed,
                )
                for control_type, control in (
                    ("immediately_simpler", baseline),
                    ("causal_stale", stale_control),
                    ("within_period_nonself_shuffle", shuffled_control),
                ):
                    comparisons.append(
                        {
                            "comparison_id": (
                                f"{cohort}__{stem}__vs_{control_type}__seed{seed}"
                            ),
                            "cohort": cohort,
                            "question_id": question["question_id"],
                            "route_id": stem,
                            "route_type": "add_one_information_family",
                            "plain_question": question["plain_question"],
                            "candidate": candidate,
                            "baseline": control,
                            "control_type": control_type,
                            "expected_controls_for_route": 3,
                            "seed": seed,
                            "targets": list(TARGETS),
                        }
                    )

            base = (MINIMAL, LEVEL)
            additions = (GEOMETRY, TREND, MARKET)
            stem = "low_dimensional_combination"
            candidate = register_profile(
                profiles,
                cohort=cohort,
                role=f"{stem}__current",
                blocks=(*base, *additions),
                seed=seed,
            )
            controls = [
                (
                    "level_only",
                    register_profile(
                        profiles,
                        cohort=cohort,
                        role=f"{stem}__level_only",
                        blocks=base,
                        seed=seed,
                    ),
                ),
                (
                    "all_additions_causal_stale",
                    register_profile(
                        profiles,
                        cohort=cohort,
                        role=f"{stem}__all_stale",
                        blocks=(*base, *stale(additions)),
                        seed=seed,
                    ),
                ),
                (
                    "all_additions_within_period_nonself_shuffle",
                    register_profile(
                        profiles,
                        cohort=cohort,
                        role=f"{stem}__all_shuffled",
                        blocks=(*base, *shuffled(additions)),
                        seed=seed,
                    ),
                ),
            ]
            for omitted in additions:
                retained = tuple(block for block in additions if block != omitted)
                controls.append(
                    (
                        f"leave_out_{omitted}",
                        register_profile(
                            profiles,
                            cohort=cohort,
                            role=f"{stem}__leave_{omitted}",
                            blocks=(*base, *retained),
                            seed=seed,
                        ),
                    )
                )
            for control_type, control in controls:
                comparisons.append(
                    {
                        "comparison_id": (
                            f"{cohort}__{stem}__vs_{control_type}__seed{seed}"
                        ),
                        "cohort": cohort,
                        "question_id": stem,
                        "route_id": stem,
                        "route_type": "bounded_multi_family_interaction",
                        "plain_question": (
                            "Does level identity plus geometry, local trend, and wider-market "
                            "state outperform the level-only model and every frozen ablation?"
                        ),
                        "candidate": candidate,
                        "baseline": control,
                        "control_type": control_type,
                        "expected_controls_for_route": len(controls),
                        "seed": seed,
                        "targets": list(TARGETS),
                    }
                )
    return profiles, comparisons


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


def validate_sources() -> dict[str, Any]:
    review = json.loads(JOINT_REVIEW.read_text(encoding="utf-8"))
    if review.get("status") != "completed_generation11_joint_direct_review_and_freqai_queue":
        raise ValueError("Generation 11 joint direct review is not terminal.")
    if review["direct_summary"].get("routes_completed") != len(g11z.ROUTES):
        raise ValueError("All direct siblings must finish before FreqAI freeze.")
    if review["next_batch"].get("direction_prediction") is not False:
        raise ValueError("Broad direction is outside the Generation 11 FreqAI batch.")
    event_manifest = json.loads(g11z.G6_EVENT_MANIFEST.read_text(encoding="utf-8"))
    if len(event_manifest.get("tasks", [])) != 20:
        raise ValueError("Generation 11 FreqAI requires both complete ten-coin cohorts.")
    return {
        "generation11_joint_review": artifact(JOINT_REVIEW),
        "generation6_causal_event_manifest": artifact(g11z.G6_EVENT_MANIFEST),
    }


def cohort_settings(cohort: str) -> dict[str, Any]:
    if cohort == "normal":
        return {
            "timerange": "20240101-20260401",
            "train_days": 730,
            "backtest_days": 90,
            "validation_periods": list(g11z.VALIDATION_PERIODS[cohort]),
        }
    if cohort == "meme":
        return {
            "timerange": "20260101-20260714",
            "train_days": 160,
            "backtest_days": 30,
            "validation_periods": list(g11z.VALIDATION_PERIODS[cohort]),
        }
    raise ValueError(cohort)


def validate_existing_freeze(path: Path = FREEZE_PATH) -> dict[str, Any]:
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_generation11_freqai_model_outcomes":
        raise ValueError("Generation 11 FreqAI freeze is not terminal.")
    for source in frozen["source_contracts"].values():
        path = Path(source["path"])
        if not path.is_file() or g0.sha256_file(path) != source["sha256"]:
            raise ValueError(f"Generation 11 FreqAI frozen source changed: {path}")
    return frozen


def freeze_freqai_batch(*, overwrite: bool = False) -> dict[str, Any]:
    if FREEZE_PATH.is_file() and not overwrite:
        return validate_existing_freeze()
    sources = validate_sources()
    profiles, comparisons = build_registry()
    manifest = json.loads(g11z.G6_EVENT_MANIFEST.read_text(encoding="utf-8"))
    pairs = {
        cohort: [
            str(task["pair"])
            for task in manifest["tasks"]
            if str(task["cohort"]) == cohort
        ]
        for cohort in g11z.COHORTS
    }
    frozen = {
        "schema_version": 1,
        "generation": 11,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation11_freqai_model_outcomes",
        "evidence_label": "reused_period_exploratory_freqai_attribution",
        "plain_objective": (
            "Test whether seven distinct causal information routes and one bounded "
            "multi-family combination improve multi-horizon reaction and relative-volume "
            "forecasts at independent calculated-area contacts."
        ),
        "question_count": 7,
        "route_count": len({item["route_id"] for item in comparisons}),
        "questions": list(QUESTION_DEFINITIONS),
        "current_feature_blocks": list(CURRENT_BLOCKS),
        "feature_block_contract": {
            block: list(suffixes) for block, suffixes in FEATURE_SUFFIXES.items()
        },
        "targets": list(TARGETS),
        "target_interpretation": {
            "reaction": (
                "Price reaches the larger of 0.5 prior ATR or the causal zone half-width "
                "and mean future volume is at least 1.25 times its trailing median."
            ),
            "volume_ratio": "Mean future volume divided by the causal trailing 24-hour median.",
            "horizons_hours": list(g11z.HORIZONS),
        },
        "cohorts": list(g11z.COHORTS),
        "pairs": pairs,
        "cohort_settings": {
            cohort: cohort_settings(cohort) for cohort in g11z.COHORTS
        },
        "initial_seed": INITIAL_SEED,
        "profiles": profiles,
        "profile_count": len(profiles),
        "comparisons": comparisons,
        "comparison_count": len(comparisons),
        "decision_rule": {
            "initial_point_lead": (
                "Candidate has positive equal-coin paired absolute-error gain over every "
                "frozen comparator in both validation periods, with at least 50 rows and "
                "five coins per comparison-period."
            ),
            "initial_strict_lead": (
                "The same complete ladder also has a weekly-block bootstrap lower bound "
                "above no improvement and is not carried by one coin."
            ),
            "reaction_target_reporting": (
                "Also report base rate, clipped prediction MAE/Brier-style error, rank AUC, "
                "and top-versus-bottom prediction bucket reaction rates; no universal AUC gate."
            ),
            "confirmation": (
                "Only routes retained after the complete normal-plus-meme initial batch may "
                "be frozen for new model seeds or genuinely later chronological data."
            ),
        },
        "sequencing": {
            "all_profiles_and_comparisons_frozen_before_cache_outcomes": True,
            "outcome_blind_cache_support_before_target_materialization": True,
            "both_cohorts_complete_before_joint_review": True,
            "no_result_inspired_descendant_until_joint_review": True,
        },
        "runtime": {
            "maximum_profile_workers": 4,
            "model_threads_per_profile": 1,
            "model": "LightGBMRegressorMultiTarget through FreqAI",
        },
        "research_boundary": {
            "profit_used": False,
            "future_signed_direction": False,
            "entry_exit_construction": False,
            "strategy_promotion": False,
        },
        "source_contracts": sources,
    }
    FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(frozen, FREEZE_PATH)
    return frozen


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze Generation 11 FreqAI attribution.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = freeze_freqai_batch(overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "status": result["status"],
                "questions": result["question_count"],
                "routes": result["route_count"],
                "profiles": result["profile_count"],
                "comparisons": result["comparison_count"],
                "targets": len(result["targets"]),
                "freeze_path": str(FREEZE_PATH.resolve()),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
