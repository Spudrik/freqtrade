"""Freeze the balanced Generation 12 chronology, attribution, and combination batch."""

from __future__ import annotations

# Keep imports deterministic and single-threaded while constructing the registry.
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
    market_reaction_zone_freqai_generation11 as g11,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation11_freqai_cache as g11c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation11_freqai_freeze as g11z,
)


FREEZE_ID = "g12_balanced_followup_freeze_20260822a"
FREEZE_PATH = g11.RECORD_ROOT / f"{FREEZE_ID}.json"
SEED = 42
TARGETS = g11z.TARGETS

MINIMAL = g11z.MINIMAL
LEVEL = g11z.LEVEL
GEOMETRY = g11z.GEOMETRY
TREND = g11z.TREND
ACTIVITY = g11z.ACTIVITY
MARKET = g11z.MARKET

STAGE_RECENT = "recent_normal_chronology"
STAGE_ATTRIBUTION = "subfamily_attribution_and_combinations"
STAGES = (STAGE_RECENT, STAGE_ATTRIBUTION)

RECENT_PERIODS = (
    {
        "period": "recent_confirmation_early",
        "start": "2026-04-01T00:00:00Z",
        "stop": "2026-07-01T00:00:00Z",
    },
    {
        "period": "recent_confirmation_late",
        "start": "2026-07-01T00:00:00Z",
        "stop": "2026-08-21T00:00:00Z",
    },
)

SUBFAMILIES: dict[str, dict[str, tuple[str, ...]]] = {
    ACTIVITY: {
        "activity_participation": (
            "relative_volume",
            "volume_acceleration",
            "absolute_pressure",
            "pressure_persistence",
        ),
        "activity_volatility": (
            "atr_fraction",
            "prior_range_atr",
            "bollinger_width",
            "range_contraction",
        ),
    },
    TREND: {
        "trend_direction_strength": (
            "ema20_slope",
            "ma_separation",
            "return_slope",
            "return_acceleration",
            "adx14",
        ),
        "trend_oscillator_momentum": (
            "rsi14_centered",
            "rsi_change",
            "macd_histogram",
            "macd_histogram_change",
        ),
    },
    MARKET: {
        "market_btc_eth_leaders": (
            "btc_return_1h",
            "btc_return_4h",
            "btc_return_24h",
            "btc_relative_volume",
            "eth_return_1h",
            "eth_return_4h",
            "eth_return_24h",
        ),
        "market_cohort_state": (
            "cohort_breadth_positive",
            "cohort_dispersion",
            "cohort_absolute_activity",
            "cohort_ready_members",
        ),
        "market_pair_relative_state": (
            "pair_minus_btc_1h",
            "pair_minus_btc_24h",
            "pair_btc_corr_168h",
            "pair_btc_beta_168h",
        ),
    },
}


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


def block_columns(
    block: str,
    *,
    suffixes: Sequence[str] | None = None,
    variant: str = "current",
) -> tuple[str, ...]:
    selected = tuple(suffixes or g11z.FEATURE_SUFFIXES[block])
    prefix = block if variant == "current" else f"{block}_{variant}"
    return tuple(f"{prefix}__{suffix}" for suffix in selected)


def base_columns() -> tuple[str, ...]:
    return (*block_columns(MINIMAL), *block_columns(LEVEL))


def variant_ready(block: str, variant: str) -> str:
    return block if variant == "current" else f"{block}_{variant}"


def profile_id(cohort: str, name: str, seed: int = SEED) -> str:
    return f"g12__{cohort}__{name}__seed{seed}"


def register_profile(
    profiles: dict[str, dict[str, Any]],
    *,
    cohort: str,
    name: str,
    feature_columns: Sequence[str],
    ready_blocks: Sequence[str],
    seed: int = SEED,
) -> str:
    identifier = profile_id(cohort, name, seed)
    definition = {
        "profile_id": identifier,
        "cohort": cohort,
        "role": name,
        "seed": seed,
        "feature_columns": list(dict.fromkeys(feature_columns)),
        "required_ready_blocks": list(dict.fromkeys(ready_blocks)),
        "blocks": list(dict.fromkeys(ready_blocks)),
        "targets": list(TARGETS),
    }
    existing = profiles.get(identifier)
    if existing is not None and existing != definition:
        raise ValueError(f"Generation 12 profile collision: {identifier}")
    profiles[identifier] = definition
    return identifier


def add_comparisons(
    comparisons: list[dict[str, Any]],
    *,
    stage: str,
    cohort: str,
    question_id: str,
    route_id: str,
    route_type: str,
    plain_question: str,
    candidate: str,
    controls: Sequence[tuple[str, str]],
) -> None:
    for control_type, baseline in controls:
        comparisons.append(
            {
                "comparison_id": (
                    f"{stage}__{cohort}__{route_id}__vs_{control_type}__seed{SEED}"
                ),
                "stage": stage,
                "cohort": cohort,
                "question_id": question_id,
                "route_id": route_id,
                "route_type": route_type,
                "plain_question": plain_question,
                "candidate": candidate,
                "baseline": baseline,
                "control_type": control_type,
                "expected_controls_for_route": len(controls),
                "seed": SEED,
                "targets": list(TARGETS),
            }
        )


def build_cohort_profiles(
    profiles: dict[str, dict[str, Any]], cohort: str
) -> dict[str, str]:
    names: dict[str, str] = {}
    names["minimal"] = register_profile(
        profiles,
        cohort=cohort,
        name="minimal",
        feature_columns=block_columns(MINIMAL),
        ready_blocks=(MINIMAL,),
    )
    names["level"] = register_profile(
        profiles,
        cohort=cohort,
        name="level",
        feature_columns=base_columns(),
        ready_blocks=(MINIMAL, LEVEL),
    )
    for variant in ("stale", "shuffled"):
        names[f"level_{variant}"] = register_profile(
            profiles,
            cohort=cohort,
            name=f"level_{variant}",
            feature_columns=(
                *block_columns(MINIMAL),
                *block_columns(LEVEL, variant=variant),
            ),
            ready_blocks=(MINIMAL, variant_ready(LEVEL, variant)),
        )
    for block, label in (
        (GEOMETRY, "geometry"),
        (ACTIVITY, "activity_full"),
        (TREND, "trend_full"),
        (MARKET, "market_full"),
    ):
        for variant in ("current", "stale", "shuffled"):
            name = label if variant == "current" else f"{label}_{variant}"
            names[name] = register_profile(
                profiles,
                cohort=cohort,
                name=name,
                feature_columns=(
                    *base_columns(),
                    *block_columns(block, variant=variant),
                ),
                ready_blocks=(MINIMAL, LEVEL, variant_ready(block, variant)),
            )
    for block, definitions in SUBFAMILIES.items():
        for label, suffixes in definitions.items():
            for variant in ("current", "stale", "shuffled"):
                name = label if variant == "current" else f"{label}_{variant}"
                names[name] = register_profile(
                    profiles,
                    cohort=cohort,
                    name=name,
                    feature_columns=(
                        *base_columns(),
                        *block_columns(block, suffixes=suffixes, variant=variant),
                    ),
                    ready_blocks=(MINIMAL, LEVEL, variant_ready(block, variant)),
                )
    combinations = {
        "activity_plus_trend": (ACTIVITY, TREND),
        "activity_plus_market": (ACTIVITY, MARKET),
        "activity_plus_geometry": (ACTIVITY, GEOMETRY),
        "activity_plus_trend_plus_market": (ACTIVITY, TREND, MARKET),
    }
    for label, blocks in combinations.items():
        for variant in ("current", "stale", "shuffled"):
            name = label if variant == "current" else f"{label}_{variant}"
            columns = [*base_columns()]
            ready = [MINIMAL, LEVEL]
            for block in blocks:
                columns.extend(block_columns(block, variant=variant))
                ready.append(variant_ready(block, variant))
            names[name] = register_profile(
                profiles,
                cohort=cohort,
                name=name,
                feature_columns=columns,
                ready_blocks=ready,
            )
    return names


def build_recent_comparisons(
    comparisons: list[dict[str, Any]], cohort: str, names: dict[str, str]
) -> None:
    if cohort != "normal":
        return
    definitions = (
        (
            "level_identity_and_timeframe",
            names["level"],
            (
                ("immediately_simpler", names["minimal"]),
                ("causal_stale", names["level_stale"]),
                ("within_period_nonself_shuffle", names["level_shuffled"]),
            ),
        ),
        (
            "isolated_and_cluster_geometry",
            names["geometry"],
            (
                ("immediately_simpler", names["level"]),
                ("causal_stale", names["geometry_stale"]),
                ("within_period_nonself_shuffle", names["geometry_shuffled"]),
            ),
        ),
        *(
            (
                route,
                names[label],
                (
                    ("immediately_simpler", names["level"]),
                    ("causal_stale", names[f"{label}_stale"]),
                    ("within_period_nonself_shuffle", names[f"{label}_shuffled"]),
                ),
            )
            for route, label in (
                ("local_activity_and_volatility", "activity_full"),
                ("local_trend_and_momentum", "trend_full"),
                ("wider_crypto_market_alignment", "market_full"),
            )
        ),
    )
    for route, candidate, controls in definitions:
        add_comparisons(
            comparisons,
            stage=STAGE_RECENT,
            cohort=cohort,
            question_id=f"recent__{route}",
            route_id=route,
            route_type="recent_chronological_retest",
            plain_question=(
                f"Does the frozen {route.replace('_', ' ')} relationship survive the "
                "later April-August 2026 normal-market chronology?"
            ),
            candidate=candidate,
            controls=controls,
        )


def build_attribution_comparisons(
    comparisons: list[dict[str, Any]], cohort: str, names: dict[str, str]
) -> None:
    for block, definitions in SUBFAMILIES.items():
        for label in definitions:
            add_comparisons(
                comparisons,
                stage=STAGE_ATTRIBUTION,
                cohort=cohort,
                question_id=f"decompose__{block}",
                route_id=label,
                route_type="one_subfamily_attribution",
                plain_question=(
                    f"Does {label.replace('_', ' ')} add reaction information beyond "
                    "calculated-level state and its stale and shuffled versions?"
                ),
                candidate=names[label],
                controls=(
                    ("level_only", names["level"]),
                    ("causal_stale", names[f"{label}_stale"]),
                    ("within_period_nonself_shuffle", names[f"{label}_shuffled"]),
                ),
            )
    full_definitions = (
        (
            "activity_full_decomposition",
            "activity_full",
            ("activity_participation", "activity_volatility"),
        ),
        (
            "trend_full_decomposition",
            "trend_full",
            ("trend_direction_strength", "trend_oscillator_momentum"),
        ),
        (
            "market_full_decomposition",
            "market_full",
            (
                "market_btc_eth_leaders",
                "market_cohort_state",
                "market_pair_relative_state",
            ),
        ),
    )
    for route, full, subsets in full_definitions:
        controls = [
            ("level_only", names["level"]),
            *((f"subset_{subset}", names[subset]) for subset in subsets),
            ("causal_stale", names[f"{full}_stale"]),
            ("within_period_nonself_shuffle", names[f"{full}_shuffled"]),
        ]
        add_comparisons(
            comparisons,
            stage=STAGE_ATTRIBUTION,
            cohort=cohort,
            question_id=f"decompose__{full}",
            route_id=route,
            route_type="full_family_decomposition",
            plain_question=(
                f"Does {full.replace('_', ' ')} add information beyond every frozen "
                "subfamily as well as level-only, stale, and shuffled controls?"
            ),
            candidate=names[full],
            controls=controls,
        )


def build_combination_comparisons(
    comparisons: list[dict[str, Any]], cohort: str, names: dict[str, str]
) -> None:
    routes = (
        (
            "activity_plus_trend",
            ("activity_full", "trend_full"),
        ),
        (
            "activity_plus_market",
            ("activity_full", "market_full"),
        ),
        (
            "activity_plus_geometry",
            ("activity_full", "geometry"),
        ),
    )
    for route, components in routes:
        controls = [
            *((f"component_{name}", names[name]) for name in components),
            ("level_only", names["level"]),
            ("all_additions_causal_stale", names[f"{route}_stale"]),
            (
                "all_additions_within_period_nonself_shuffle",
                names[f"{route}_shuffled"],
            ),
        ]
        add_comparisons(
            comparisons,
            stage=STAGE_ATTRIBUTION,
            cohort=cohort,
            question_id="bounded_pairwise_combinations",
            route_id=route,
            route_type="bounded_two_family_combination",
            plain_question=(
                f"Does {route.replace('_', ' ')} improve on both component models, "
                "level-only, stale, and shuffled controls?"
            ),
            candidate=names[route],
            controls=controls,
        )
    triple = "activity_plus_trend_plus_market"
    add_comparisons(
        comparisons,
        stage=STAGE_ATTRIBUTION,
        cohort=cohort,
        question_id="bounded_three_family_combination",
        route_id=triple,
        route_type="bounded_three_family_combination",
        plain_question=(
            "Does local activity plus local trend plus wider crypto state improve on "
            "both activity-led pairwise models, every single family, level-only, stale, "
            "and shuffled controls?"
        ),
        candidate=names[triple],
        controls=(
            ("pair_activity_trend", names["activity_plus_trend"]),
            ("pair_activity_market", names["activity_plus_market"]),
            ("single_activity", names["activity_full"]),
            ("single_trend", names["trend_full"]),
            ("single_market", names["market_full"]),
            ("level_only", names["level"]),
            ("all_additions_causal_stale", names[f"{triple}_stale"]),
            (
                "all_additions_within_period_nonself_shuffle",
                names[f"{triple}_shuffled"],
            ),
        ),
    )


def build_registry() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    profiles: dict[str, dict[str, Any]] = {}
    comparisons: list[dict[str, Any]] = []
    for cohort in ("normal", "meme"):
        names = build_cohort_profiles(profiles, cohort)
        build_recent_comparisons(comparisons, cohort, names)
        build_attribution_comparisons(comparisons, cohort, names)
        build_combination_comparisons(comparisons, cohort, names)
    return profiles, comparisons


def validate_sources() -> dict[str, Any]:
    joint_path = (
        g11.RECORD_ROOT
        / "g11_initial_joint_review_20260822a"
        / "g11_joint_freqai_review.json"
    )
    joint = json.loads(joint_path.read_text(encoding="utf-8"))
    if joint.get("status") != "completed_generation11_normal_meme_joint_freqai_review":
        raise ValueError("Generation 11 joint review is not terminal.")
    if joint.get("route_targets_reviewed") != 56:
        raise ValueError("Generation 11 joint route coverage drifted.")
    if joint.get("automatic_descendant_launch") is not False:
        raise ValueError("Generation 11 descendant sequencing drifted.")
    sources = {
        "generation11_joint_review": artifact(joint_path),
        "generation11_freqai_freeze": artifact(g11z.FREEZE_PATH),
    }
    for cohort in ("normal", "meme"):
        cache_path = g11c.RECORD_ROOT / f"{cohort}_manifest.json"
        cache = json.loads(cache_path.read_text(encoding="utf-8"))
        if cache.get("status") != "completed_generation11_freqai_cache":
            raise ValueError(f"Generation 11 {cohort} cache is not terminal.")
        sources[f"generation11_{cohort}_cache"] = artifact(cache_path)
    return sources


def validate_existing_freeze(path: Path = FREEZE_PATH) -> dict[str, Any]:
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_generation12_outcomes":
        raise ValueError("Generation 12 freeze is not terminal.")
    for source in frozen["source_contracts"].values():
        source_path = Path(source["path"])
        if not source_path.is_file() or g0.sha256_file(source_path) != source["sha256"]:
            raise ValueError(f"Generation 12 frozen source changed: {source_path}")
    return frozen


def freeze_generation12(*, overwrite: bool = False) -> dict[str, Any]:
    if FREEZE_PATH.is_file() and not overwrite:
        return validate_existing_freeze()
    sources = validate_sources()
    profiles, comparisons = build_registry()
    stage_counts = {
        stage: len([item for item in comparisons if item["stage"] == stage])
        for stage in STAGES
    }
    frozen = {
        "schema_version": 1,
        "generation": 12,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation12_outcomes",
        "plain_objective": (
            "Challenge the complete Generation 11 leads on later normal-market history, "
            "identify which trader-readable subfamilies carry the activity/trend/market "
            "signal, and test four restrained combinations without allowing any early "
            "combination result to redirect the sibling batch."
        ),
        "evidence_boundaries": {
            STAGE_RECENT: (
                "Later chronological stress test. The dates are not globally pristine: "
                "previous research opened some other targets and relative-volume questions."
            ),
            STAGE_ATTRIBUTION: (
                "Reused-period exploratory mechanism attribution; not independent temporal "
                "confirmation."
            ),
        },
        "stages": list(STAGES),
        "recent_periods": list(RECENT_PERIODS),
        "recent_settings": {
            "cohort": "normal",
            "timerange": "20260401-20260821",
            "train_days": 730,
            "backtest_days": 30,
            "source_period": "exposed_recent_diagnostic",
        },
        "attribution_settings": {
            cohort: dict(g11z.cohort_settings(cohort))
            for cohort in ("normal", "meme")
        },
        "targets": list(TARGETS),
        "subfamilies": {
            block: {name: list(suffixes) for name, suffixes in definitions.items()}
            for block, definitions in SUBFAMILIES.items()
        },
        "profiles": profiles,
        "profile_count": len(profiles),
        "comparisons": comparisons,
        "comparison_count": len(comparisons),
        "stage_comparison_counts": stage_counts,
        "decision_rule": {
            "point_lead": (
                "For one route and target, the candidate must have positive equal-coin "
                "paired absolute-error gain over every frozen comparator in both declared "
                "evaluation periods, with at least five positive coins and adequate rows."
            ),
            "strict_lead": (
                "The complete point ladder must also have weekly-block bootstrap lower "
                "bounds above no improvement and must not be dominated by one coin."
            ),
            "combination": (
                "A combination receives credit only if it beats every declared component, "
                "level-only, stale, and shuffled control. More features are not presumed "
                "better."
            ),
        },
        "sequencing": {
            "all_stage_profiles_frozen_before_any_generation12_outcome": True,
            "recent_and_attribution_stages_complete_before_joint_review": True,
            "no_generation13_descendant_before_joint_review": True,
        },
        "parallel_queue_not_executed_in_generation12": [
            {
                "plain_question": (
                    "Does the leading context still add at matched ordinary/no-level "
                    "timestamps, or is it merely general activity persistence?"
                ),
                "route": "matched_no_level_and_near_miss_controls",
            },
            {
                "plain_question": (
                    "Do timestamp-safe 4h, 8h, and 1d indicator states add beyond the "
                    "current 1h indicator context?"
                ),
                "route": "multi_timeframe_indicator_context",
            },
            {
                "plain_question": (
                    "Do 12h, 24h, and 48h reaction paths change the conclusions for 4h, "
                    "8h, and 1d source levels?"
                ),
                "route": "extended_reaction_horizons",
            },
        ],
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
            "joint_55_percent_direction_target_reached": False,
        },
        "source_contracts": sources,
    }
    FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(frozen, FREEZE_PATH)
    return frozen


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze Generation 12 follow-up batch.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = freeze_generation12(overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "status": result["status"],
                "profiles": result["profile_count"],
                "comparisons": result["comparison_count"],
                "stage_comparisons": result["stage_comparison_counts"],
                "freeze_path": str(FREEZE_PATH.resolve()),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
