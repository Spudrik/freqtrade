"""Freeze the broad Generation 13 control, timeframe, and horizon siblings."""

from __future__ import annotations

# Bind numerical pools before pandas/FreqAI-adjacent imports.
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
    market_reaction_zone_generation11_freeze as g11base,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation11_freqai_freeze as g11f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation12_freeze as g12z,
)


OUTPUT_ROOT = g11base.OUTPUT_ROOT
G12_JOINT_REVIEW = (
    OUTPUT_ROOT
    / "generation11_review"
    / "generation11_branches"
    / "g12_chronology_attribution_and_combinations"
    / "g12_joint_review_20260822a"
    / "g12_joint_review.json"
)
FREEZE_PATH = (
    OUTPUT_ROOT
    / "generation11_review"
    / "generation11_branches"
    / "g12_chronology_attribution_and_combinations"
    / "g13_broad_siblings_freeze_20260822a.json"
)
G11_CACHE_ROOT = OUTPUT_ROOT / "generation11_shared" / "g11_freqai_cache_20260822a"
G6_EVENT_MANIFEST = g11base.G6_EVENT_MANIFEST

COHORTS = ("normal", "meme")
STANDARD = "standard_validation"
RECENT = "recent_normal_chronology"
WINDOWS = (STANDARD, RECENT)
MTF_STAGE = "multi_timeframe_indicator_context"
LONG_STAGE = "extended_reaction_horizons"
STAGES = (MTF_STAGE, LONG_STAGE)
SEED = 42
MTF_TIMEFRAMES = ("4h", "8h", "1d")
LONG_HORIZONS = (12, 24, 48)
MTF_TARGETS = tuple(
    [
        *(f"&-g11_reaction_h{horizon}" for horizon in g11base.HORIZONS),
        *(f"&-g11_volume_ratio_h{horizon}" for horizon in g11base.HORIZONS),
    ]
)
LONG_TARGETS = tuple(
    [
        *(f"&-g13_reaction_h{horizon}" for horizon in LONG_HORIZONS),
        *(f"&-g13_volume_ratio_h{horizon}" for horizon in LONG_HORIZONS),
    ]
)

MINIMAL = g11f.MINIMAL
LEVEL = g11f.LEVEL
ACTIVITY = g11f.ACTIVITY
TREND = g11f.TREND
MARKET = g11f.MARKET

MTF_FAMILIES = {
    "activity_volatility": tuple(g11f.FEATURE_SUFFIXES[ACTIVITY]),
    "trend_momentum": tuple(g11f.FEATURE_SUFFIXES[TREND]),
}
LONG_FAMILIES = {
    "local_activity_volatility": (ACTIVITY, tuple(g11f.FEATURE_SUFFIXES[ACTIVITY])),
    "local_oscillator_momentum": (
        TREND,
        tuple(g12z.SUBFAMILIES[TREND]["trend_oscillator_momentum"]),
    ),
    "btc_eth_leader_state": (
        MARKET,
        tuple(g12z.SUBFAMILIES[MARKET]["market_btc_eth_leaders"]),
    ),
}
DIRECT_CONTROLS = ("matched_random_time", "near_miss")
DIRECT_MATCH_FEATURES = tuple(
    f"state__{suffix}" for suffix in g11f.FEATURE_SUFFIXES[ACTIVITY]
)
DIRECT_BALANCE_AUDIT_FEATURES = tuple(
    [
        *(f"state__{suffix}" for suffix in g11f.FEATURE_SUFFIXES[TREND]),
        "xm_btc_return_1h",
        "xm_btc_return_4h",
        "xm_eth_return_1h",
        "xm_eth_return_4h",
        "xm_cohort_breadth_positive",
        "xm_cohort_absolute_activity",
    ]
)


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


def block_columns(
    block: str,
    suffixes: Sequence[str] | None = None,
    *,
    variant: str = "current",
) -> tuple[str, ...]:
    selected = tuple(suffixes or g11f.FEATURE_SUFFIXES[block])
    prefix = block if variant == "current" else f"{block}_{variant}"
    return tuple(f"{prefix}__{suffix}" for suffix in selected)


def mtf_block(timeframe: str, family: str, variant: str = "current") -> str:
    base = f"g13_mtf_{timeframe}_{family}"
    return base if variant == "current" else f"{base}_{variant}"


def mtf_columns(
    timeframe: str, family: str, *, variant: str = "current"
) -> tuple[str, ...]:
    return tuple(
        f"{mtf_block(timeframe, family, variant)}__{suffix}"
        for suffix in MTF_FAMILIES[family]
    )


def base_columns() -> tuple[str, ...]:
    return (
        *block_columns(MINIMAL),
        *block_columns(LEVEL),
    )


def mtf_base_columns() -> tuple[str, ...]:
    return (*base_columns(), *block_columns(ACTIVITY))


def profile_id(stage: str, cohort: str, role: str) -> str:
    return f"g13__{stage}__{cohort}__{role}__seed{SEED}"


def register_profile(
    profiles: dict[str, dict[str, Any]],
    *,
    stage: str,
    cohort: str,
    role: str,
    feature_columns: Sequence[str],
    ready_blocks: Sequence[str],
    targets: Sequence[str],
) -> str:
    identifier = profile_id(stage, cohort, role)
    definition = {
        "profile_id": identifier,
        "stage": stage,
        "cohort": cohort,
        "role": role,
        "seed": SEED,
        "feature_columns": list(dict.fromkeys(feature_columns)),
        "required_ready_blocks": list(dict.fromkeys(ready_blocks)),
        "targets": list(targets),
    }
    existing = profiles.get(identifier)
    if existing is not None and existing != definition:
        raise ValueError(f"Generation 13 profile collision: {identifier}")
    profiles[identifier] = definition
    return identifier


def add_comparisons(
    comparisons: list[dict[str, Any]],
    *,
    stage: str,
    cohort: str,
    route_id: str,
    plain_question: str,
    candidate: str,
    controls: Sequence[tuple[str, str]],
    targets: Sequence[str],
) -> None:
    for control_type, baseline in controls:
        comparisons.append(
            {
                "comparison_id": (
                    f"{stage}__{cohort}__{route_id}__vs_{control_type}__seed{SEED}"
                ),
                "stage": stage,
                "cohort": cohort,
                "question_id": route_id,
                "route_id": route_id,
                "route_type": stage,
                "plain_question": plain_question,
                "candidate": candidate,
                "baseline": baseline,
                "control_type": control_type,
                "expected_controls_for_route": len(controls),
                "seed": SEED,
                "targets": list(targets),
            }
        )


def build_mtf_registry(
    profiles: dict[str, dict[str, Any]],
    comparisons: list[dict[str, Any]],
    cohort: str,
) -> None:
    base = register_profile(
        profiles,
        stage=MTF_STAGE,
        cohort=cohort,
        role="one_hour_activity_base",
        feature_columns=mtf_base_columns(),
        ready_blocks=(MINIMAL, LEVEL, ACTIVITY),
        targets=MTF_TARGETS,
    )
    for timeframe in MTF_TIMEFRAMES:
        for family in MTF_FAMILIES:
            route = f"{timeframe}_{family}"
            names: dict[str, str] = {}
            for variant in ("current", "stale", "shuffled"):
                role = route if variant == "current" else f"{route}_{variant}"
                names[variant] = register_profile(
                    profiles,
                    stage=MTF_STAGE,
                    cohort=cohort,
                    role=role,
                    feature_columns=(
                        *mtf_base_columns(),
                        *mtf_columns(timeframe, family, variant=variant),
                    ),
                    ready_blocks=(
                        MINIMAL,
                        LEVEL,
                        ACTIVITY,
                        mtf_block(timeframe, family, variant),
                    ),
                    targets=MTF_TARGETS,
                )
            add_comparisons(
                comparisons,
                stage=MTF_STAGE,
                cohort=cohort,
                route_id=route,
                plain_question=(
                    f"Does causal {timeframe} {family.replace('_', ' ')} state add "
                    "reaction information beyond current 1h activity and calculated-level "
                    "state?"
                ),
                candidate=names["current"],
                controls=(
                    ("one_hour_context_only", base),
                    ("causal_72h_old_higher_timeframe", names["stale"]),
                    ("within_period_shuffled_higher_timeframe", names["shuffled"]),
                ),
                targets=MTF_TARGETS,
            )


def build_long_registry(
    profiles: dict[str, dict[str, Any]],
    comparisons: list[dict[str, Any]],
    cohort: str,
) -> None:
    level = register_profile(
        profiles,
        stage=LONG_STAGE,
        cohort=cohort,
        role="level_only",
        feature_columns=base_columns(),
        ready_blocks=(MINIMAL, LEVEL),
        targets=LONG_TARGETS,
    )
    for route, (block, suffixes) in LONG_FAMILIES.items():
        names: dict[str, str] = {}
        for variant in ("current", "stale", "shuffled"):
            role = route if variant == "current" else f"{route}_{variant}"
            names[variant] = register_profile(
                profiles,
                stage=LONG_STAGE,
                cohort=cohort,
                role=role,
                feature_columns=(
                    *base_columns(),
                    *block_columns(block, suffixes, variant=variant),
                ),
                ready_blocks=(
                    MINIMAL,
                    LEVEL,
                    block if variant == "current" else f"{block}_{variant}",
                ),
                targets=LONG_TARGETS,
            )
        add_comparisons(
            comparisons,
            stage=LONG_STAGE,
            cohort=cohort,
            route_id=route,
            plain_question=(
                f"Does {route.replace('_', ' ')} help distinguish sustained unsigned "
                "reaction and volume over 12h, 24h, and 48h at 4h, 8h, and 1d levels?"
            ),
            candidate=names["current"],
            controls=(
                ("level_only", level),
                ("causal_72h_old_context", names["stale"]),
                ("within_period_shuffled_context", names["shuffled"]),
            ),
            targets=LONG_TARGETS,
        )


def build_registry() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    profiles: dict[str, dict[str, Any]] = {}
    comparisons: list[dict[str, Any]] = []
    for cohort in COHORTS:
        build_mtf_registry(profiles, comparisons, cohort)
        build_long_registry(profiles, comparisons, cohort)
    return profiles, comparisons


def validate_source_results() -> dict[str, Any]:
    if not G12_JOINT_REVIEW.is_file():
        raise FileNotFoundError(G12_JOINT_REVIEW)
    joint = json.loads(G12_JOINT_REVIEW.read_text(encoding="utf-8"))
    if joint.get("status") != "completed_generation12_joint_review":
        raise ValueError("Generation 12 joint review is not terminal.")
    sources: dict[str, Any] = {
        "generation12_joint_review": artifact(G12_JOINT_REVIEW),
        "generation6_event_manifest": artifact(G6_EVENT_MANIFEST),
    }
    for cohort in COHORTS:
        path = G11_CACHE_ROOT / f"{cohort}_manifest.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed_generation11_freqai_cache":
            raise ValueError(f"Generation 11 cache is not terminal for {cohort}.")
        sources[f"generation11_{cohort}_cache"] = artifact(path)
    return sources


def freeze_generation13(*, overwrite: bool = False) -> dict[str, Any]:
    if FREEZE_PATH.is_file() and not overwrite:
        existing = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        if existing.get("status") != "frozen_before_generation13_outcomes":
            raise ValueError("Existing Generation 13 freeze has an invalid status.")
        return existing
    sources = validate_source_results()
    profiles, comparisons = build_registry()
    frozen = {
        "schema_version": 1,
        "generation": 13,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation13_outcomes",
        "plain_objective": (
            "Complete three broad siblings before branching: distinguish calculated-level "
            "reaction from general busy-market persistence using tightly matched ordinary "
            "and near-miss controls; test causal 4h, 8h, and 1d indicator state beyond 1h "
            "context; and extend unsigned reaction paths to 12h, 24h, and 48h."
        ),
        "siblings": {
            "matched_no_level_and_near_miss_controls": {
                "cohorts": list(COHORTS),
                "controls": list(DIRECT_CONTROLS),
                "match_features": list(DIRECT_MATCH_FEATURES),
                "balance_audit_features": list(DIRECT_BALANCE_AUDIT_FEATURES),
                "horizons_hours": list(g11base.HORIZONS),
                "plain_question": (
                    "After current activity and volatility are closely matched, do actual "
                    "calculated-level contacts react more often than ordinary no-level "
                    "timestamps and genuine near misses?"
                ),
            },
            MTF_STAGE: {
                "timeframes": list(MTF_TIMEFRAMES),
                "families": {
                    key: list(value) for key, value in MTF_FAMILIES.items()
                },
                "targets": list(MTF_TARGETS),
            },
            LONG_STAGE: {
                "source_timeframes": ["4h", "8h", "1d"],
                "horizons_hours": list(LONG_HORIZONS),
                "families": {
                    key: {"source_block": value[0], "suffixes": list(value[1])}
                    for key, value in LONG_FAMILIES.items()
                },
                "targets": list(LONG_TARGETS),
                "reaction_definition": (
                    "Unsigned excursion from the calculated area reaches the larger of "
                    "0.5 ATR and zone half-width, while mean future volume across the "
                    "declared horizon is at least 1.25 times the causal prior-24h median."
                ),
            },
        },
        "evaluation_windows": {
            STANDARD: {
                "normal": list(g11base.VALIDATION_PERIODS["normal"]),
                "meme": list(g11base.VALIDATION_PERIODS["meme"]),
            },
            RECENT: {
                "normal": list(g12z.RECENT_PERIODS),
                "meme": [],
            },
        },
        "profiles": profiles,
        "profile_count": len(profiles),
        "comparisons": comparisons,
        "comparison_count": len(comparisons),
        "decision_rule": {
            "direct_point": (
                "Actual-minus-control equal-coin reaction-rate difference must be positive "
                "for both controls in every declared period with at least five coins and "
                "adequately balanced causal state."
            ),
            "direct_strict": (
                "The complete direct point ladder must also have weekly-block bootstrap "
                "lower bounds above zero."
            ),
            "freqai_point": (
                "The candidate must reduce equal-coin absolute error against all three "
                "comparators in both declared periods, with at least five positive coins."
            ),
            "freqai_strict": (
                "The complete point ladder must also have weekly-block bootstrap lower "
                "bounds above zero and no single-coin domination."
            ),
            "long_horizon_scope": (
                "Long-horizon conclusions are reported separately for 4h, 8h, and 1d "
                "source levels; pooled support cannot conceal a failed source timeframe."
            ),
        },
        "sequencing": {
            "all_siblings_frozen_before_any_generation13_outcome": True,
            "all_direct_and_freqai_cells_terminal_before_joint_review": True,
            "no_generation14_descendant_before_joint_review": True,
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
            "joint_55_percent_direction_target_reached": False,
        },
        "source_contracts": sources,
    }
    FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(frozen, FREEZE_PATH)
    return frozen


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze broad Generation 13 siblings.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = freeze_generation13(overwrite=args.overwrite)
    print(
        json.dumps(
            {
                "status": result["status"],
                "profiles": result["profile_count"],
                "comparisons": result["comparison_count"],
                "freeze_path": str(FREEZE_PATH.resolve()),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
