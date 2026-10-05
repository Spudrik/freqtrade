"""Freeze Generation 14's broad robustness and restrained-combination siblings."""

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
    market_reaction_zone_generation11_freqai_freeze as g11f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation12_freeze as g12z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation13_cache as g13c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation13_freeze as g13z,
)


OUTPUT_ROOT = g13z.OUTPUT_ROOT
G13_JOINT_REVIEW = (
    g13z.FREEZE_PATH.parent
    / "g13_broad_siblings"
    / "freqai"
    / "g13_joint_review_20260822a"
    / "g13_joint_review.json"
)
FREEZE_PATH = (
    g13z.FREEZE_PATH.parent
    / "g13_broad_siblings"
    / "g14_broad_combination_freeze_20260822a.json"
)

COHORTS = ("normal", "meme")
STANDARD = g13z.STANDARD
RECENT = g13z.RECENT
MTF_STAGE = "eight_hour_context_combinations"
LONG_STAGE = "four_hour_level_long_combinations"
STAGES = (MTF_STAGE, LONG_STAGE)
SEED = 42

MTF_TARGETS = tuple(f"&-g11_reaction_h{horizon}" for horizon in (1, 2, 4, 8))
LONG_TARGETS = tuple(
    [
        *(f"&-g13_reaction_h{horizon}" for horizon in (12, 24, 48)),
        *(f"&-g13_volume_ratio_h{horizon}" for horizon in (12, 24, 48)),
    ]
)

MINIMAL = g11f.MINIMAL
LEVEL = g11f.LEVEL
GEOMETRY = g11f.GEOMETRY
TREND = g11f.TREND
ACTIVITY = g11f.ACTIVITY
MARKET = g11f.MARKET
EIGHT_HOUR = "g13_mtf_8h_activity_volatility"

PARTICIPATION_SUFFIXES = tuple(
    g12z.SUBFAMILIES[ACTIVITY]["activity_participation"]
)
VOLATILITY_SUFFIXES = tuple(g12z.SUBFAMILIES[ACTIVITY]["activity_volatility"])
OSCILLATOR_SUFFIXES = tuple(
    g12z.SUBFAMILIES[TREND]["trend_oscillator_momentum"]
)
MARKET_SUFFIXES = tuple(
    [
        *g12z.SUBFAMILIES[MARKET]["market_btc_eth_leaders"],
        *g12z.SUBFAMILIES[MARKET]["market_cohort_state"],
    ]
)

REACTION_PRICE_THRESHOLDS_ATR = (0.35, 0.5, 0.75)
REACTION_VOLUME_THRESHOLDS = (1.1, 1.25, 1.5)
REACTION_HORIZONS = (1, 2, 4, 8)

NORMAL_GROUPS = {
    "btc_separate": ("BTC/USDT:USDT",),
    "smart_contract_platforms": (
        "ADA/USDT:USDT",
        "AVAX/USDT:USDT",
        "BNB/USDT:USDT",
        "ETH/USDT:USDT",
        "SOL/USDT:USDT",
    ),
    "other_established_alts": (
        "DOGE/USDT:USDT",
        "LINK/USDT:USDT",
        "TRX/USDT:USDT",
        "XRP/USDT:USDT",
    ),
}


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


def eight_hour_columns(
    suffixes: Sequence[str] | None = None,
    *,
    variant: str = "current",
) -> tuple[str, ...]:
    selected = tuple(suffixes or g13z.MTF_FAMILIES["activity_volatility"])
    prefix = EIGHT_HOUR if variant == "current" else f"{EIGHT_HOUR}_{variant}"
    return tuple(f"{prefix}__{suffix}" for suffix in selected)


def ready(block: str, variant: str = "current") -> str:
    return block if variant == "current" else f"{block}_{variant}"


def base_columns(*, include_level: bool = True) -> tuple[str, ...]:
    return (
        *block_columns(MINIMAL),
        *(block_columns(LEVEL) if include_level else ()),
        *block_columns(ACTIVITY),
    )


def profile_id(stage: str, cohort: str, role: str) -> str:
    return f"g14__{stage}__{cohort}__{role}__seed{SEED}"


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
        raise ValueError(f"Generation 14 profile collision: {identifier}")
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
                "plain_question": plain_question,
                "candidate": candidate,
                "baseline": baseline,
                "control_type": control_type,
                "expected_controls_for_route": len(controls),
                "seed": SEED,
                "targets": list(targets),
            }
        )


def mtf_profile(
    profiles: dict[str, dict[str, Any]],
    *,
    cohort: str,
    role: str,
    additions: Sequence[tuple[str, Sequence[str], str]],
    include_level: bool = True,
) -> str:
    columns = list(base_columns(include_level=include_level))
    ready_blocks = [MINIMAL, ACTIVITY]
    if include_level:
        ready_blocks.append(LEVEL)
    for block, suffixes, variant in additions:
        columns.extend(
            eight_hour_columns(suffixes, variant=variant)
            if block == EIGHT_HOUR
            else block_columns(block, suffixes, variant=variant)
        )
        ready_blocks.append(ready(block, variant))
    return register_profile(
        profiles,
        stage=MTF_STAGE,
        cohort=cohort,
        role=role,
        feature_columns=columns,
        ready_blocks=ready_blocks,
        targets=MTF_TARGETS,
    )


def build_mtf_registry(
    profiles: dict[str, dict[str, Any]],
    comparisons: list[dict[str, Any]],
    cohort: str,
) -> None:
    base = mtf_profile(profiles, cohort=cohort, role="one_hour_base", additions=())
    full: dict[str, str] = {}
    for variant in ("current", "stale", "shuffled"):
        full[variant] = mtf_profile(
            profiles,
            cohort=cohort,
            role=f"eight_hour_full_{variant}",
            additions=((EIGHT_HOUR, (), variant),),
        )

    subfamilies: dict[str, dict[str, str]] = {}
    for label, suffixes in (
        ("participation_pressure", PARTICIPATION_SUFFIXES),
        ("volatility_compression", VOLATILITY_SUFFIXES),
    ):
        names: dict[str, str] = {}
        for variant in ("current", "stale", "shuffled"):
            names[variant] = mtf_profile(
                profiles,
                cohort=cohort,
                role=f"eight_hour_{label}_{variant}",
                additions=((EIGHT_HOUR, suffixes, variant),),
            )
        subfamilies[label] = names
        add_comparisons(
            comparisons,
            stage=MTF_STAGE,
            cohort=cohort,
            route_id=f"eight_hour_{label}",
            plain_question=(
                f"Does completed 8h {label.replace('_', ' ')} add reaction information "
                "beyond current 1h state and its stale and shuffled controls?"
            ),
            candidate=names["current"],
            controls=(
                ("one_hour_base", base),
                ("causal_72h_old_eight_hour_subfamily", names["stale"]),
                ("within_period_shuffled_eight_hour_subfamily", names["shuffled"]),
            ),
            targets=MTF_TARGETS,
        )

    add_comparisons(
        comparisons,
        stage=MTF_STAGE,
        cohort=cohort,
        route_id="eight_hour_full_decomposition",
        plain_question=(
            "Does the full 8h activity family add information beyond both its "
            "participation/pressure and volatility/compression halves?"
        ),
        candidate=full["current"],
        controls=(
            ("one_hour_base", base),
            ("participation_pressure_only", subfamilies["participation_pressure"]["current"]),
            ("volatility_compression_only", subfamilies["volatility_compression"]["current"]),
            ("causal_72h_old_full", full["stale"]),
            ("within_period_shuffled_full", full["shuffled"]),
        ),
        targets=MTF_TARGETS,
    )

    level_stale = mtf_profile(
        profiles,
        cohort=cohort,
        role="eight_hour_with_stale_level_identity",
        additions=((EIGHT_HOUR, (), "current"), (LEVEL, (), "stale")),
        include_level=False,
    )
    level_shuffled = mtf_profile(
        profiles,
        cohort=cohort,
        role="eight_hour_with_shuffled_level_identity",
        additions=((EIGHT_HOUR, (), "current"), (LEVEL, (), "shuffled")),
        include_level=False,
    )
    no_level = mtf_profile(
        profiles,
        cohort=cohort,
        role="eight_hour_without_level_identity",
        additions=((EIGHT_HOUR, (), "current"),),
        include_level=False,
    )
    add_comparisons(
        comparisons,
        stage=MTF_STAGE,
        cohort=cohort,
        route_id="eight_hour_level_identity",
        plain_question=(
            "After local and completed 8h activity are known, does calculated-level "
            "family/timeframe identity still add information?"
        ),
        candidate=full["current"],
        controls=(
            ("no_level_identity", no_level),
            ("causal_72h_old_level_identity", level_stale),
            ("within_period_shuffled_level_identity", level_shuffled),
        ),
        targets=MTF_TARGETS,
    )

    for route, block, suffixes in (
        ("eight_hour_cluster_interaction", GEOMETRY, tuple(g11f.FEATURE_SUFFIXES[GEOMETRY])),
        ("eight_hour_wider_market_interaction", MARKET, MARKET_SUFFIXES),
        ("eight_hour_oscillator_interaction", TREND, OSCILLATOR_SUFFIXES),
    ):
        candidate = mtf_profile(
            profiles,
            cohort=cohort,
            role=f"{route}_current",
            additions=((EIGHT_HOUR, (), "current"), (block, suffixes, "current")),
        )
        component = mtf_profile(
            profiles,
            cohort=cohort,
            role=f"{route}_component_without_eight_hour",
            additions=((block, suffixes, "current"),),
        )
        stale = mtf_profile(
            profiles,
            cohort=cohort,
            role=f"{route}_stale_component",
            additions=((EIGHT_HOUR, (), "current"), (block, suffixes, "stale")),
        )
        shuffled = mtf_profile(
            profiles,
            cohort=cohort,
            role=f"{route}_shuffled_component",
            additions=((EIGHT_HOUR, (), "current"), (block, suffixes, "shuffled")),
        )
        add_comparisons(
            comparisons,
            stage=MTF_STAGE,
            cohort=cohort,
            route_id=route,
            plain_question=(
                f"Does {route.replace('eight_hour_', '').replace('_', ' ')} add "
                "information beyond the 8h activity lead and each component alone?"
            ),
            candidate=candidate,
            controls=(
                ("eight_hour_activity_only", full["current"]),
                ("additional_component_without_eight_hour", component),
                ("causal_72h_old_additional_component", stale),
                ("within_period_shuffled_additional_component", shuffled),
            ),
            targets=MTF_TARGETS,
        )


def build_long_registry(
    profiles: dict[str, dict[str, Any]],
    comparisons: list[dict[str, Any]],
    cohort: str,
) -> None:
    def make(
        role: str,
        *,
        local_variant: str | None,
        eight_variant: str | None,
    ) -> str:
        columns = [*block_columns(MINIMAL), *block_columns(LEVEL)]
        readiness = [MINIMAL, LEVEL]
        if local_variant is not None:
            columns.extend(block_columns(ACTIVITY, variant=local_variant))
            readiness.append(ready(ACTIVITY, local_variant))
        if eight_variant is not None:
            columns.extend(eight_hour_columns(variant=eight_variant))
            readiness.append(ready(EIGHT_HOUR, eight_variant))
        return register_profile(
            profiles,
            stage=LONG_STAGE,
            cohort=cohort,
            role=role,
            feature_columns=columns,
            ready_blocks=readiness,
            targets=LONG_TARGETS,
        )

    candidate = make(
        "local_activity_plus_eight_hour_activity",
        local_variant="current",
        eight_variant="current",
    )
    controls = (
        ("level_only", make("level_only", local_variant=None, eight_variant=None)),
        (
            "local_activity_only",
            make("local_activity_only", local_variant="current", eight_variant=None),
        ),
        (
            "eight_hour_activity_only",
            make("eight_hour_activity_only", local_variant=None, eight_variant="current"),
        ),
        (
            "causal_72h_old_local_activity",
            make(
                "stale_local_plus_current_eight_hour",
                local_variant="stale",
                eight_variant="current",
            ),
        ),
        (
            "causal_72h_old_eight_hour_activity",
            make(
                "current_local_plus_stale_eight_hour",
                local_variant="current",
                eight_variant="stale",
            ),
        ),
        (
            "within_period_shuffled_both_activity_blocks",
            make(
                "shuffled_local_plus_shuffled_eight_hour",
                local_variant="shuffled",
                eight_variant="shuffled",
            ),
        ),
    )
    add_comparisons(
        comparisons,
        stage=LONG_STAGE,
        cohort=cohort,
        route_id="four_hour_level_local_plus_eight_hour_activity",
        plain_question=(
            "At contacts with 4h source levels, do current local activity and completed "
            "8h activity together improve 12h, 24h, or 48h reaction paths beyond either "
            "activity scale alone and timestamp controls?"
        ),
        candidate=candidate,
        controls=controls,
        targets=LONG_TARGETS,
    )


def build_registry() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    profiles: dict[str, dict[str, Any]] = {}
    comparisons: list[dict[str, Any]] = []
    for cohort in COHORTS:
        build_mtf_registry(profiles, comparisons, cohort)
        build_long_registry(profiles, comparisons, cohort)
    return profiles, comparisons


def validate_sources() -> dict[str, Any]:
    if not G13_JOINT_REVIEW.is_file():
        raise FileNotFoundError(G13_JOINT_REVIEW)
    review = json.loads(G13_JOINT_REVIEW.read_text(encoding="utf-8"))
    if review.get("status") != "completed_generation13_joint_review":
        raise ValueError("Generation 13 joint review is not terminal.")
    if not review.get("all_frozen_siblings_completed_before_review"):
        raise ValueError("Generation 13 siblings were not jointly completed.")
    sources = {"generation13_joint_review": artifact(G13_JOINT_REVIEW)}
    for cohort in COHORTS:
        path = g13c.RECORD_ROOT / f"{cohort}_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed_generation13_cache":
            raise ValueError(f"Generation 13 cache is incomplete for {cohort}.")
        sources[f"generation13_{cohort}_cache"] = artifact(path)
    return sources


def freeze_generation14(*, overwrite: bool = False) -> dict[str, Any]:
    if FREEZE_PATH.is_file() and not overwrite:
        existing = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        if existing.get("status") != "frozen_before_generation14_outcomes":
            raise ValueError("Existing Generation 14 freeze has an invalid status.")
        return existing
    profiles, comparisons = build_registry()
    frozen = {
        "schema_version": 1,
        "generation": 14,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation14_outcomes",
        "plain_objective": (
            "Test a complete broad sibling set before following one Generation 13 lead: "
            "reaction-label robustness, 8h activity attribution, level and cluster "
            "interaction, wider-market and oscillator conditioning, restrained interaction "
            "between the 8h and 48h leads, external-source readiness, and coin/time stability."
        ),
        "siblings": {
            "reaction_definition_sensitivity": {
                "price_thresholds_atr": list(REACTION_PRICE_THRESHOLDS_ATR),
                "zone_width_floor_retained": True,
                "volume_thresholds": list(REACTION_VOLUME_THRESHOLDS),
                "horizons_hours": list(REACTION_HORIZONS),
                "controls": list(g13z.DIRECT_CONTROLS),
                "plain_question": (
                    "Does the genuine-level reaction difference survive reasonable nearby "
                    "definitions instead of depending on exactly 0.5 ATR and 1.25x volume?"
                ),
            },
            MTF_STAGE: {
                "routes": sorted(
                    {
                        item["route_id"]
                        for item in comparisons
                        if item["stage"] == MTF_STAGE
                    }
                ),
                "targets": list(MTF_TARGETS),
            },
            LONG_STAGE: {
                "source_timeframes": ["4h"],
                "horizons_hours": [12, 24, 48],
                "targets": list(LONG_TARGETS),
            },
            "coin_and_time_stability": {
                "normal_groups_defined_before_outcomes": {
                    key: list(value) for key, value in NORMAL_GROUPS.items()
                },
                "meme_cohort": "frozen_top_ten_traded_meme_cohort",
                "checks": [
                    "per_coin_and_per_period",
                    "leave_one_coin_out",
                    "normal_predeclared_groups",
                    "standard_normal_vs_later_normal_vs_meme",
                ],
            },
            "external_source_readiness_refresh": {
                "sources": ["historical_news", "historical_orderbook"],
                "minimum_model_gate": (
                    "At least 50 eligible rows and five coins in every declared period; "
                    "otherwise record coverage and park without inventing neutral values."
                ),
            },
        },
        "profiles": profiles,
        "profile_count": len(profiles),
        "comparisons": comparisons,
        "comparison_count": len(comparisons),
        "decision_rule": {
            "freqai_point": (
                "Candidate reduces equal-coin absolute error against every declared "
                "component, stale, and shuffled control in both periods, with at least "
                "five positive coins on identical prediction rows."
            ),
            "freqai_strict": (
                "The complete point ladder also has weekly-block bootstrap lower bounds "
                "above zero and is not dominated by one coin."
            ),
            "label_robust_point": (
                "For each cohort/window/horizon, both matched controls remain positive in "
                "every period for the original definition and at least seven of nine "
                "predeclared adjacent definitions."
            ),
            "label_robust_strict": (
                "The robust point rule holds and at least five of nine definitions retain "
                "positive weekly-block uncertainty lower bounds throughout the ladder."
            ),
            "combination": (
                "A combination receives credit only when it beats every named component "
                "alone and all timestamp controls; more columns are not presumed better."
            ),
        },
        "sequencing": {
            "complete_sibling_set_frozen_before_any_generation14_outcome": True,
            "all_siblings_terminal_or_honestly_parked_before_joint_review": True,
            "no_generation15_descendant_before_joint_review": True,
        },
        "evaluation_windows": {
            STANDARD: {
                "normal": list(g13z.g11base.VALIDATION_PERIODS["normal"]),
                "meme": list(g13z.g11base.VALIDATION_PERIODS["meme"]),
            },
            RECENT: {"normal": list(g12z.RECENT_PERIODS), "meme": []},
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
        "source_contracts": validate_sources(),
    }
    FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(frozen, FREEZE_PATH)
    return frozen


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Freeze broad Generation 14 siblings.")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = freeze_generation14(overwrite=args.overwrite)
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
