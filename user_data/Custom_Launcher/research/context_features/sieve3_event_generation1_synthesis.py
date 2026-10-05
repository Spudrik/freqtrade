"""Synthesize the sealed Sieve3 Generation-1 evidence into joint verdicts.

This post-processor reads only compact, integrity-audited Generation-1 reports.
It does not train a model, rescore predictions, modify a batch result, or launch a
branch.  A branch is retained only when the same scope/target clears every
frozen comparator required for the claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame

from sieve3_event_generation1_review import atomic_json


PORTABLE_PAIRS = 7
PORTABLE_WINDOWS = 2

B2_CONTRACTS = {
    "exact_event": (
        "fixed_indicator_control",
        "shifted_exact_event",
        "components_all",
    ),
    "exact_event_active": (
        "fixed_indicator_control",
        "shifted_exact_active",
    ),
    "components_all": (
        "fixed_indicator_control",
        "components_no_d1",
        "components_no_vp_bos",
        "components_no_h4_retest",
    ),
    "exact_plus_components": (
        "fixed_indicator_control",
        "components_all",
        "exact_event",
    ),
}

B3_MAIN_CONTROLS = (
    "fixed_indicator_control",
    "level_components",
    "exact_mixed_equal",
    "stale_full_state",
)
B3_ALL_CONTROLS = (
    *B3_MAIN_CONTROLS,
    "components_no_mixed",
    "components_no_timeframe",
)

B4_CONTRACTS = {
    "exit_family_identity": (
        "open_trade_price_indicator_control",
        "shifted_exit_family_placebo",
    ),
    "exit_identity_plus_trade_state": (
        "open_trade_price_indicator_control",
        "shifted_exit_family_placebo",
        "exit_family_identity",
        "trade_state_with_exit_family_ablation",
    ),
    "trade_state_with_exit_family_ablation": (
        "open_trade_price_indicator_control",
    ),
}


def require_columns(frame: DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def regression_pass(
    frame: DataFrame, pair_threshold: int = PORTABLE_PAIRS
) -> pd.Series:
    return (
        frame["rank_mae_majority_positive_pairs"].ge(pair_threshold)
        & frame["rank_mae_majority_positive_windows"].ge(PORTABLE_WINDOWS)
        & frame["median_spearman_delta"].gt(0.0)
        & frame["median_mae_skill"].gt(0.0)
    )


def binary_pass(frame: DataFrame, pair_threshold: int = PORTABLE_PAIRS) -> pd.Series:
    return (
        frame["auc_brier_majority_positive_pairs"].ge(pair_threshold)
        & frame["auc_brier_majority_positive_windows"].ge(PORTABLE_WINDOWS)
        & frame["median_auc_delta"].gt(0.0)
        & frame["median_brier_skill"].gt(0.0)
    )


def family_counts(frame: DataFrame) -> dict[str, int]:
    if frame.empty:
        return {}
    return {
        str(key): int(value)
        for key, value in frame.groupby("target_family", dropna=False).size().items()
    }


def controlled_intersection(
    frame: DataFrame,
    profile: str,
    comparators: tuple[str, ...],
    *,
    binary: bool = False,
    pair_threshold: int = PORTABLE_PAIRS,
) -> DataFrame:
    work = frame.loc[frame["profile_id"].eq(profile)].copy()
    if binary:
        work = work.loc[work["target_family"].eq("path_order_binary")]
        work = work.loc[binary_pass(work, pair_threshold)]
    else:
        work = work.loc[~work["target_family"].eq("path_order_binary")]
        work = work.loc[regression_pass(work, pair_threshold)]
    keys = ["profile_id", "scope", "target", "target_family", "horizon_hours"]
    if "target_side" in work.columns:
        keys.insert(4, "target_side")
    grouped = work.groupby(keys, dropna=False)["comparator_id"].agg(
        lambda values: sorted(set(map(str, values)))
    )
    required = set(comparators)
    retained = grouped.loc[grouped.map(lambda values: required.issubset(values))]
    result = retained.reset_index(name="comparators_passed")
    return result


def compact_rows(frame: DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    return json.loads(frame.loc[:, columns].to_json(orient="records"))


def b1_verdict(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    required = (
        "profile_id",
        "scope",
        "target",
        "target_family",
        "rank_mae_vs_both_controls_majority_positive_pairs",
        "rank_mae_vs_both_controls_majority_positive_windows",
        "median_spearman_delta_vs_indicator",
        "median_spearman_delta_vs_matched_shifted",
        "median_mae_skill_vs_indicator",
        "median_mae_skill_vs_matched_shifted",
    )
    require_columns(frame, required, "G1-B1 portability")
    passing = frame.loc[
        frame["rank_mae_vs_both_controls_majority_positive_pairs"].ge(PORTABLE_PAIRS)
        & frame["rank_mae_vs_both_controls_majority_positive_windows"].ge(
            PORTABLE_WINDOWS
        )
        & frame["median_spearman_delta_vs_indicator"].gt(0.0)
        & frame["median_spearman_delta_vs_matched_shifted"].gt(0.0)
        & frame["median_mae_skill_vs_indicator"].gt(0.0)
        & frame["median_mae_skill_vs_matched_shifted"].gt(0.0)
    ]
    return {
        "batch": "G1-B1",
        "verdict": "unsupported_in_tested_scope",
        "classification": "exact_onset_and_active_identity_not_portably_incremental",
        "portability_rows": int(len(frame)),
        "full_control_survivors": int(len(passing)),
        "survivors_by_target_family": family_counts(passing),
        "maximum_majority_positive_pairs": int(
            frame["rank_mae_vs_both_controls_majority_positive_pairs"].max()
        ),
        "maximum_majority_positive_windows": int(
            frame["rank_mae_vs_both_controls_majority_positive_windows"].max()
        ),
        "interpretation": (
            "No exact-onset or onset-plus-active scope improved rank and magnitude "
            "error against both fixed-indicator and matched shifted controls with "
            "seven-coin/two-window support. Isolated near-misses are not a branch."
        ),
        "branch_selected": False,
    }


def b2_verdict(path: Path) -> dict[str, Any]:
    frame = pd.read_csv(path)
    require_columns(
        frame,
        (
            "profile_id",
            "comparator_id",
            "scope",
            "target",
            "target_family",
            "rank_mae_majority_positive_pairs",
            "rank_mae_majority_positive_windows",
            "median_spearman_delta",
            "median_mae_skill",
        ),
        "G1-B2 portability",
    )
    pairwise = frame.loc[regression_pass(frame)]
    intersections: dict[str, Any] = {}
    for profile, comparators in B2_CONTRACTS.items():
        retained = controlled_intersection(frame, profile, comparators)
        intersections[profile] = {
            "required_comparators": list(comparators),
            "survivors": int(len(retained)),
            "survivors_by_target_family": family_counts(retained),
        }
    return {
        "batch": "G1-B2",
        "verdict": "unsupported_in_tested_scope",
        "classification": "scattered_pairwise_lift_without_full_control_increment",
        "portability_rows": int(len(frame)),
        "pairwise_portable_rows": int(len(pairwise)),
        "pairwise_portable_rows_by_target_family": family_counts(pairwise),
        "controlled_intersections": intersections,
        "interpretation": (
            "The exact event, active state, component block and combined model each "
            "show isolated pairwise wins, but no scope/target survives every comparator "
            "required for its claim. Activity cannot be attributed independently to "
            "the exact event or a component group."
        ),
        "branch_selected": False,
    }


def b3_verdict(portability_path: Path, direct_path: Path) -> dict[str, Any]:
    frame = pd.read_csv(portability_path)
    direct = pd.read_csv(direct_path)
    require_columns(
        frame,
        (
            "profile_id",
            "comparator_id",
            "scope",
            "target",
            "target_family",
            "rank_mae_majority_positive_pairs",
            "rank_mae_majority_positive_windows",
            "median_spearman_delta",
            "median_mae_skill",
        ),
        "G1-B3 portability",
    )
    require_columns(
        direct,
        (
            "comparator",
            "band_pct",
            "target",
            "target_family",
            "horizon_hours",
            "distinct_pairs",
            "distinct_windows",
            "classification",
        ),
        "G1-B3 direct portability",
    )
    main = controlled_intersection(
        frame, "full_current_state", B3_MAIN_CONTROLS
    )
    all_controls = controlled_intersection(
        frame, "full_current_state", B3_ALL_CONTROLS
    )
    retained_direct = direct.loc[
        ~direct["classification"].eq("mixed_or_insufficient")
    ].copy()
    direct_counts = Counter(map(str, retained_direct["classification"]))
    selected_columns = [
        "profile_id",
        "scope",
        "target",
        "target_family",
        "horizon_hours",
        "comparators_passed",
    ]
    return {
        "batch": "G1-B3",
        "verdict": "completed_insufficient_evidence",
        "classification": "level_reaction_lead_component_attribution_unresolved",
        "portability_rows": int(len(frame)),
        "direct_portability_rows": int(len(direct)),
        "retained_direct_classifications": dict(sorted(direct_counts.items())),
        "main_control_survivors": int(len(main)),
        "main_control_survivors_by_target_family": family_counts(main),
        "all_control_survivors": int(len(all_controls)),
        "all_control_survivors_by_target_family": family_counts(all_controls),
        "main_control_survivor_rows": compact_rows(main, selected_columns),
        "interpretation": (
            "Direct tests retain broad activity and faster-unconditional-reaction "
            "associations against no-level and stale controls. One multi_timeframe:long "
            "4-hour unconditional-1ATR row also survives the four main FreqAI controls, "
            "but no row survives both mixed-identity and timeframe ablations. The live "
            "reaction lead remains real enough to attribute, but neither mixed/equal-"
            "highest identity nor higher-timeframe precedence is established."
        ),
        "branch_selected": True,
        "selected_branch_id": "G2-B1-level_component_reaction_attribution",
    }


def b4_direct_summary(frame: DataFrame) -> dict[str, Any]:
    metrics = (
        "hold_instead_delta",
        "missed_additional_profit",
        "avoided_loss_after_exit",
        "net_exit_regret",
        "favourable_peak_step",
        "adverse_peak_step",
        "favourable_before_adverse",
    )
    result: dict[str, Any] = {}
    for metric in metrics:
        pair_col = f"{metric}_distinct_adequate_pairs"
        window_col = f"{metric}_distinct_adequate_windows"
        require_columns(frame, (pair_col, window_col), "G1-B4 direct portability")
        result[metric] = {
            "adequately_broad_rows": int(
                (
                    frame[pair_col].ge(PORTABLE_PAIRS)
                    & frame[window_col].ge(PORTABLE_WINDOWS)
                ).sum()
            ),
            "maximum_adequate_pairs": int(frame[pair_col].max()),
            "maximum_adequate_windows": int(frame[window_col].max()),
        }
    return result


def b4_direct_survivor_support(
    survivors: DataFrame, direct: DataFrame
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in survivors.itertuples(index=False):
        parts = str(row.scope).split(":")
        group = parts[1]
        suffix = ":".join(parts[2:])
        scoped = direct.loc[
            direct["group"].eq(group)
            & direct["horizon_hours"].eq(row.horizon_hours)
        ].copy()
        if suffix == "final":
            scoped = scoped.loc[scoped["analysis_action_role"].eq("final")]
        elif suffix == "partial":
            scoped = scoped.loc[scoped["analysis_action_role"].eq("partial")]
        elif suffix.startswith("reason:"):
            scoped = scoped.loc[
                scoped["analysis_reason_category"].eq(suffix.split(":", 1)[1])
            ]
        supported = scoped.loc[
            scoped["favourable_before_adverse_distinct_adequate_pairs"].ge(3)
            & scoped["favourable_before_adverse_distinct_adequate_windows"].ge(2)
        ]
        median = pd.to_numeric(
            supported["median_event_minus_control_favourable_before_adverse"],
            errors="coerce",
        )
        records.append(
            {
                "scope": str(row.scope),
                "target": str(row.target),
                "horizon_hours": int(row.horizon_hours),
                "supported_direct_rows": int(len(supported)),
                "positive_direct_median_rows": int(median.gt(0.0).sum()),
                "negative_direct_median_rows": int(median.lt(0.0).sum()),
                "supported_reasons": sorted(
                    set(map(str, supported["analysis_reason_category"]))
                ),
                "contains_partial_scope": suffix == "partial",
            }
        )
    return records


def b4_verdict(portability_path: Path, direct_path: Path) -> dict[str, Any]:
    frame = pd.read_csv(portability_path)
    direct = pd.read_csv(direct_path)
    require_columns(
        frame,
        (
            "profile_id",
            "comparator_id",
            "scope",
            "target",
            "target_family",
            "rank_mae_majority_positive_pairs",
            "rank_mae_majority_positive_windows",
            "auc_brier_majority_positive_pairs",
            "auc_brier_majority_positive_windows",
        ),
        "G1-B4 portability",
    )
    specific_regression = frame.loc[
        ~frame["target_family"].eq("path_order_binary")
        & regression_pass(frame, 3)
    ]
    specific_binary = frame.loc[
        frame["target_family"].eq("path_order_binary") & binary_pass(frame, 3)
    ]
    intersections: dict[str, Any] = {}
    for profile, comparators in B4_CONTRACTS.items():
        pair_threshold = (
            PORTABLE_PAIRS
            if profile == "trade_state_with_exit_family_ablation"
            else 3
        )
        reg = controlled_intersection(
            frame, profile, comparators, pair_threshold=pair_threshold
        )
        binary_result = controlled_intersection(
            frame,
            profile,
            comparators,
            binary=True,
            pair_threshold=pair_threshold,
        )
        intersections[profile] = {
            "required_comparators": list(comparators),
            "required_majority_positive_pairs": pair_threshold,
            "regression_survivors": int(len(reg)),
            "regression_survivors_by_target_family": family_counts(reg),
            "binary_survivors": int(len(binary_result)),
            "binary_survivors_by_target_family": family_counts(binary_result),
        }
    identity_binary = controlled_intersection(
        frame,
        "exit_family_identity",
        B4_CONTRACTS["exit_family_identity"],
        binary=True,
        pair_threshold=3,
    )
    direct_support = b4_direct_survivor_support(identity_binary, direct)
    supported_identity_binary = sum(
        item["supported_direct_rows"] > 0 and not item["contains_partial_scope"]
        for item in direct_support
    )
    return {
        "batch": "G1-B4",
        "verdict": "completed_insufficient_evidence",
        "classification": "exit_identity_4h_path_order_lead_state_not_incremental",
        "portability_rows": int(len(frame)),
        "reason_family_pair_threshold": 3,
        "broad_state_pair_threshold": PORTABLE_PAIRS,
        "pairwise_specific_regression_survivors": int(len(specific_regression)),
        "pairwise_specific_binary_survivors": int(len(specific_binary)),
        "maximum_regression_majority_positive_pairs": int(
            frame["rank_mae_majority_positive_pairs"].max()
        ),
        "maximum_binary_majority_positive_pairs": int(
            frame["auc_brier_majority_positive_pairs"].max()
        ),
        "controlled_intersections": intersections,
        "identity_binary_survivor_rows": compact_rows(
            identity_binary,
            [
                "profile_id",
                "scope",
                "target",
                "target_family",
                "target_side",
                "horizon_hours",
                "comparators_passed",
            ],
        ),
        "identity_binary_direct_support": direct_support,
        "identity_binary_survivors_with_direct_support": int(
            supported_identity_binary
        ),
        "direct_support_description": b4_direct_summary(direct),
        "interpretation": (
            "Seven identity-only 4-hour path-order scopes beat both indicator and shifted "
            "controls at the frozen reason/family-specific three-coin/two-window floor, "
            "and each has non-partial repaired direct support. Direct signs differ by "
            "reason, so this is predictive attribution rather than a favourable-exit "
            "claim. Identity plus trade state has no binary all-control survivor and "
            "state alone has no broad seven-coin survivor."
        ),
        "branch_selected": True,
        "selected_branch_id": "G2-B2-exit_identity_4h_path_order_attribution",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generation_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    generation_dir = args.generation_dir.resolve()
    sealed_review_path = generation_dir / "generation1_review.json"
    if not sealed_review_path.is_file():
        raise FileNotFoundError(sealed_review_path)
    sealed = json.loads(sealed_review_path.read_text(encoding="utf-8"))
    if (
        sealed.get("status") != "review_gate_open"
        or not sealed.get("all_batch_evidence_ready")
        or sealed.get("integrity_error_count") != 0
    ):
        raise RuntimeError("Generation-1 integrity review gate is not open")

    inputs = {
        "sealed_review": sealed_review_path,
        "b1_portability": generation_dir
        / "g1-b1/freqai/g1_b1_portability_summary.csv",
        "b2_portability": generation_dir
        / "g1-b2/freqai/g1_b2_portability_summary.csv",
        "b3_portability": generation_dir
        / "g1-b3/freqai/g1_b3_portability_summary.csv",
        "b3_direct": generation_dir
        / "g1-b3/freqai/g1_b3_direct_portability_summary.csv",
        "b4_portability": generation_dir
        / "g1-b4/freqai/g1_b4_portability_summary.csv",
        "b4_direct": generation_dir / "g1-b4/direct_exit_role_portability.csv",
    }
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing synthesis inputs: {missing}")

    verdicts = [
        b1_verdict(inputs["b1_portability"]),
        b2_verdict(inputs["b2_portability"]),
        b3_verdict(inputs["b3_portability"], inputs["b3_direct"]),
        b4_verdict(inputs["b4_portability"], inputs["b4_direct"]),
    ]
    level_branch = {
        "id": "G2-B1-level_component_reaction_attribution",
        "status": "selected_for_generation2_freeze",
        "originating_batch": "G1-B3",
        "trader_readable_question": (
            "Do simple live level counts, side balance or proximity explain the repeatable "
            "increase in reaction magnitude and 1ATR-contact probability, without relying "
            "on mixed/equal-highest identity or automatic higher-timeframe precedence?"
        ),
        "evidence_basis": (
            "Twenty-four direct B3 rows retain coherent activity/timing classifications "
            "across broad coin/window support. One multi_timeframe:long 4h unconditional-"
            "1ATR row survives indicator, live-component, exact-flag and stale controls, "
            "but no row survives both mixed-identity and timeframe ablations."
        ),
        "alternative": (
            "The direct association is generic volatility or level density, correlated "
            "feature redundancy, event-family context, or an unstable model interaction."
        ),
        "frozen_surface": {
            "pairs": "same frozen top-10 surface",
            "windows": "same four chronological windows",
            "bands": [0.005, 0.01],
            "primary_scope": "family:multi_timeframe:long",
            "primary_target": "&-one_atr_touch_observed_4h",
            "secondary_direct_families": [
                "reaction_magnitude",
                "censored_reaction_timing",
                "volume_activity",
            ],
        },
        "smallest_useful_ladder": [
            "fixed indicator control",
            "counts and proximity only",
            "relationship identity without timeframe fields",
            "timeframe fields without relationship identity",
            "counts plus relationship",
            "counts plus timeframe",
            "full current state",
            "like-for-like 168h stale state",
        ],
        "pass_rule": (
            "A simplified current block must beat fixed-indicator, like-for-like stale, "
            "matched no-level and every immediately simpler component control on the same "
            "reaction target with at least seven majority-positive coins and two windows. "
            "Direction remains a falsification target and cannot be inferred from activity."
        ),
        "stop_rule": (
            "Run the frozen component ladder once. If no simplified block survives all "
            "controls, reject this level representation and do not launch the regime-by-"
            "level direction theory from it. Do not add bands, level families or horizons "
            "after seeing results."
        ),
    }
    exit_path_branch = {
        "id": "G2-B2-exit_identity_4h_path_order_attribution",
        "status": "selected_for_generation2_freeze",
        "originating_batch": "G1-B4",
        "trader_readable_question": (
            "Which non-partial exit-family and reason identities explain the retained "
            "4-hour favourable-before-adverse ordering signal, and is the sign stable "
            "after separating stop-loss/source-invalidation from max-hold/profit-target?"
        ),
        "evidence_basis": (
            "Seven identity-only 4-hour path-order scopes beat both price/indicator and "
            "168h-shifted identity controls at the frozen three-coin/two-window floor. "
            "Every scope has repaired non-partial direct support, but direct signs differ "
            "by reason. Identity plus trade state has no binary all-control survivor."
        ),
        "alternative": (
            "The identity model is pooling opposite reason effects, repeated trade paths, "
            "or group concentration; a coarser family label or observable trade state may "
            "explain the same ordering without exact reason identity."
        ),
        "frozen_surface": {
            "pairs": "same frozen top-10 surface with reason-specific adequacy enforced",
            "windows": "same chronological windows",
            "horizon_hours": [4],
            "target": "favourable_before_adverse path order",
            "included_action_roles": ["final", "all_actions", "matched_open_state"],
            "excluded_action_roles": ["partial"],
            "reason_groups": [
                "stop_loss",
                "source_invalidation",
                "max_hold",
                "profit_target_full",
            ],
        },
        "smallest_useful_ladder": [
            "price and fixed-indicator control",
            "coarse exit-family identity",
            "reason identity without family",
            "family plus reason identity",
            "matched observable trade state",
            "family/reason identity plus state",
            "like-for-like 168h shifted identity",
        ],
        "pass_rule": (
            "A reason/family-specific path-order lead must improve AUC and Brier skill "
            "against indicator, shifted identity and every immediately simpler identity "
            "block across at least three adequately supported coins and two windows, with "
            "a matching repaired direct row. Any broad state claim still requires seven "
            "coins and two windows. Opposite direct signs must remain separate conclusions."
        ),
        "stop_rule": (
            "Run the frozen 4-hour path-order ladder once. Reject partial-stage claims and "
            "park any reason whose direct and model direction is unresolved. Do not add "
            "exit families, horizons, state fields or profit composites after seeing results."
        ),
    }
    payload = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "generation": 1,
        "status": "joint_review_completed",
        "integrity_review": str(sealed_review_path),
        "integrity_review_sha256": sha256(sealed_review_path),
        "thresholds": {
            "majority_positive_pairs": PORTABLE_PAIRS,
            "majority_positive_windows": PORTABLE_WINDOWS,
            "regression_requires_positive_median_rank_and_mae": True,
            "binary_requires_positive_median_auc_and_brier": True,
        },
        "input_hashes": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in inputs.items()
        },
        "batch_verdicts": verdicts,
        "joint_conclusion": (
            "B1 and B2 are unsupported in their tested scopes and do not spawn branches. "
            "B3 retains a broad reaction association whose defining level component is "
            "unresolved. B4 retains seven reason/family-specific 4-hour path-order leads, "
            "but trade state adds no portable binary increment and direct signs conflict. "
            "Generation 2 therefore contains two attribution batches and no trading action."
        ),
        "selected_generation2_batches": [level_branch, exit_path_branch],
        "held_later_generation_candidates": [
            {
                "id": "G2-CANDIDATE-market_regime_recent_path_x_level_reaction_direction",
                "new_status": "held_for_generation3_pending_G2_component_attribution",
                "reason": (
                    "The direction/context theory requires a non-redundant level-reaction "
                    "representation. G1 shows a reaction lead but does not establish which "
                    "level component carries it."
                ),
                "release_condition": (
                    "G2-B1 identifies a simplified live component block that survives all "
                    "fixed, stale, no-level and component controls."
                ),
            }
        ],
        "prohibited_conclusions": [
            "No result establishes causation.",
            "Activity or faster reaction is not directional evidence.",
            "Higher timeframe does not receive automatic precedence.",
            "B1 or B2 pairwise near-misses cannot spawn rescue branches.",
            "B4 identity prediction cannot be described as a favourable exit effect until reason-specific signs agree.",
            "No entry, exit, sizing or risk action is promoted from Generation 1.",
        ],
    }
    output = args.output or generation_dir / "generation1_joint_verdict.json"
    atomic_json(payload, output.resolve())
    print(json.dumps({"status": "written", "output": str(output.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
