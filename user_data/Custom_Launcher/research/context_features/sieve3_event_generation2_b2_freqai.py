"""Run frozen G2-B2 exit identity/reason 4h path-order attribution."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame
from sieve3_event_freqai_round1 import (
    REPO_ROOT,
    USER_DATA_DIR,
    append_log,
    atomic_json,
    load_predictions,
    now_iso,
    parse_pairs,
    score_values,
)
from sieve3_event_generation1_b1_freqai import (
    DEFAULT_BASE_CONFIG,
    INCLUDE_TIMEFRAMES,
    profile_config,
    run_manifest,
)
from sieve3_event_generation1_b4_cache import (
    DEFAULT_PAIRS,
    ELIGIBLE_GROUPS,
    REASON_CATEGORIES,
    group_token,
)
from sieve3_event_generation1_b4_freqai import (
    TARGETS,
    actuals_for_pair,
    scope_dates,
    scope_side,
    target_metadata,
)
from sieve3_event_generation2_common import (
    add_comparisons,
    cache_contract_audit,
    completed_output_audit,
    controlled_portability,
    generation_batch,
    read_json,
    reuse_profiles,
    review_batch,
    sha256_file,
    sibling_preflights_pass,
)


GENERATION_DIR = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "sieve3_event_reaction"
    / "generation2"
)
DEFAULT_OUTPUT = GENERATION_DIR / "g2-b2" / "freqai"
GENERATION_MANIFEST = GENERATION_DIR / "generation_manifest.json"
G1_REVIEW = GENERATION_DIR.parent / "generation1" / "generation1_review.json"
G1_JOINT_VERDICT = GENERATION_DIR.parent / "generation1" / "generation1_joint_verdict.json"
SOURCE_OUTPUT = GENERATION_DIR.parent / "generation1" / "g1-b4" / "freqai"
SOURCE_MANIFEST = SOURCE_OUTPUT / "manifest.json"
DEFAULT_CACHE = SOURCE_OUTPUT.parent / "cache"
DEFAULT_SCOPES = SOURCE_OUTPUT.parent / "event_scopes.parquet"
DEFAULT_DIRECT = SOURCE_OUTPUT.parent / "direct_exit_role_portability.csv"
DEFAULT_PYTHON = (
    REPO_ROOT / "runtime" / "venvs" / "freqtrade-backtest-05" / "Scripts" / "python.exe"
)


GENERIC_ROLE_COLUMNS = [
    *[f"exit_event_any__{side}" for side in ("long", "short")],
    *[f"exit_event_partial__{side}" for side in ("long", "short")],
    *[f"exit_event_final__{side}" for side in ("long", "short")],
    "exit_event_action_count",
]
FAMILY_COLUMNS = [f"exit_event__{group_token(family, side)}" for family, side in ELIGIBLE_GROUPS]
REASON_COLUMNS = [f"exit_reason__{category}" for category in REASON_CATEGORIES]
FULL_IDENTITY_COLUMNS = [
    *FAMILY_COLUMNS,
    *GENERIC_ROLE_COLUMNS[:-1],
    *REASON_COLUMNS,
    "exit_event_action_count",
]


def _profile(columns: list[str], shift_hours: int, theory: str) -> dict[str, Any]:
    return {
        "strategy": "Sieve3EventReactionFixedIndicatorIdentityFreqAIResearchStrategy",
        "shift_hours": shift_hours,
        "feature_columns": columns,
        "theory": theory,
    }


PROFILES: dict[str, dict[str, Any]] = {
    "fixed_indicator_control": {
        "strategy": "Sieve3EventReactionFixedIndicatorControlFreqAIResearchStrategy",
        "shift_hours": 0,
        "feature_columns": [],
        "theory": "Price, structure and fixed indicators without exit identity or trade state.",
    },
    "generic_event_role_current": _profile(
        GENERIC_ROLE_COLUMNS,
        0,
        "Current generic any/partial/final event role and action count without family or reason.",
    ),
    "generic_event_role_stale": _profile(
        GENERIC_ROLE_COLUMNS,
        168,
        "Like-for-like generic event role delayed 168 hours.",
    ),
    "family_only_current": _profile(
        [*FAMILY_COLUMNS, *GENERIC_ROLE_COLUMNS],
        0,
        "Current exit family/side identity plus generic event role without reason.",
    ),
    "family_only_stale": _profile(
        [*FAMILY_COLUMNS, *GENERIC_ROLE_COLUMNS],
        168,
        "Like-for-like family/side and generic role delayed 168 hours.",
    ),
    "reason_only_current": _profile(
        [*REASON_COLUMNS, *GENERIC_ROLE_COLUMNS],
        0,
        "Current exit reason identity plus generic event role without family.",
    ),
    "reason_only_stale": _profile(
        [*REASON_COLUMNS, *GENERIC_ROLE_COLUMNS],
        168,
        "Like-for-like reason identity and generic role delayed 168 hours.",
    ),
    "full_identity_current": _profile(
        FULL_IDENTITY_COLUMNS, 0, "Current family, role and exact reason identity."
    ),
    "full_identity_stale": _profile(
        FULL_IDENTITY_COLUMNS, 168, "Full family/role/reason identity delayed 168 hours."
    ),
    "trade_state_only": {
        "strategy": "Sieve3EventReactionFixedIndicatorIdentityFreqAIResearchStrategy",
        "shift_hours": 0,
        "feature_columns": [],
        "theory": "Observable trade state without action identity.",
    },
    "full_identity_plus_state": {
        "strategy": "Sieve3EventReactionFixedIndicatorIdentityFreqAIResearchStrategy",
        "shift_hours": 0,
        "feature_columns": [],
        "theory": "Current full identity plus observable trade state.",
    },
}

REUSED = {
    "fixed_indicator_control": "open_trade_price_indicator_control",
    "full_identity_current": "exit_family_identity",
    "full_identity_stale": "shifted_exit_family_placebo",
    "trade_state_only": "trade_state_with_exit_family_ablation",
    "full_identity_plus_state": "exit_identity_plus_trade_state",
}
NEW_PROFILE_IDS = tuple(profile for profile in PROFILES if profile not in REUSED)
ALL_PROFILE_IDS = tuple(PROFILES)
COMPARISON_CONTRACT = {
    "generic_event_role_current": ["fixed_indicator_control", "generic_event_role_stale"],
    "family_only_current": [
        "fixed_indicator_control",
        "generic_event_role_current",
        "family_only_stale",
    ],
    "reason_only_current": [
        "fixed_indicator_control",
        "generic_event_role_current",
        "reason_only_stale",
    ],
    "full_identity_current": [
        "fixed_indicator_control",
        "generic_event_role_current",
        "family_only_current",
        "reason_only_current",
        "full_identity_stale",
    ],
    "full_identity_plus_state": [
        "fixed_indicator_control",
        "full_identity_current",
        "trade_state_only",
        "full_identity_stale",
    ],
}
COMPARATOR_PREFIXES = {
    profile: f"g2b2_{index:02d}" for index, profile in enumerate(PROFILES, start=1)
}
RETAINED_SCOPES = (
    "group:profit_ladder_three_stage_no_ratchet__short:all_actions",
    "group:profit_ladder_three_stage_ratchet__long:final",
    "group:profit_ladder_three_stage_ratchet__long:matched_open_state",
    "group:profit_level_full_or_ladder__long:reason:max_hold",
    "group:profit_level_full_or_ladder__short:reason:profit_target_full",
    "group:target_partial_invalidation_remainder__long:all_actions",
    "group:target_partial_invalidation_remainder__long:final",
)
CURRENT_IDENTITY_PROFILES = (
    "generic_event_role_current",
    "family_only_current",
    "reason_only_current",
    "full_identity_current",
    "full_identity_plus_state",
)


def _source_profile_columns(source_profile: str) -> list[str]:
    source = read_json(SOURCE_MANIFEST)
    command = next(item for item in source["commands"] if item["profile_id"] == source_profile)
    config = read_json(Path(command["config_path"]))
    return list(config.get("sieve3_event_reaction", {}).get("event_feature_columns") or [])


# Reused trade-state contracts are authoritative in their sealed G1 configs.
PROFILES["trade_state_only"]["feature_columns"] = _source_profile_columns(
    "trade_state_with_exit_family_ablation"
)
PROFILES["full_identity_plus_state"]["feature_columns"] = _source_profile_columns(
    "exit_identity_plus_trade_state"
)


def _new_command(
    *,
    profile_id: str,
    output_dir: Path,
    run_id: str,
    base: dict[str, Any],
    python_exe: Path,
    pairs: tuple[str, ...],
    timerange: str,
    cache_dir: Path,
) -> dict[str, Any]:
    definition = PROFILES[profile_id]
    identifier = f"{run_id}_{profile_id}"
    profile_dir = output_dir / profile_id
    profile_dir.mkdir(parents=True, exist_ok=True)
    config = profile_config(base, identifier, definition, pairs, cache_dir)
    settings = config["sieve3_event_reaction"]
    settings["target_names"] = TARGETS
    settings["event_feature_columns"] = list(definition["feature_columns"])
    config_path = profile_dir / "config.json"
    atomic_json(config_path, config)
    return {
        "profile_id": profile_id,
        "strategy": definition["strategy"],
        "theory": definition["theory"],
        "identifier": identifier,
        "config_path": str(config_path),
        "output_dir": str(profile_dir),
        "event_feature_columns": list(definition["feature_columns"]),
        "command": [
            str(python_exe),
            "-m",
            "freqtrade",
            "backtesting",
            "--userdir",
            str(USER_DATA_DIR),
            "--config",
            str(config_path),
            "--strategy",
            definition["strategy"],
            "--freqaimodel",
            "LightGBMRegressorMultiTarget",
            "--timerange",
            timerange,
            "--export",
            "signals",
            "--export-directory",
            str(profile_dir),
        ],
        "status": "pending",
        "attempts": 0,
    }


def build_manifest(
    *,
    output_dir: Path,
    python_exe: Path,
    base_config: Path,
    pairs: tuple[str, ...],
    timerange: str,
    cache_dir: Path,
    scopes_path: Path,
    selected_profiles: tuple[str, ...] = ALL_PROFILE_IDS,
    allow_reuse: bool = True,
) -> dict[str, Any]:
    source_manifest = read_json(SOURCE_MANIFEST)
    sealed_batch = review_batch(read_json(G1_REVIEW), "G1-B4")
    run_id = f"sieve3_event_g2_b2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    selected_reuse = {key: value for key, value in REUSED.items() if key in selected_profiles}
    reused_commands: list[dict[str, Any]] = []
    reuse_audits: list[dict[str, Any]] = []
    if selected_reuse:
        if not allow_reuse:
            raise ValueError("Smoke/new-only manifests cannot include reused profiles")
        reused_commands, reuse_audits = reuse_profiles(
            alias_to_source=selected_reuse,
            expected_profiles=PROFILES,
            source_manifest=source_manifest,
            sealed_batch_audit=sealed_batch,
            expected_pairs=pairs,
            expected_targets=TARGETS,
            expected_timerange=timerange,
            expected_cache=cache_dir,
            expected_timeframes=INCLUDE_TIMEFRAMES,
        )
    base = read_json(base_config)
    command_by_profile = {item["profile_id"]: item for item in reused_commands}
    for profile_id in selected_profiles:
        if profile_id in REUSED:
            continue
        command_by_profile[profile_id] = _new_command(
            profile_id=profile_id,
            output_dir=output_dir,
            run_id=run_id,
            base=base,
            python_exe=python_exe,
            pairs=pairs,
            timerange=timerange,
            cache_dir=cache_dir,
        )
    return {
        "schema_version": 1,
        "batch": "G2-B2",
        "run_id": run_id,
        "created_at": now_iso(),
        "pairs": list(pairs),
        "windows": source_manifest["windows"],
        "timerange": timerange,
        "targets": TARGETS,
        "include_timeframes": INCLUDE_TIMEFRAMES,
        "event_cache_dir": str(cache_dir),
        "event_scopes": str(scopes_path),
        "worker_python": str(python_exe),
        "source_manifest": str(SOURCE_MANIFEST),
        "generation_manifest": str(GENERATION_MANIFEST),
        "comparison_contract": COMPARISON_CONTRACT,
        "reused_profiles": selected_reuse,
        "reuse_audits": reuse_audits,
        "commands": [command_by_profile[profile] for profile in selected_profiles],
    }


def _new_config_audit(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    pairs = tuple(manifest["pairs"])
    for item in manifest["commands"]:
        if item["profile_id"] in REUSED:
            continue
        definition = PROFILES[item["profile_id"]]
        config = read_json(Path(item["config_path"]))
        settings = config.get("sieve3_event_reaction", {})
        if list(settings.get("event_feature_columns") or []) != list(definition["feature_columns"]):
            errors.append(f"{item['profile_id']}: feature columns differ")
        if int(settings.get("event_feature_shift_hours", 0) or 0) != int(definition["shift_hours"]):
            errors.append(f"{item['profile_id']}: feature shift differs")
        if list(settings.get("target_names") or []) != TARGETS:
            errors.append(f"{item['profile_id']}: targets differ")
        if tuple(config.get("exchange", {}).get("pair_whitelist", [])) != pairs:
            errors.append(f"{item['profile_id']}: pairs differ")
        if (
            str(Path(settings.get("event_cache_dir", "")).resolve()).casefold()
            != str(Path(manifest["event_cache_dir"]).resolve()).casefold()
        ):
            errors.append(f"{item['profile_id']}: cache path differs")
    return errors


def refresh_reuse_audits(manifest: dict[str, Any], manifest_path: Path) -> None:
    reused_commands, audits = reuse_profiles(
        alias_to_source=REUSED,
        expected_profiles=PROFILES,
        source_manifest=read_json(SOURCE_MANIFEST),
        sealed_batch_audit=review_batch(read_json(G1_REVIEW), "G1-B4"),
        expected_pairs=tuple(manifest["pairs"]),
        expected_targets=TARGETS,
        expected_timerange=str(manifest["timerange"]),
        expected_cache=Path(manifest["event_cache_dir"]),
        expected_timeframes=INCLUDE_TIMEFRAMES,
    )
    replacements = {item["profile_id"]: item for item in reused_commands}
    manifest["commands"] = [
        replacements.get(item["profile_id"], item) for item in manifest["commands"]
    ]
    manifest["reuse_audits"] = audits
    atomic_json(manifest_path, manifest)


def preflight(manifest: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    generation = read_json(GENERATION_MANIFEST)
    frozen = generation_batch(generation, "G2-B2")
    required_columns = [
        column for profile in NEW_PROFILE_IDS for column in PROFILES[profile]["feature_columns"]
    ]
    cache_audit = cache_contract_audit(
        Path(manifest["event_cache_dir"]), tuple(manifest["pairs"]), required_columns
    )
    errors = list(cache_audit["errors"])
    errors.extend(
        f"reuse {item['profile_id']}: {error}"
        for item in manifest["reuse_audits"]
        for error in item["errors"]
    )
    errors.extend(_new_config_audit(manifest))
    if generation.get("status") != "generation2_frozen_preflight_pending":
        errors.append(f"unexpected generation status: {generation.get('status')}")
    if frozen.get("status") != "frozen_pending_preflight":
        errors.append(f"unexpected frozen batch status: {frozen.get('status')}")
    if Path(frozen["planned_runner"]).resolve() != Path(__file__).resolve():
        errors.append("frozen planned runner differs from this file")
    if tuple(item["profile_id"] for item in manifest["commands"]) != ALL_PROFILE_IDS:
        errors.append("full manifest profile order differs from frozen implementation")
    if manifest["comparison_contract"] != frozen["comparison_contract"]:
        errors.append("comparison contract differs from frozen generation manifest")
    if list(frozen["retained_scopes"]) != list(RETAINED_SCOPES):
        errors.append("retained scope list differs from frozen generation manifest")
    if sha256_file(G1_JOINT_VERDICT) != generation["source_generation"]["joint_verdict_sha256"]:
        errors.append("Generation-1 joint verdict hash differs from frozen source hash")
    for path in (Path(manifest["event_scopes"]), DEFAULT_DIRECT, SOURCE_MANIFEST, G1_REVIEW):
        if not path.is_file():
            errors.append(f"required source artifact missing: {path}")
    result = {
        "schema_version": 1,
        "batch": "G2-B2",
        "checked_at": now_iso(),
        "manifest": str(output_dir / "manifest.json"),
        "generation_manifest_sha256": sha256_file(GENERATION_MANIFEST),
        "source_manifest_sha256": sha256_file(SOURCE_MANIFEST),
        "source_direct_sha256": sha256_file(DEFAULT_DIRECT),
        "profile_count": len(manifest["commands"]),
        "reused_profile_count": len(REUSED),
        "new_profile_count": len(NEW_PROFILE_IDS),
        "cache_audit": cache_audit,
        "reuse_audits": manifest["reuse_audits"],
        "errors": errors,
        "passed": not errors,
    }
    atomic_json(output_dir / "preflight.json", result)
    return result


def score_manifest(manifest: dict[str, Any], output_dir: Path) -> DataFrame:
    pairs = tuple(manifest["pairs"])
    windows = {
        name: (pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1]))
        for name, bounds in manifest["windows"].items()
    }
    actuals = {pair: actuals_for_pair(pair) for pair in pairs}
    scopes = {pair: scope_dates(Path(manifest["event_scopes"]), pair) for pair in pairs}
    rows: list[dict[str, Any]] = []
    for item in manifest["commands"]:
        predictions = load_predictions(item["identifier"], pairs)
        for pair in pairs:
            merged = predictions[predictions["pair"].eq(pair)].merge(
                actuals[pair], on="date", how="left", validate="one_to_one"
            )
            merged["decision_time"] = merged["date"] + pd.Timedelta(hours=1)
            if "do_predict" in merged:
                merged = merged[pd.to_numeric(merged["do_predict"], errors="coerce").eq(1.0)]
            for window, (start, end) in windows.items():
                window_frame = merged[
                    merged["decision_time"].ge(start) & merged["decision_time"].lt(end)
                ]
                scope_map = {"all": set(window_frame["date"]), **scopes[pair]}
                for scope, dates in scope_map.items():
                    scoped = window_frame[window_frame["date"].isin(dates)]
                    side = scope_side(scope)
                    eligible_targets = (
                        TARGETS
                        if side is None
                        else [target for target in TARGETS if target.startswith(f"&-exit_{side}_")]
                    )
                    for target in eligible_targets:
                        rows.append(
                            {
                                "profile_id": item["profile_id"],
                                "identifier": item["identifier"],
                                "pair": pair,
                                "window": window,
                                "scope": scope,
                                "target": target,
                                **score_values(scoped, target, f"{target}_actual"),
                            }
                        )
    scores = DataFrame(rows)
    metadata = scores["target"].map(target_metadata)
    scores["target_side"] = metadata.map(lambda item: item[0])
    scores["target_family"] = metadata.map(lambda item: item[1])
    scores["horizon_hours"] = metadata.map(lambda item: item[2])
    scores = add_comparisons(scores, COMPARATOR_PREFIXES)
    scores.to_csv(output_dir / "g2_b2_scores.csv", index=False)
    return scores


def direct_support_for_scopes(output_dir: Path) -> list[dict[str, Any]]:
    direct = pd.read_csv(DEFAULT_DIRECT)
    records: list[dict[str, Any]] = []
    support_rows: list[DataFrame] = []
    for scope in RETAINED_SCOPES:
        parts = scope.split(":")
        group = parts[1]
        suffix = ":".join(parts[2:])
        scoped = direct[direct["group"].eq(group) & direct["horizon_hours"].eq(4)].copy()
        if suffix == "final":
            scoped = scoped[scoped["analysis_action_role"].eq("final")]
        elif suffix.startswith("reason:"):
            scoped = scoped[scoped["analysis_reason_category"].eq(suffix.split(":", 1)[1])]
        elif suffix in {"all_actions", "matched_open_state"}:
            scoped = scoped[~scoped["analysis_action_role"].eq("partial")]
        supported = scoped[
            scoped["favourable_before_adverse_distinct_adequate_pairs"].ge(3)
            & scoped["favourable_before_adverse_distinct_adequate_windows"].ge(2)
        ].copy()
        supported["retained_scope"] = scope
        support_rows.append(supported)
        medians = pd.to_numeric(
            supported["median_event_minus_control_favourable_before_adverse"],
            errors="coerce",
        )
        records.append(
            {
                "scope": scope,
                "supported_direct_rows": len(supported),
                "positive_direct_median_rows": int(medians.gt(0.0).sum()),
                "negative_direct_median_rows": int(medians.lt(0.0).sum()),
                "supported_reasons": sorted(set(map(str, supported["analysis_reason_category"]))),
                "support_passed": bool(len(supported)),
                "direct_sign": (
                    "mixed"
                    if medians.gt(0.0).any() and medians.lt(0.0).any()
                    else "positive"
                    if medians.gt(0.0).any()
                    else "negative"
                    if medians.lt(0.0).any()
                    else "unresolved"
                ),
            }
        )
    combined = pd.concat(support_rows, ignore_index=True) if support_rows else DataFrame()
    combined.to_csv(output_dir / "g2_b2_direct_support.csv", index=False)
    atomic_json(output_dir / "g2_b2_direct_support_summary.json", {"scopes": records})
    return records


def terminal_verdict(
    intersection: DataFrame, direct_support: list[dict[str, Any]]
) -> dict[str, Any]:
    direct_by_scope = {item["scope"]: item for item in direct_support}
    candidates = intersection[
        intersection["profile_id"].isin(CURRENT_IDENTITY_PROFILES)
        & intersection["scope"].isin(RETAINED_SCOPES)
        & intersection["target_family"].eq("path_order_binary")
        & intersection["horizon_hours"].eq(4)
    ].copy()
    candidates["target_matches_scope_side"] = candidates.apply(
        lambda row: row["target_side"] == scope_side(row["scope"]), axis=1
    )
    candidates = candidates[candidates["target_matches_scope_side"]]
    candidates["model_passed"] = (
        candidates["all_controls_auc_brier_majority_positive_pairs"].ge(3)
        & candidates["all_controls_auc_brier_majority_positive_windows"].ge(2)
        & candidates["median_worst_auc_delta"].gt(0.0)
        & candidates["median_worst_brier_skill"].gt(0.0)
    )
    candidates["direct_passed"] = candidates["scope"].map(
        lambda scope: bool(direct_by_scope.get(scope, {}).get("support_passed"))
    )
    candidates["direct_sign"] = candidates["scope"].map(
        lambda scope: str(direct_by_scope.get(scope, {}).get("direct_sign", "unresolved"))
    )
    candidates["passes_frozen_rule"] = candidates["model_passed"] & candidates["direct_passed"]
    survivors = candidates[candidates["passes_frozen_rule"]][
        ["profile_id", "scope", "target", "direct_sign"]
    ].to_dict("records")
    reason_or_family = [
        item
        for item in survivors
        if item["profile_id"]
        in {"family_only_current", "reason_only_current", "full_identity_current"}
    ]
    stable_direct_sign = [
        item for item in reason_or_family if item["direct_sign"] in {"positive", "negative"}
    ]
    released = bool(stable_direct_sign)
    return {
        "schema_version": 1,
        "batch": "G2-B2",
        "finished_at": now_iso(),
        "status": "completed" if released else "unsupported_in_tested_scope",
        "survivors": survivors,
        "reason_or_family_survivors": reason_or_family,
        "stable_direct_sign_survivors": stable_direct_sign,
        "direct_support": direct_support,
        "release_generation3_exit_reason_context_theory": released,
        "interpretation": (
            "At least one non-partial reason/family identity retained 4h path-order "
            "information against every frozen control with matching direct support."
            if released
            else "No non-partial reason/family identity satisfied the complete frozen "
            "AUC/Brier and direct-support rule."
        ),
    }


def run_full(manifest: dict[str, Any], output_dir: Path, score_only: bool) -> int:
    gate_paths = (
        GENERATION_DIR / "g2-b1" / "freqai" / "preflight.json",
        GENERATION_DIR / "g2-b2" / "freqai" / "preflight.json",
        GENERATION_DIR / "g2-b1" / "freqai" / "smoke_audit.json",
        GENERATION_DIR / "g2-b2" / "freqai" / "smoke_audit.json",
    )
    gate_passed, gate_errors = sibling_preflights_pass(gate_paths)
    if not gate_passed:
        raise RuntimeError("Generation-2 joint preflight gate is closed: " + "; ".join(gate_errors))
    log_path = output_dir.parent / "research_log.md"
    if not score_only:
        returncode = run_manifest(manifest, output_dir / "manifest.json", log_path)
        if returncode:
            return returncode
    output_audit = completed_output_audit(manifest)
    atomic_json(output_dir / "output_audit.json", output_audit)
    if not output_audit["passed"]:
        raise RuntimeError("G2-B2 output audit failed: " + "; ".join(output_audit["errors"]))
    scores = score_manifest(manifest, output_dir)
    long, pairwise, intersection = controlled_portability(
        scores=scores,
        comparison_contract=COMPARISON_CONTRACT,
        comparator_prefixes=COMPARATOR_PREFIXES,
        metadata_columns=["target_side", "target_family", "horizon_hours"],
        include_binary=True,
    )
    long.to_csv(output_dir / "g2_b2_controlled_observations.csv", index=False)
    pairwise.to_csv(output_dir / "g2_b2_pairwise_portability.csv", index=False)
    intersection.to_csv(output_dir / "g2_b2_controlled_intersection.csv", index=False)
    direct = direct_support_for_scopes(output_dir)
    verdict = terminal_verdict(intersection, direct)
    atomic_json(output_dir / "terminal_verdict.json", verdict)
    summary = {
        "finished_at": now_iso(),
        "batch": "G2-B2",
        "profiles": len(manifest["commands"]),
        "reused_profiles": len(REUSED),
        "new_profiles": len(NEW_PROFILE_IDS),
        "score_rows": len(scores),
        "pairwise_rows": len(pairwise),
        "intersection_rows": len(intersection),
        "verdict": verdict["status"],
    }
    atomic_json(output_dir / "run_summary.json", summary)
    append_log(
        log_path,
        f"G2-B2 completed with {len(scores)} score rows; verdict `{verdict['status']}`.",
    )
    print(json.dumps(summary, indent=2))
    return 0


def run_smoke(*, output_dir: Path, python_exe: Path, base_config: Path, cache_dir: Path) -> int:
    smoke_dir = output_dir / "smoke"
    smoke_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = smoke_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
    else:
        manifest = build_manifest(
            output_dir=smoke_dir,
            python_exe=python_exe,
            base_config=base_config,
            pairs=("BTC/USDT:USDT",),
            timerange="20260501-20260601",
            cache_dir=cache_dir,
            scopes_path=DEFAULT_SCOPES,
            selected_profiles=NEW_PROFILE_IDS,
            allow_reuse=False,
        )
        atomic_json(manifest_path, manifest)
    returncode = run_manifest(manifest, manifest_path, smoke_dir / "research_log.md")
    if returncode:
        return returncode
    audit = completed_output_audit(manifest)
    audit.update(
        {
            "schema_version": 1,
            "batch": "G2-B2",
            "checked_at": now_iso(),
            "purpose": "BTC one-month execution smoke for every distinct new feature contract.",
        }
    )
    atomic_json(output_dir / "smoke_audit.json", audit)
    print(
        json.dumps(
            {
                "batch": "G2-B2",
                "smoke_passed": audit["passed"],
                "profiles": len(manifest["commands"]),
            },
            indent=2,
        )
    )
    return 0 if audit["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-exe", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--scopes", type=Path, default=DEFAULT_SCOPES)
    parser.add_argument("--pairs", default=DEFAULT_PAIRS)
    parser.add_argument("--timerange", default="20240401-20260629")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    if not args.python_exe.is_file():
        raise FileNotFoundError(args.python_exe)
    for path in (
        args.base_config,
        args.cache_dir,
        args.scopes,
        GENERATION_MANIFEST,
        SOURCE_MANIFEST,
        G1_REVIEW,
    ):
        if not path.exists():
            raise FileNotFoundError(path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        return run_smoke(
            output_dir=output_dir,
            python_exe=args.python_exe.resolve(),
            base_config=args.base_config.resolve(),
            cache_dir=args.cache_dir.resolve(),
        )
    manifest_path = output_dir / "manifest.json"
    pairs = parse_pairs(args.pairs)
    if manifest_path.is_file():
        manifest = read_json(manifest_path)
        if tuple(item["profile_id"] for item in manifest["commands"]) != ALL_PROFILE_IDS:
            raise ValueError("Existing G2-B2 manifest profile surface differs")
    else:
        manifest = build_manifest(
            output_dir=output_dir,
            python_exe=args.python_exe.resolve(),
            base_config=args.base_config.resolve(),
            pairs=pairs,
            timerange=args.timerange,
            cache_dir=args.cache_dir.resolve(),
            scopes_path=args.scopes.resolve(),
        )
        atomic_json(manifest_path, manifest)
    if args.preflight:
        refresh_reuse_audits(manifest, manifest_path)
        result = preflight(manifest, output_dir)
        print(
            json.dumps(
                {"batch": "G2-B2", "passed": result["passed"], "errors": result["errors"]}, indent=2
            )
        )
        return 0 if result["passed"] else 1
    return run_full(manifest, output_dir, score_only=args.score_only)


if __name__ == "__main__":
    raise SystemExit(main())
