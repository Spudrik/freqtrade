"""Independently audit and jointly gate frozen Sieve3 Generation 2 results."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame
from sieve3_event_freqai_round1 import USER_DATA_DIR, atomic_json
from sieve3_event_generation1_b4_freqai import scope_side
from sieve3_event_generation1_review import frame_audit, key_digest, model_output_audit
from sieve3_event_generation2_b1_freqai import (
    ALL_PROFILE_IDS as B1_PROFILES,
)
from sieve3_event_generation2_b1_freqai import (
    PRIMARY_SCOPE,
    PRIMARY_TARGET,
    SIMPLIFIED_CURRENT,
)
from sieve3_event_generation2_b2_freqai import (
    ALL_PROFILE_IDS as B2_PROFILES,
)
from sieve3_event_generation2_b2_freqai import (
    CURRENT_IDENTITY_PROFILES,
    RETAINED_SCOPES,
)
from sieve3_event_generation2_common import (
    completed_output_audit,
    generation_batch,
    read_json,
    sha256_file,
)


GENERATION_DIR = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "sieve3_event_reaction"
    / "generation2"
)
GENERATION_MANIFEST = GENERATION_DIR / "generation_manifest.json"
TERMINAL_STATES = {
    "completed",
    "completed_insufficient_evidence",
    "unsupported_in_tested_scope",
    "deferred_for_data",
    "blocked_integrity_defect",
}
SCORE_KEY = ["profile_id", "pair", "window", "scope", "target"]


BATCH_SPECS: dict[str, dict[str, Any]] = {
    "G2-B1": {
        "directory": "g2-b1/freqai",
        "profiles": B1_PROFILES,
        "scores": "g2_b1_scores.csv",
        "observations": "g2_b1_controlled_observations.csv",
        "pairwise": "g2_b1_pairwise_portability.csv",
        "intersection": "g2_b1_controlled_intersection.csv",
        "direct": "g2_b1_direct_support.csv",
        "score_required": [
            *SCORE_KEY,
            "status",
            "rows",
            "target_family",
            "horizon_hours",
        ],
        "intersection_required": [
            "profile_id",
            "scope",
            "target",
            "target_family",
            "horizon_hours",
            "all_controls_rank_mae_majority_positive_pairs",
            "all_controls_rank_mae_majority_positive_windows",
            "median_worst_spearman_delta",
            "median_worst_mae_skill",
        ],
    },
    "G2-B2": {
        "directory": "g2-b2/freqai",
        "profiles": B2_PROFILES,
        "scores": "g2_b2_scores.csv",
        "observations": "g2_b2_controlled_observations.csv",
        "pairwise": "g2_b2_pairwise_portability.csv",
        "intersection": "g2_b2_controlled_intersection.csv",
        "direct": "g2_b2_direct_support.csv",
        "score_required": [
            *SCORE_KEY,
            "status",
            "rows",
            "target_side",
            "target_family",
            "horizon_hours",
        ],
        "intersection_required": [
            "profile_id",
            "scope",
            "target",
            "target_side",
            "target_family",
            "horizon_hours",
            "all_controls_auc_brier_majority_positive_pairs",
            "all_controls_auc_brier_majority_positive_windows",
            "median_worst_auc_delta",
            "median_worst_brier_skill",
        ],
    },
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def _json_audit(path: Path, required: tuple[str, ...]) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False, "errors": ["file missing"]}
    try:
        payload = read_json(path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        return {"path": str(path), "exists": True, "errors": [f"invalid JSON: {exc}"]}
    missing = sorted(set(required).difference(payload))
    errors = [f"missing required fields: {missing}"] if missing else []
    return {
        "path": str(path),
        "exists": True,
        "sha256": sha256_file(path),
        "missing_fields": missing,
        "errors": errors,
    }


def _csv_audit(path: Path, required: list[str], key: list[str] | None) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "exists": False, "errors": ["file missing"]}
    audit = frame_audit(path, required, key)
    audit["sha256"] = sha256_file(path)
    errors: list[str] = []
    if audit.get("missing_required_columns"):
        errors.append(f"missing columns: {audit['missing_required_columns']}")
    if audit.get("duplicate_key_rows"):
        errors.append(f"duplicate keys: {audit['duplicate_key_rows']}")
    if audit.get("infinite_numeric_cells"):
        errors.append(f"infinite numeric cells: {audit['infinite_numeric_cells']}")
    audit["errors"] = errors
    return audit


def _score_surface_audit(path: Path, expected_profiles: tuple[str, ...]) -> dict[str, Any]:
    frame = pd.read_csv(path, usecols=SCORE_KEY)
    profiles = tuple(frame["profile_id"].dropna().drop_duplicates())
    signatures: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        selected = frame[frame["profile_id"].eq(profile)]
        signatures[str(profile)] = {
            "rows": len(selected),
            "comparable_key_digest": key_digest(selected, ["pair", "window", "scope", "target"]),
        }
    errors: list[str] = []
    if set(map(str, profiles)) != set(expected_profiles):
        errors.append("score profile surface differs")
    if len({item["comparable_key_digest"] for item in signatures.values()}) != 1:
        errors.append("score profiles do not share the same comparable keys")
    return {
        "profiles": list(map(str, profiles)),
        "signatures": signatures,
        "all_profiles_share_keys": not any(
            error == "score profiles do not share the same comparable keys" for error in errors
        ),
        "errors": errors,
    }


def _comparison_prefix_columns(frame: DataFrame) -> list[str]:
    return sorted(
        column for column in frame.columns if "_delta_vs_g2b" in column or "_skill_vs_g2b" in column
    )


def _rederive_b1(intersection_path: Path, direct_path: Path) -> dict[str, Any]:
    intersection = pd.read_csv(intersection_path)
    direct = pd.read_csv(direct_path)
    primary = intersection[
        intersection["profile_id"].isin(SIMPLIFIED_CURRENT)
        & intersection["scope"].eq(PRIMARY_SCOPE)
        & intersection["target"].eq(PRIMARY_TARGET)
    ].copy()
    primary["passes_frozen_model_rule"] = (
        pd.to_numeric(primary["all_controls_rank_mae_majority_positive_pairs"], errors="coerce").ge(
            7
        )
        & pd.to_numeric(
            primary["all_controls_rank_mae_majority_positive_windows"], errors="coerce"
        ).ge(2)
        & pd.to_numeric(primary["median_worst_spearman_delta"], errors="coerce").gt(0.0)
        & pd.to_numeric(primary["median_worst_mae_skill"], errors="coerce").gt(0.0)
    )
    survivors = sorted(
        set(map(str, primary.loc[primary["passes_frozen_model_rule"], "profile_id"]))
    )
    direct["support_adequate_rederived"] = (
        pd.to_numeric(direct["distinct_pairs"], errors="coerce").ge(7)
        & pd.to_numeric(direct["distinct_windows"], errors="coerce").ge(2)
        & pd.to_numeric(direct["median_delta"], errors="coerce").gt(0.0)
    )
    required = {"no_nearby_level", "stale_mixed_equal_highest"}
    comparator_pass = direct.groupby("comparator", observed=True)[
        "support_adequate_rederived"
    ].any()
    direct_agrees = required.issubset(set(comparator_pass.index)) and bool(
        comparator_pass.reindex(sorted(required)).fillna(False).all()
    )
    released = bool(survivors) and direct_agrees
    return {
        "model_survivors": survivors,
        "direct_comparator_support": {
            str(key): bool(value) for key, value in comparator_pass.items()
        },
        "direct_agrees": direct_agrees,
        "release_generation3_market_regime_theory": released,
        "status": "completed" if released else "unsupported_in_tested_scope",
    }


def _b2_direct_summary(path: Path) -> dict[str, dict[str, Any]]:
    summary_path = path.parent / "g2_b2_direct_support_summary.json"
    payload = read_json(summary_path)
    return {str(item["scope"]): item for item in payload["scopes"]}


def _rederive_b2(intersection_path: Path, direct_path: Path) -> dict[str, Any]:
    intersection = pd.read_csv(intersection_path)
    direct_by_scope = _b2_direct_summary(direct_path)
    candidates = intersection[
        intersection["profile_id"].isin(CURRENT_IDENTITY_PROFILES)
        & intersection["scope"].isin(RETAINED_SCOPES)
        & intersection["target_family"].eq("path_order_binary")
        & pd.to_numeric(intersection["horizon_hours"], errors="coerce").eq(4)
    ].copy()
    candidates = candidates[
        candidates.apply(
            lambda row: str(row["target_side"]) == str(scope_side(str(row["scope"]))),
            axis=1,
        )
    ]
    candidates["model_passed"] = (
        pd.to_numeric(
            candidates["all_controls_auc_brier_majority_positive_pairs"],
            errors="coerce",
        ).ge(3)
        & pd.to_numeric(
            candidates["all_controls_auc_brier_majority_positive_windows"],
            errors="coerce",
        ).ge(2)
        & pd.to_numeric(candidates["median_worst_auc_delta"], errors="coerce").gt(0.0)
        & pd.to_numeric(candidates["median_worst_brier_skill"], errors="coerce").gt(0.0)
    )
    candidates["direct_passed"] = candidates["scope"].map(
        lambda scope: bool(direct_by_scope.get(str(scope), {}).get("support_passed"))
    )
    candidates["direct_sign"] = candidates["scope"].map(
        lambda scope: str(direct_by_scope.get(str(scope), {}).get("direct_sign", "unresolved"))
    )
    candidates["passes_frozen_rule"] = candidates["model_passed"] & candidates["direct_passed"]
    survivors = sorted(
        (
            str(row.profile_id),
            str(row.scope),
            str(row.target),
            str(row.direct_sign),
        )
        for row in candidates[candidates["passes_frozen_rule"]].itertuples(index=False)
    )
    reason_or_family = [
        item
        for item in survivors
        if item[0] in {"family_only_current", "reason_only_current", "full_identity_current"}
    ]
    stable = [item for item in reason_or_family if item[3] in {"positive", "negative"}]
    released = bool(stable)
    return {
        "survivors": [list(item) for item in survivors],
        "reason_or_family_survivors": [list(item) for item in reason_or_family],
        "stable_direct_sign_survivors": [list(item) for item in stable],
        "release_generation3_exit_reason_context_theory": released,
        "status": "completed" if released else "unsupported_in_tested_scope",
    }


def _terminal_matches(
    batch_id: str, terminal: dict[str, Any], rederived: dict[str, Any]
) -> list[str]:
    errors: list[str] = []
    if terminal.get("status") != rederived["status"]:
        errors.append("terminal status differs from independent derivation")
    if batch_id == "G2-B1":
        if sorted(map(str, terminal.get("model_survivors", []))) != rederived["model_survivors"]:
            errors.append("B1 survivor list differs from independent derivation")
        if bool(terminal.get("release_generation3_market_regime_theory")) != bool(
            rederived["release_generation3_market_regime_theory"]
        ):
            errors.append("B1 Generation-3 release decision differs")
    else:
        terminal_stable = sorted(
            (
                str(item["profile_id"]),
                str(item["scope"]),
                str(item["target"]),
                str(item["direct_sign"]),
            )
            for item in terminal.get("stable_direct_sign_survivors", [])
        )
        rederived_stable = sorted(tuple(item) for item in rederived["stable_direct_sign_survivors"])
        if terminal_stable != rederived_stable:
            errors.append("B2 stable-direct-sign survivor list differs")
        if bool(terminal.get("release_generation3_exit_reason_context_theory")) != bool(
            rederived["release_generation3_exit_reason_context_theory"]
        ):
            errors.append("B2 Generation-3 release decision differs")
    return errors


def batch_audit(  # noqa: C901
    generation: dict[str, Any], batch_id: str
) -> dict[str, Any]:
    spec = BATCH_SPECS[batch_id]
    frozen = generation_batch(generation, batch_id)
    directory = GENERATION_DIR / spec["directory"]
    manifest_path = directory / "manifest.json"
    audit: dict[str, Any] = {
        "batch": batch_id,
        "frozen_status": frozen.get("status"),
        "directory": str(directory),
        "manifest": str(manifest_path),
        "errors": [],
    }
    if not manifest_path.is_file():
        audit["errors"].append("batch manifest missing")
        audit["ready"] = False
        return audit
    manifest = read_json(manifest_path)
    profiles = tuple(item["profile_id"] for item in manifest.get("commands", []))
    audit["manifest_profiles"] = list(profiles)
    audit["profile_contract_matches"] = profiles == tuple(spec["profiles"])
    if not audit["profile_contract_matches"]:
        audit["errors"].append("profile contract differs")
    statuses = (
        pd.Series(
            [str(item.get("status")) for item in manifest.get("commands", [])], dtype="string"
        )
        .value_counts()
        .to_dict()
    )
    audit["command_status_counts"] = {str(key): int(value) for key, value in statuses.items()}
    failed = [item for item in manifest["commands"] if item.get("status") == "failed"]
    if failed:
        audit["errors"].extend(
            f"profile failed: {item['profile_id']} returncode={item.get('returncode')}"
            for item in failed
        )
    commands_complete = bool(manifest["commands"]) and all(
        item.get("status") == "completed" and int(item.get("returncode", 1)) == 0
        for item in manifest["commands"]
    )
    audit["commands_complete"] = commands_complete
    completed_new = [
        item
        for item in manifest["commands"]
        if item.get("status") == "completed" and not item.get("reuse_source")
    ]
    completed_new_audits = [model_output_audit(item, manifest) for item in completed_new]
    audit["completed_new_profile_output_audits"] = completed_new_audits
    sealed_masks = {
        str(item.get("eligible_mask_digest"))
        for item in manifest.get("reuse_audits", [])
        if item.get("eligible_mask_digest")
    }
    for item in completed_new_audits:
        audit["errors"].extend(
            f"completed profile {item['profile_id']}: {error}" for error in item["errors"]
        )
        if sealed_masks and item["eligible_mask_digest"] not in sealed_masks:
            audit["errors"].append(
                f"completed profile {item['profile_id']}: eligible mask differs from sealed reuse"
            )
    preflight = _json_audit(directory / "preflight.json", ("batch", "passed", "errors"))
    smoke = _json_audit(directory / "smoke_audit.json", ("batch", "passed", "errors", "profiles"))
    audit["preflight"] = preflight
    audit["smoke"] = smoke
    for name, item in (("preflight", preflight), ("smoke", smoke)):
        if item["errors"]:
            audit["errors"].extend(f"{name}: {error}" for error in item["errors"])
        elif not read_json(Path(item["path"])).get("passed"):
            audit["errors"].append(f"{name} did not pass")
    if not commands_complete:
        audit["ready"] = False
        return audit
    output_audit = completed_output_audit(manifest)
    audit["current_output_audit"] = output_audit
    if not output_audit["passed"]:
        audit["errors"].extend(f"model output: {error}" for error in output_audit["errors"])
    artifacts = {
        "scores": _csv_audit(directory / spec["scores"], spec["score_required"], SCORE_KEY),
        "observations": _csv_audit(
            directory / spec["observations"],
            [*SCORE_KEY, "comparator_id", "rows"],
            ["profile_id", "comparator_id", "pair", "window", "scope", "target"],
        ),
        "pairwise": _csv_audit(
            directory / spec["pairwise"],
            ["profile_id", "comparator_id", "scope", "target"],
            ["profile_id", "comparator_id", "scope", "target"],
        ),
        "intersection": _csv_audit(
            directory / spec["intersection"],
            spec["intersection_required"],
            ["profile_id", "scope", "target"],
        ),
        "direct": _csv_audit(directory / spec["direct"], [], None),
        "terminal": _json_audit(
            directory / "terminal_verdict.json", ("batch", "status", "finished_at")
        ),
        "summary": _json_audit(directory / "run_summary.json", ("batch", "finished_at", "verdict")),
        "saved_output_audit": _json_audit(
            directory / "output_audit.json", ("profiles", "passed", "errors")
        ),
    }
    audit["artifacts"] = artifacts
    for name, item in artifacts.items():
        audit["errors"].extend(f"{name}: {error}" for error in item["errors"])
    if audit["errors"]:
        audit["ready"] = False
        return audit
    scores = pd.read_csv(directory / spec["scores"], nrows=5)
    comparison_columns = _comparison_prefix_columns(scores)
    audit["comparison_columns"] = len(comparison_columns)
    if not comparison_columns:
        audit["errors"].append("scores contain no Generation-2 comparison columns")
    score_surface = _score_surface_audit(directory / spec["scores"], tuple(spec["profiles"]))
    audit["score_surface"] = score_surface
    audit["errors"].extend(score_surface["errors"])
    saved_output = read_json(directory / "output_audit.json")
    if not saved_output.get("passed"):
        audit["errors"].append("saved output audit did not pass")
    current_masks = output_audit.get("eligible_mask_digests", [])
    saved_masks = saved_output.get("eligible_mask_digests", [])
    if current_masks != saved_masks:
        audit["errors"].append("current eligible-mask digest differs from saved output audit")
    intersection_path = directory / spec["intersection"]
    direct_path = directory / spec["direct"]
    rederived = (
        _rederive_b1(intersection_path, direct_path)
        if batch_id == "G2-B1"
        else _rederive_b2(intersection_path, direct_path)
    )
    terminal = read_json(directory / "terminal_verdict.json")
    audit["rederived_verdict"] = rederived
    audit["errors"].extend(_terminal_matches(batch_id, terminal, rederived))
    audit["terminal_status"] = str(terminal.get("status"))
    if audit["terminal_status"] not in TERMINAL_STATES:
        audit["errors"].append("terminal status is not allowed by generation gate")
    audit["ready"] = not audit["errors"]
    return audit


def build_review() -> dict[str, Any]:
    generation = read_json(GENERATION_MANIFEST)
    audits = [batch_audit(generation, batch_id) for batch_id in BATCH_SPECS]
    errors = [f"{audit['batch']}: {error}" for audit in audits for error in audit.get("errors", [])]
    nonterminal = {
        audit["batch"]: audit.get("command_status_counts", {})
        for audit in audits
        if not audit.get("commands_complete")
    }
    all_ready = all(audit.get("ready") for audit in audits)
    return {
        "schema_version": 1,
        "generated_at": now_iso(),
        "generation": 2,
        "status": (
            "review_gate_open"
            if all_ready and not errors
            else "execution_active"
            if nonterminal and not errors
            else "integrity_errors"
        ),
        "generation_manifest": str(GENERATION_MANIFEST),
        "generation_manifest_sha256": sha256_file(GENERATION_MANIFEST),
        "all_batch_evidence_ready": all_ready,
        "nonterminal_batches": nonterminal,
        "integrity_error_count": len(errors),
        "integrity_errors": errors,
        "batch_audits": audits,
        "generation3_prohibited": not all_ready or bool(errors),
        "interpretation_contract": {
            "joint_review_required_before_generation3_freeze": True,
            "release_flags_are_candidates_not_automatic_execution": True,
            "activity_is_not_direction": True,
            "no_trading_action_promotion": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output", type=Path, default=GENERATION_DIR / "generation2_review.json")
    args = parser.parse_args()
    review = build_review()
    if args.preflight:
        print(
            json.dumps(
                {
                    "status": review["status"],
                    "all_batch_evidence_ready": review["all_batch_evidence_ready"],
                    "nonterminal_batches": review["nonterminal_batches"],
                    "integrity_error_count": review["integrity_error_count"],
                    "integrity_errors": review["integrity_errors"],
                },
                indent=2,
            )
        )
        return 1 if review["integrity_errors"] else 0
    if not review["all_batch_evidence_ready"] or review["integrity_errors"]:
        raise RuntimeError(
            "Generation-2 review gate is not ready: "
            + json.dumps(
                {
                    "nonterminal": review["nonterminal_batches"],
                    "errors": review["integrity_errors"],
                }
            )
        )
    atomic_json(args.output.resolve(), review)
    print(json.dumps({"status": review["status"], "output": str(args.output.resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
