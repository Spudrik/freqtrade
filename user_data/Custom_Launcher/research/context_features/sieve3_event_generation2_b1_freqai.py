"""Run frozen G2-B1 level-component reaction attribution with exact G1 reuse."""

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
from sieve3_event_generation1_b3_freqai import (
    COUNT_FIELDS,
    EXACT_FIELDS,
    RELATIONSHIP_FIELDS,
    TARGETS,
    TIMEFRAME_FIELDS,
    actuals_for_pair,
    fields,
    scope_dates,
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
DEFAULT_OUTPUT = GENERATION_DIR / "g2-b1" / "freqai"
GENERATION_MANIFEST = GENERATION_DIR / "generation_manifest.json"
G1_REVIEW = GENERATION_DIR.parent / "generation1" / "generation1_review.json"
G1_JOINT_VERDICT = GENERATION_DIR.parent / "generation1" / "generation1_joint_verdict.json"
SOURCE_OUTPUT = GENERATION_DIR.parent / "generation1" / "g1-b3" / "freqai"
SOURCE_MANIFEST = SOURCE_OUTPUT / "manifest.json"
DEFAULT_CACHE = SOURCE_OUTPUT.parent / "cache"
DEFAULT_SCOPES = SOURCE_OUTPUT.parent / "event_scopes.parquet"
DEFAULT_DIRECT = SOURCE_OUTPUT / "g1_b3_direct_portability_summary.csv"
DEFAULT_PYTHON = (
    REPO_ROOT / "runtime" / "venvs" / "freqtrade-backtest-06" / "Scripts" / "python.exe"
)
DEFAULT_PAIRS = (
    "BTC/USDT:USDT,ETH/USDT:USDT,BNB/USDT:USDT,SOL/USDT:USDT,"
    "XRP/USDT:USDT,ADA/USDT:USDT,DOGE/USDT:USDT,TRX/USDT:USDT,"
    "AVAX/USDT:USDT,LINK/USDT:USDT"
)


COUNTS_CURRENT = fields("level", COUNT_FIELDS)
COUNTS_STALE = fields("stale_level", COUNT_FIELDS)
RELATIONSHIP_CURRENT = fields("level", RELATIONSHIP_FIELDS)
RELATIONSHIP_STALE = fields("stale_level", RELATIONSHIP_FIELDS)
TIMEFRAME_CURRENT = fields("level", TIMEFRAME_FIELDS)
TIMEFRAME_STALE = fields("stale_level", TIMEFRAME_FIELDS)
COUNTS_RELATIONSHIP_CURRENT = fields("level", (*RELATIONSHIP_FIELDS, *COUNT_FIELDS))
COUNTS_RELATIONSHIP_STALE = fields("stale_level", (*RELATIONSHIP_FIELDS, *COUNT_FIELDS))
COUNTS_TIMEFRAME_CURRENT = fields("level", (*COUNT_FIELDS, *TIMEFRAME_FIELDS))
COUNTS_TIMEFRAME_STALE = fields("stale_level", (*COUNT_FIELDS, *TIMEFRAME_FIELDS))
LEVEL_COMPONENTS_CURRENT = fields("level", (*RELATIONSHIP_FIELDS, *COUNT_FIELDS, *TIMEFRAME_FIELDS))
LEVEL_COMPONENTS_STALE = fields(
    "stale_level", (*RELATIONSHIP_FIELDS, *COUNT_FIELDS, *TIMEFRAME_FIELDS)
)
EXACT_CURRENT = fields("level", EXACT_FIELDS)
FULL_CURRENT = [*LEVEL_COMPONENTS_CURRENT, *EXACT_CURRENT]
FULL_STALE = fields(
    "stale_level", (*RELATIONSHIP_FIELDS, *COUNT_FIELDS, *TIMEFRAME_FIELDS, *EXACT_FIELDS)
)


def _profile(columns: list[str], theory: str) -> dict[str, Any]:
    return {
        "strategy": "Sieve3EventReactionFixedIndicatorIdentityFreqAIResearchStrategy",
        "shift_hours": 0,
        "feature_columns": columns,
        "theory": theory,
    }


PROFILES: dict[str, dict[str, Any]] = {
    "fixed_indicator_control": {
        "strategy": "Sieve3EventReactionFixedIndicatorControlFreqAIResearchStrategy",
        "shift_hours": 0,
        "feature_columns": [],
        "theory": "Frozen price and indicator state without nearby-level fields.",
    },
    "counts_only_current": _profile(
        COUNTS_CURRENT, "Only current same-direction and opposing level counts."
    ),
    "counts_only_stale": _profile(
        COUNTS_STALE, "Like-for-like counts recomputed from levels stale by 168 hours."
    ),
    "relationship_only_current": _profile(
        RELATIONSHIP_CURRENT,
        "Only current one-sided, mixed, displaced or no-level relationship state.",
    ),
    "relationship_only_stale": _profile(
        RELATIONSHIP_STALE, "Like-for-like relationship state from 168-hour-stale levels."
    ),
    "timeframe_only_current": _profile(
        TIMEFRAME_CURRENT, "Only current maximum-timeframe and mixed-precedence fields."
    ),
    "timeframe_only_stale": _profile(
        TIMEFRAME_STALE, "Like-for-like timeframe state from 168-hour-stale levels."
    ),
    "counts_plus_relationship_current": _profile(
        COUNTS_RELATIONSHIP_CURRENT,
        "Current counts plus relationship state without timeframe fields.",
    ),
    "counts_plus_relationship_stale": _profile(
        COUNTS_RELATIONSHIP_STALE, "Stale counts plus stale relationship state."
    ),
    "counts_plus_timeframe_current": _profile(
        COUNTS_TIMEFRAME_CURRENT,
        "Current counts plus timeframe fields without relationship identity.",
    ),
    "counts_plus_timeframe_stale": _profile(
        COUNTS_TIMEFRAME_STALE, "Stale counts plus stale timeframe fields."
    ),
    "level_components_current": _profile(
        LEVEL_COMPONENTS_CURRENT, "Current counts, relationship and timeframe components."
    ),
    "level_components_stale": _profile(
        LEVEL_COMPONENTS_STALE, "Like-for-like full component block from stale levels."
    ),
    "exact_mixed_equal_current": _profile(
        EXACT_CURRENT, "Only the explicit current mixed/equal-highest identity."
    ),
    "full_current_state": _profile(
        FULL_CURRENT, "Current components plus explicit mixed/equal-highest identity."
    ),
    "full_stale_state": _profile(
        FULL_STALE, "Full like-for-like state recomputed from stale levels."
    ),
}

REUSED = {
    "fixed_indicator_control": "fixed_indicator_control",
    "counts_plus_relationship_current": "components_no_timeframe",
    "counts_plus_timeframe_current": "components_no_mixed",
    "level_components_current": "level_components",
    "exact_mixed_equal_current": "exact_mixed_equal",
    "full_current_state": "full_current_state",
    "full_stale_state": "stale_full_state",
}
NEW_PROFILE_IDS = tuple(profile for profile in PROFILES if profile not in REUSED)
ALL_PROFILE_IDS = tuple(PROFILES)
COMPARISON_CONTRACT = {
    "counts_only_current": ["fixed_indicator_control", "counts_only_stale"],
    "relationship_only_current": ["fixed_indicator_control", "relationship_only_stale"],
    "timeframe_only_current": ["fixed_indicator_control", "timeframe_only_stale"],
    "counts_plus_relationship_current": [
        "fixed_indicator_control",
        "counts_only_current",
        "relationship_only_current",
        "counts_plus_relationship_stale",
    ],
    "counts_plus_timeframe_current": [
        "fixed_indicator_control",
        "counts_only_current",
        "timeframe_only_current",
        "counts_plus_timeframe_stale",
    ],
    "level_components_current": [
        "fixed_indicator_control",
        "counts_plus_relationship_current",
        "counts_plus_timeframe_current",
        "level_components_stale",
    ],
    "full_current_state": [
        "fixed_indicator_control",
        "level_components_current",
        "exact_mixed_equal_current",
        "full_stale_state",
    ],
}
COMPARATOR_PREFIXES = {
    profile: f"g2b1_{index:02d}" for index, profile in enumerate(PROFILES, start=1)
}
PRIMARY_SCOPE = "family:multi_timeframe:long"
PRIMARY_TARGET = "&-one_atr_touch_observed_4h"
SIMPLIFIED_CURRENT = (
    "counts_only_current",
    "relationship_only_current",
    "timeframe_only_current",
    "counts_plus_relationship_current",
    "counts_plus_timeframe_current",
    "level_components_current",
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
    review = read_json(G1_REVIEW)
    sealed_batch = review_batch(review, "G1-B3")
    run_id = f"sieve3_event_g2_b1_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    reused_commands: list[dict[str, Any]] = []
    reuse_audits: list[dict[str, Any]] = []
    selected_reuse = {key: value for key, value in REUSED.items() if key in selected_profiles}
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
    commands = [command_by_profile[profile] for profile in selected_profiles]
    return {
        "schema_version": 1,
        "batch": "G2-B1",
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
        "commands": commands,
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
        if item["strategy"] != definition["strategy"]:
            errors.append(f"{item['profile_id']}: strategy differs")
        if list(settings.get("event_feature_columns") or []) != list(definition["feature_columns"]):
            errors.append(f"{item['profile_id']}: feature columns differ")
        if int(settings.get("event_feature_shift_hours", 0) or 0) != 0:
            errors.append(f"{item['profile_id']}: unexpected feature shift")
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


def refresh_pending_new_configs(
    manifest: dict[str, Any], manifest_path: Path, base_config: Path
) -> None:
    """Regenerate never-run G2 configs from the current frozen definitions."""
    base = read_json(base_config)
    pairs = tuple(manifest["pairs"])
    cache_dir = Path(manifest["event_cache_dir"])
    changed = False
    for item in manifest["commands"]:
        if item["profile_id"] in REUSED or item.get("status") != "pending":
            continue
        definition = PROFILES[item["profile_id"]]
        config = profile_config(base, item["identifier"], definition, pairs, cache_dir)
        settings = config["sieve3_event_reaction"]
        settings["target_names"] = TARGETS
        settings["event_feature_columns"] = list(definition["feature_columns"])
        atomic_json(Path(item["config_path"]), config)
        item["event_feature_columns"] = list(definition["feature_columns"])
        changed = True
    if changed:
        atomic_json(manifest_path, manifest)


def refresh_reuse_audits(manifest: dict[str, Any], manifest_path: Path) -> None:
    reused_commands, audits = reuse_profiles(
        alias_to_source=REUSED,
        expected_profiles=PROFILES,
        source_manifest=read_json(SOURCE_MANIFEST),
        sealed_batch_audit=review_batch(read_json(G1_REVIEW), "G1-B3"),
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
    frozen = generation_batch(generation, "G2-B1")
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
    if (
        tuple(
            manifest["commands"][index]["profile_id"] for index in range(len(manifest["commands"]))
        )
        != ALL_PROFILE_IDS
    ):
        errors.append("full manifest profile order differs from frozen implementation")
    if manifest["comparison_contract"] != frozen["comparison_contract"]:
        errors.append("comparison contract differs from frozen generation manifest")
    if sha256_file(G1_JOINT_VERDICT) != generation["source_generation"]["joint_verdict_sha256"]:
        errors.append("Generation-1 joint verdict hash differs from frozen source hash")
    for path in (Path(manifest["event_scopes"]), DEFAULT_DIRECT, SOURCE_MANIFEST, G1_REVIEW):
        if not path.is_file():
            errors.append(f"required source artifact missing: {path}")
    result = {
        "schema_version": 1,
        "batch": "G2-B1",
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
                    for target in TARGETS:
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
    scores["target_family"] = metadata.map(lambda item: item[0])
    scores["horizon_hours"] = metadata.map(lambda item: item[1])
    scores = add_comparisons(scores, COMPARATOR_PREFIXES)
    scores.to_csv(output_dir / "g2_b1_scores.csv", index=False)
    return scores


def direct_support(output_dir: Path) -> dict[str, Any]:
    direct = pd.read_csv(DEFAULT_DIRECT)
    selected = direct[
        direct["target"].eq(PRIMARY_TARGET)
        & direct["comparator"].isin(["no_nearby_level", "stale_mixed_equal_highest"])
        & direct["band_pct"].isin([0.005, 0.01])
    ].copy()
    selected["support_adequate"] = (
        selected["distinct_pairs"].ge(7)
        & selected["distinct_windows"].ge(2)
        & selected["median_delta"].gt(0.0)
    )
    selected.to_csv(output_dir / "g2_b1_direct_support.csv", index=False)
    comparator_pass = selected.groupby("comparator", observed=True)["support_adequate"].any()
    required = {"no_nearby_level", "stale_mixed_equal_highest"}
    return {
        "source": str(DEFAULT_DIRECT),
        "rows": len(selected),
        "comparators": comparator_pass.to_dict(),
        "agrees": required.issubset(comparator_pass.index)
        and bool(comparator_pass.loc[list(required)].all()),
    }


def terminal_verdict(intersection: DataFrame, direct: dict[str, Any]) -> dict[str, Any]:
    primary = intersection[
        intersection["profile_id"].isin(SIMPLIFIED_CURRENT)
        & intersection["scope"].eq(PRIMARY_SCOPE)
        & intersection["target"].eq(PRIMARY_TARGET)
    ].copy()
    primary["passes_frozen_model_rule"] = (
        primary["all_controls_rank_mae_majority_positive_pairs"].ge(7)
        & primary["all_controls_rank_mae_majority_positive_windows"].ge(2)
        & primary["median_worst_spearman_delta"].gt(0.0)
        & primary["median_worst_mae_skill"].gt(0.0)
    )
    survivors = primary.loc[primary["passes_frozen_model_rule"], "profile_id"].tolist()
    supported = bool(survivors) and bool(direct["agrees"])
    return {
        "schema_version": 1,
        "batch": "G2-B1",
        "finished_at": now_iso(),
        "status": "completed" if supported else "unsupported_in_tested_scope",
        "primary_scope": PRIMARY_SCOPE,
        "primary_target": PRIMARY_TARGET,
        "direct_support": direct,
        "model_survivors": survivors,
        "release_generation3_market_regime_theory": supported,
        "interpretation": (
            "At least one simplified current level block retained reaction information "
            "against every frozen control."
            if supported
            else "No simplified current level block satisfied the complete frozen "
            "attribution and direct-support rule."
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
        raise RuntimeError("G2-B1 output audit failed: " + "; ".join(output_audit["errors"]))
    scores = score_manifest(manifest, output_dir)
    long, pairwise, intersection = controlled_portability(
        scores=scores,
        comparison_contract=COMPARISON_CONTRACT,
        comparator_prefixes=COMPARATOR_PREFIXES,
        metadata_columns=["target_family", "horizon_hours"],
        include_binary=False,
    )
    long.to_csv(output_dir / "g2_b1_controlled_observations.csv", index=False)
    pairwise.to_csv(output_dir / "g2_b1_pairwise_portability.csv", index=False)
    intersection.to_csv(output_dir / "g2_b1_controlled_intersection.csv", index=False)
    direct = direct_support(output_dir)
    verdict = terminal_verdict(intersection, direct)
    atomic_json(output_dir / "terminal_verdict.json", verdict)
    summary = {
        "finished_at": now_iso(),
        "batch": "G2-B1",
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
        f"G2-B1 completed with {len(scores)} score rows; verdict `{verdict['status']}`.",
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
            "batch": "G2-B1",
            "checked_at": now_iso(),
            "purpose": "BTC one-month execution smoke for every distinct new feature contract.",
        }
    )
    atomic_json(output_dir / "smoke_audit.json", audit)
    print(
        json.dumps(
            {
                "batch": "G2-B1",
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
            raise ValueError("Existing G2-B1 manifest profile surface differs")
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
        refresh_pending_new_configs(manifest, manifest_path, args.base_config.resolve())
        refresh_reuse_audits(manifest, manifest_path)
        result = preflight(manifest, output_dir)
        print(
            json.dumps(
                {"batch": "G2-B1", "passed": result["passed"], "errors": result["errors"]}, indent=2
            )
        )
        return 0 if result["passed"] else 1
    return run_full(manifest, output_dir, score_only=args.score_only)


if __name__ == "__main__":
    raise SystemExit(main())
