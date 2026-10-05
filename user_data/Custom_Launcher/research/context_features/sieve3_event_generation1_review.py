"""Audit the batch gate and compact evidence surfaces for Sieve3 Generation 1.

This is a read-only post-processor until every G1 batch is terminal.  It does
not rerun FreqAI, change a batch verdict, select a branch, or promote trading
logic.  Final mode writes one compact ``generation1_review.json`` beside the
generation manifest; ``--preflight`` only reports readiness to stdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.feather as feather
from pandas import DataFrame


TERMINAL_STATES = {
    "completed",
    "completed_insufficient_evidence",
    "unsupported_in_tested_scope",
    "deferred_for_data",
    "blocked_integrity_defect",
}
EVIDENCE_STATES = {
    "completed",
    "completed_insufficient_evidence",
    "unsupported_in_tested_scope",
}
SCORE_KEY = ["profile_id", "pair", "window", "scope", "target"]
COMPARABLE_KEY = ["pair", "window", "scope", "target"]
SCORE_REQUIRED = [
    *SCORE_KEY,
    "rows",
    "status",
    "target_family",
    "horizon_hours",
]

BATCH_SPECS: dict[str, dict[str, Any]] = {
    "G1-B1": {
        "directory": "g1-b1",
        "profiles": (
            "lean_price_control",
            "fixed_indicator_control",
            "event_onset_identity",
            "event_onset_active_identity",
            "shifted_event_placebo",
            "shifted_onset_placebo",
        ),
        "comparison_prefixes": ("lean", "indicator", "shifted_active", "shifted_onset"),
        "scores": "g1_b1_scores.csv",
        "portability": "g1_b1_portability_summary.csv",
        "portability_required": (
            "profile_id",
            "target",
            "target_family",
            "horizon_hours",
            "rank_mae_vs_both_controls_majority_positive_pairs",
            "rank_mae_vs_both_controls_majority_positive_windows",
        ),
    },
    "G1-B2": {
        "directory": "g1-b2",
        "profiles": (
            "fixed_indicator_control",
            "shifted_exact_event",
            "shifted_exact_active",
            "exact_event",
            "exact_event_active",
            "components_all",
            "components_no_d1",
            "components_no_vp_bos",
            "components_no_h4_retest",
            "exact_plus_components",
        ),
        "comparison_prefixes": (
            "indicator",
            "shifted_exact",
            "shifted_active",
            "exact",
            "components",
            "no_d1",
            "no_vp_bos",
            "no_h4_retest",
        ),
        "scores": "g1_b2_scores.csv",
        "portability": "g1_b2_portability_summary.csv",
        "portability_required": (
            "profile_id",
            "target",
            "target_family",
            "horizon_hours",
            "rank_mae_majority_positive_pairs",
            "rank_mae_majority_positive_windows",
        ),
    },
    "G1-B3": {
        "directory": "g1-b3",
        "profiles": (
            "fixed_indicator_control",
            "level_components",
            "exact_mixed_equal",
            "stale_full_state",
            "full_current_state",
            "components_no_mixed",
            "components_no_timeframe",
        ),
        "comparison_prefixes": (
            "indicator",
            "components",
            "exact",
            "stale",
            "no_mixed",
            "no_timeframe",
        ),
        "scores": "g1_b3_scores.csv",
        "portability": "g1_b3_portability_summary.csv",
        "portability_required": (
            "profile_id",
            "target",
            "target_family",
            "horizon_hours",
            "rank_mae_majority_positive_pairs",
            "rank_mae_majority_positive_windows",
        ),
        "extra_artifacts": (
            {
                "name": "g1_b3_direct_state_summary.csv",
                "required": ["pair", "window", "side", "band_pct", "target"],
                "key": ["pair", "window", "side", "band_pct", "target"],
            },
            {
                "name": "g1_b3_direct_portability_summary.csv",
                "required": [
                    "comparator",
                    "band_pct",
                    "target",
                    "target_family",
                    "horizon_hours",
                    "sample_rows",
                    "comparator_sample_rows",
                    "classification",
                ],
                "key": ["comparator", "band_pct", "target"],
            },
        ),
    },
    "G1-B4": {
        "directory": "g1-b4",
        "profiles": (
            "open_trade_price_indicator_control",
            "shifted_exit_family_placebo",
            "exit_family_identity",
            "exit_identity_plus_trade_state",
            "trade_state_with_exit_family_ablation",
        ),
        "comparison_prefixes": ("control", "shifted", "identity", "state"),
        "scores": "g1_b4_scores.csv",
        "portability": "g1_b4_portability_summary.csv",
        "portability_required": (
            "profile_id",
            "target",
            "target_side",
            "target_family",
            "horizon_hours",
            "rank_mae_majority_positive_pairs",
            "rank_mae_majority_positive_windows",
            "auc_brier_majority_positive_pairs",
            "auc_brier_majority_positive_windows",
        ),
        "extra_artifacts": (
            {
                "name": "../direct_exit_role_summary.csv",
                "required": [
                    "group",
                    "exit_family",
                    "entry_side",
                    "analysis_reason_category",
                    "analysis_action_role",
                    "pair",
                    "window",
                    "horizon_hours",
                    "event_favourable_before_adverse_rows",
                    "control_favourable_before_adverse_rows",
                    "unique_event_favourable_before_adverse_paths",
                    "unique_event_favourable_before_adverse_times",
                ],
                "key": [
                    "group",
                    "exit_family",
                    "entry_side",
                    "analysis_reason_category",
                    "analysis_action_role",
                    "pair",
                    "window",
                    "horizon_hours",
                ],
            },
            {
                "name": "../direct_exit_role_portability.csv",
                "required": [
                    "group",
                    "exit_family",
                    "entry_side",
                    "analysis_reason_category",
                    "analysis_action_role",
                    "horizon_hours",
                    "adequate_cells",
                    *[
                        f"{metric}_{suffix}"
                        for metric in (
                            "hold_instead_delta",
                            "missed_additional_profit",
                            "avoided_loss_after_exit",
                            "net_exit_regret",
                            "favourable_peak_step",
                            "adverse_peak_step",
                            "favourable_before_adverse",
                        )
                        for suffix in (
                            "adequate_cells",
                            "distinct_adequate_pairs",
                            "distinct_adequate_windows",
                        )
                    ],
                ],
                "key": [
                    "group",
                    "exit_family",
                    "entry_side",
                    "analysis_reason_category",
                    "analysis_action_role",
                    "horizon_hours",
                ],
            },
        ),
    },
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def infinite_cells(frame: DataFrame) -> int:
    total = 0
    for column in frame.columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        total += int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum())
    return total


def key_digest(frame: DataFrame, columns: list[str]) -> str:
    ordered = frame[columns].astype("string").sort_values(columns, kind="stable")
    hashed = pd.util.hash_pandas_object(ordered, index=False).to_numpy()
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def frame_audit(path: Path, required: list[str], key: list[str] | None) -> dict[str, Any]:
    frame = pd.read_csv(path, low_memory=False)
    missing = sorted(set(required).difference(frame.columns))
    duplicate_rows = int(frame.duplicated().sum())
    duplicate_keys = None
    if key and not missing and set(key).issubset(frame.columns):
        duplicate_keys = int(frame.duplicated(key, keep=False).sum())
    return {
        "path": str(path),
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "missing_required_columns": missing,
        "duplicate_rows": duplicate_rows,
        "duplicate_key_rows": duplicate_keys,
        "infinite_numeric_cells": infinite_cells(frame),
    }


def score_audit(path: Path, expected_profiles: tuple[str, ...]) -> dict[str, Any]:
    frame = pd.read_csv(path, low_memory=False)
    audit = frame_audit(path, SCORE_REQUIRED, SCORE_KEY)
    if audit["missing_required_columns"]:
        return audit
    profiles = tuple(frame["profile_id"].dropna().drop_duplicates().tolist())
    signatures: dict[str, dict[str, Any]] = {}
    for profile in profiles:
        selected = frame[frame["profile_id"].eq(profile)]
        signatures[profile] = {
            "rows": int(len(selected)),
            "comparable_key_digest": key_digest(selected, COMPARABLE_KEY),
        }
    audit.update(
        {
            "profiles": list(profiles),
            "profile_contract_matches": set(profiles) == set(expected_profiles),
            "comparable_key_signatures": signatures,
            "all_profiles_share_keys": len(
                {item["comparable_key_digest"] for item in signatures.values()}
            )
            <= 1,
            "status_counts": {
                str(key): int(value)
                for key, value in frame["status"].value_counts(dropna=False).items()
            },
            "pairs": int(frame["pair"].nunique(dropna=True)),
            "windows": int(frame["window"].nunique(dropna=True)),
            "scopes": int(frame["scope"].nunique(dropna=True)),
            "targets": int(frame["target"].nunique(dropna=True)),
            "target_families": int(frame["target_family"].nunique(dropna=True)),
            "sample_rows_sum": int(pd.to_numeric(frame["rows"], errors="coerce").fillna(0).sum()),
        }
    )
    return audit


def comparison_columns_missing(
    score_path: Path, prefixes: tuple[str, ...]
) -> list[str]:
    columns = set(pd.read_csv(score_path, nrows=0).columns)
    required: list[str] = []
    for prefix in prefixes:
        required.extend(
            [
                f"prediction_actual_spearman_delta_vs_{prefix}",
                f"mean_absolute_error_skill_vs_{prefix}",
            ]
        )
    return sorted(set(required).difference(columns))


def model_output_audit(
    command: dict[str, Any], manifest: dict[str, Any]
) -> dict[str, Any]:
    command_line = command.get("command", [])
    userdir_index = command_line.index("--userdir") + 1
    userdir = Path(command_line[userdir_index])
    model_dir = userdir / "models" / command["identifier"]
    prediction_dir = model_dir / "backtesting_predictions"
    files = sorted(prediction_dir.glob("*.feather"))
    expected_targets = set(manifest["targets"])
    pair_dates: dict[str, list[pd.Series]] = {}
    eligible_pair_dates: dict[str, list[pd.Series]] = {}
    missing_target_files = 0
    duplicate_column_files = 0
    missing_do_predict_files = 0
    prediction_null_cells = 0
    prediction_infinite_cells = 0
    do_predict_values: set[float] = set()
    for path in files:
        table = feather.read_table(path)
        columns = table.column_names
        missing_target_files += int(not expected_targets.issubset(columns))
        duplicate_column_files += int(len(columns) != len(set(columns)))
        pair_token = path.name.split("_", 2)[1]
        dates = table.column("date").to_pandas()
        pair_dates.setdefault(pair_token, []).append(dates)
        if "do_predict" not in columns:
            missing_do_predict_files += 1
        else:
            do_predict = pd.to_numeric(
                table.column("do_predict").to_pandas(), errors="coerce"
            )
            do_predict_values.update(
                float(value) for value in do_predict.dropna().unique()
            )
            eligible_pair_dates.setdefault(pair_token, []).append(
                dates[do_predict.eq(1.0)]
            )
        if expected_targets.issubset(columns):
            predictions = table.select(sorted(expected_targets)).to_pandas()
            numeric = predictions.to_numpy(dtype=float)
            prediction_null_cells += int(np.isnan(numeric).sum())
            prediction_infinite_cells += int(np.isinf(numeric).sum())

    start_text, end_text = str(manifest["timerange"]).split("-", 1)
    expected_start = pd.Timestamp(start_text, tz="UTC")
    expected_end_exclusive = pd.Timestamp(end_text, tz="UTC")
    expected_rows = int(
        (expected_end_exclusive - expected_start) / pd.Timedelta(hours=1)
    )
    coverage: dict[str, dict[str, Any]] = {}
    for pair_token, chunks in sorted(pair_dates.items()):
        dates = pd.concat(chunks, ignore_index=True).sort_values()
        gaps = dates.diff().dropna()
        coverage[pair_token] = {
            "rows": int(len(dates)),
            "duplicate_dates": int(dates.duplicated().sum()),
            "non_hourly_gaps": int(gaps.ne(pd.Timedelta(hours=1)).sum()),
            "start": None if dates.empty else dates.iloc[0].isoformat(),
            "end": None if dates.empty else dates.iloc[-1].isoformat(),
        }
    expected_end = expected_end_exclusive - pd.Timedelta(hours=1)
    exact_coverage = bool(coverage) and all(
        item["rows"] == expected_rows
        and item["duplicate_dates"] == 0
        and item["non_hourly_gaps"] == 0
        and item["start"] == expected_start.isoformat()
        and item["end"] == expected_end.isoformat()
        for item in coverage.values()
    )
    eligible_coverage: dict[str, dict[str, Any]] = {}
    for pair_token, chunks in sorted(eligible_pair_dates.items()):
        dates = pd.concat(chunks, ignore_index=True).sort_values()
        encoded = dates.astype("string").str.cat(sep="|").encode()
        eligible_coverage[pair_token] = {
            "rows": int(len(dates)),
            "date_digest": hashlib.sha256(encoded).hexdigest(),
        }
    eligible_mask_digest = hashlib.sha256(
        json.dumps(eligible_coverage, sort_keys=True).encode()
    ).hexdigest()
    config_path = Path(command["config_path"])
    config_identifier = None
    if config_path.is_file():
        config_identifier = read_json(config_path).get("freqai", {}).get("identifier")
    subtrains = list(model_dir.glob("sub-train-*")) if model_dir.is_dir() else []
    exports = list(Path(command["output_dir"]).glob("*.zip"))
    errors: list[str] = []
    if len(pair_dates) != len(manifest["pairs"]):
        errors.append("prediction pair count differs from manifest")
    if not exact_coverage:
        errors.append("prediction timeline is not exact, hourly and gap-free")
    if missing_target_files:
        errors.append("one or more prediction files omit declared targets")
    if duplicate_column_files:
        errors.append("one or more prediction files contain duplicate columns")
    if missing_do_predict_files:
        errors.append("one or more prediction files omit do_predict")
    if prediction_null_cells:
        errors.append("prediction target cells contain nulls")
    if prediction_infinite_cells:
        errors.append("prediction target cells contain infinities")
    if len(subtrains) != len(files):
        errors.append("trained-submodel and prediction-file counts differ")
    if config_identifier != command["identifier"]:
        errors.append("profile config identifier differs from command manifest")
    if len(exports) != 1:
        errors.append("completed profile does not have exactly one exported archive")
    return {
        "profile_id": command["profile_id"],
        "identifier": command["identifier"],
        "model_dir": str(model_dir),
        "trained_submodels": len(subtrains),
        "prediction_files": len(files),
        "prediction_pairs": len(pair_dates),
        "expected_pairs": len(manifest["pairs"]),
        "expected_rows_per_pair": expected_rows,
        "exact_gap_free_coverage": exact_coverage,
        "missing_target_files": missing_target_files,
        "duplicate_column_files": duplicate_column_files,
        "missing_do_predict_files": missing_do_predict_files,
        "do_predict_values": sorted(do_predict_values),
        "eligible_mask_digest": eligible_mask_digest,
        "eligible_coverage": eligible_coverage,
        "prediction_null_cells": prediction_null_cells,
        "prediction_infinite_cells": prediction_infinite_cells,
        "config_identifier_matches": config_identifier == command["identifier"],
        "export_archives": len(exports),
        "coverage": coverage,
        "errors": errors,
    }


def batch_audit(
    generation_dir: Path,
    batch_record: dict[str, Any] | None,
    batch_id: str,
    spec: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    batch_dir = generation_dir / spec["directory"]
    freqai_dir = batch_dir / "freqai"
    manifest_path = freqai_dir / "manifest.json"
    audit: dict[str, Any] = {
        "batch": batch_id,
        "generation_status": None if batch_record is None else batch_record.get("status"),
        "freqai_manifest": str(manifest_path),
    }
    if batch_record is None:
        errors.append(f"{batch_id}: missing from generation manifest")
        return audit, errors
    status = batch_record.get("status")
    audit["generation_status_terminal"] = status in TERMINAL_STATES
    if not manifest_path.is_file():
        errors.append(f"{batch_id}: missing FreqAI manifest {manifest_path}")
        return audit, errors

    manifest = read_json(manifest_path)
    commands = manifest.get("commands", [])
    profiles = tuple(item.get("profile_id") for item in commands)
    expected = tuple(spec["profiles"])
    status_counts = Counter(str(item.get("status")) for item in commands)
    audit.update(
        {
            "run_id": manifest.get("run_id"),
            "expected_profiles": list(expected),
            "manifest_profiles": list(profiles),
            "profile_contract_matches": profiles == expected,
            "command_status_counts": dict(status_counts),
            "command_identifiers_unique": len(
                {item.get("identifier") for item in commands}
            )
            == len(commands),
        }
    )
    if profiles != expected:
        errors.append(f"{batch_id}: frozen profile contract differs")
    if not audit["command_identifiers_unique"]:
        errors.append(f"{batch_id}: duplicate command identifiers")

    commands_complete = bool(commands) and all(
        item.get("status") == "completed" and item.get("returncode") == 0
        for item in commands
    )
    audit["commands_complete"] = commands_complete
    output_audits = [
        model_output_audit(item, manifest)
        for item in commands
        if item.get("status") == "completed" and item.get("returncode") == 0
    ]
    audit["completed_profile_output_audits"] = output_audits
    for output_audit in output_audits:
        for error in output_audit["errors"]:
            errors.append(
                f"{batch_id}/{output_audit['profile_id']}: {error}"
            )
    eligible_mask_digests = {
        item["eligible_mask_digest"] for item in output_audits
    }
    audit["completed_profiles_share_eligible_prediction_mask"] = (
        len(eligible_mask_digests) <= 1
    )
    if len(eligible_mask_digests) > 1:
        errors.append(f"{batch_id}: completed profiles use different prediction masks")
    summary_path = freqai_dir / "run_summary.json"
    audit["run_summary_exists"] = summary_path.is_file()

    evidence_required = status in EVIDENCE_STATES or commands_complete
    score_path = freqai_dir / spec["scores"]
    portability_path = freqai_dir / spec["portability"]
    audit["scores_exists"] = score_path.is_file()
    audit["portability_exists"] = portability_path.is_file()
    if evidence_required:
        for artifact in (summary_path, score_path, portability_path):
            if not artifact.is_file():
                errors.append(f"{batch_id}: required completed-run artifact missing: {artifact}")
        for extra_spec in spec.get("extra_artifacts", ()):
            name = extra_spec if isinstance(extra_spec, str) else extra_spec["name"]
            artifact = freqai_dir / name
            if not artifact.is_file():
                errors.append(f"{batch_id}: required extra artifact missing: {artifact}")

    if score_path.is_file():
        audit["score_audit"] = score_audit(score_path, expected)
        score_info = audit["score_audit"]
        if score_info["missing_required_columns"]:
            errors.append(f"{batch_id}: score columns missing")
        if score_info["duplicate_key_rows"]:
            errors.append(f"{batch_id}: duplicate score keys")
        if score_info["infinite_numeric_cells"]:
            errors.append(f"{batch_id}: infinite score cells")
        if not score_info.get("profile_contract_matches", False):
            errors.append(f"{batch_id}: score profile contract differs")
        if not score_info.get("all_profiles_share_keys", False):
            errors.append(f"{batch_id}: score profiles do not share comparison keys")
        missing_comparisons = comparison_columns_missing(
            score_path, tuple(spec["comparison_prefixes"])
        )
        audit["missing_comparison_columns"] = missing_comparisons
        if missing_comparisons:
            errors.append(f"{batch_id}: comparison columns missing")

    if portability_path.is_file():
        audit["portability_audit"] = frame_audit(
            portability_path,
            list(spec["portability_required"]),
            None,
        )
        portability_info = audit["portability_audit"]
        if portability_info["missing_required_columns"]:
            errors.append(f"{batch_id}: portability columns missing")
        if portability_info["duplicate_rows"]:
            errors.append(f"{batch_id}: duplicate portability rows")
        if portability_info["infinite_numeric_cells"]:
            errors.append(f"{batch_id}: infinite portability cells")

    extra_audits: list[dict[str, Any]] = []
    for extra_spec in spec.get("extra_artifacts", ()):
        if isinstance(extra_spec, str):
            extra_spec = {"name": extra_spec, "required": [], "key": None}
        artifact = freqai_dir / extra_spec["name"]
        if not artifact.is_file():
            continue
        extra_audit = frame_audit(
            artifact, list(extra_spec["required"]), extra_spec.get("key")
        )
        extra_audits.append(extra_audit)
        if evidence_required and extra_audit["missing_required_columns"]:
            errors.append(f"{batch_id}: {extra_spec['name']} columns missing")
        if evidence_required and extra_audit["duplicate_key_rows"]:
            errors.append(f"{batch_id}: {extra_spec['name']} has duplicate keys")
        if evidence_required and extra_audit["infinite_numeric_cells"]:
            errors.append(f"{batch_id}: {extra_spec['name']} has infinite cells")
    audit["extra_artifact_audits"] = extra_audits

    if batch_id == "G1-B4":
        cache_audit_path = freqai_dir.parent / "cache_audit.json"
        contract_path = freqai_dir.parent / "feature_contract.json"
        audit["cache_audit_exists"] = cache_audit_path.is_file()
        audit["feature_contract_exists"] = contract_path.is_file()
        if evidence_required and not cache_audit_path.is_file():
            errors.append(f"{batch_id}: cache audit missing")
        if evidence_required and not contract_path.is_file():
            errors.append(f"{batch_id}: feature contract missing")
        if cache_audit_path.is_file():
            cache = read_json(cache_audit_path)
            path_order = cache.get("path_order_audit", {})
            metric_adequacy = cache.get("direct_metric_adequacy", {})
            path_metric = metric_adequacy.get("favourable_before_adverse", {})
            audit["path_order_audit"] = path_order
            audit["direct_metric_adequacy"] = metric_adequacy
            if evidence_required:
                required_metrics = {
                    "hold_instead_delta",
                    "missed_additional_profit",
                    "avoided_loss_after_exit",
                    "net_exit_regret",
                    "favourable_peak_step",
                    "adverse_peak_step",
                    "favourable_before_adverse",
                }
                if set(metric_adequacy) != required_metrics:
                    errors.append(f"{batch_id}: direct metric-adequacy surface differs")
                if path_order.get("same_candle_indeterminate_rows", 0) <= 0:
                    errors.append(f"{batch_id}: no indeterminate path-order ties audited")
                if path_order.get("strict_order_rows", 0) <= 0:
                    errors.append(f"{batch_id}: no strict path-order rows audited")
                for key in (
                    "same_candle_rows_with_binary_label",
                    "incomplete_order_rows_with_binary_label",
                    "strict_order_rows_without_binary_label",
                    "nonbinary_observed_labels",
                ):
                    if path_order.get(key) != 0:
                        errors.append(f"{batch_id}: invalid path-order audit value for {key}")
                raw_cells = path_metric.get("sequence_horizon_raw_adequate_cells")
                metric_cells = path_metric.get(
                    "sequence_horizon_metric_adequate_cells"
                )
                if raw_cells is None or metric_cells is None:
                    errors.append(f"{batch_id}: direct path-order adequacy audit missing")
                elif not 0 <= metric_cells <= raw_cells:
                    errors.append(f"{batch_id}: direct path-order adequacy counts invalid")
        if contract_path.is_file():
            contract = read_json(contract_path)
            audit["direct_metric_adequacy_rule_exists"] = bool(
                contract.get("direct_metric_adequacy_rule")
            )
            if evidence_required and not audit["direct_metric_adequacy_rule_exists"]:
                errors.append(f"{batch_id}: direct metric-adequacy contract missing")

    audit["errors"] = errors
    return audit, errors


def build_review(generation_dir: Path) -> dict[str, Any]:
    generation_manifest_path = generation_dir / "generation_manifest.json"
    manifest = read_json(generation_manifest_path)
    records = {item["id"]: item for item in manifest.get("batches", [])}
    audits: list[dict[str, Any]] = []
    errors: list[str] = []
    for batch_id, spec in BATCH_SPECS.items():
        audit, batch_errors = batch_audit(
            generation_dir, records.get(batch_id), batch_id, spec
        )
        audits.append(audit)
        errors.extend(batch_errors)
    nonterminal = {
        batch_id: None if records.get(batch_id) is None else records[batch_id].get("status")
        for batch_id in BATCH_SPECS
        if records.get(batch_id, {}).get("status") not in TERMINAL_STATES
    }
    evidence_ready = all(
        audit.get("commands_complete")
        and audit.get("run_summary_exists")
        and audit.get("scores_exists")
        and audit.get("portability_exists")
        and not audit.get("errors")
        for audit in audits
    )
    gate_open = not nonterminal and not errors
    return {
        "schema_version": 1,
        "generated_at": now_iso(),
        "generation": 1,
        "status": "review_gate_open" if gate_open else "review_gate_closed",
        "generation_manifest": str(generation_manifest_path),
        "all_batch_evidence_ready": evidence_ready,
        "nonterminal_batches": nonterminal,
        "integrity_error_count": len(errors),
        "integrity_errors": errors,
        "batch_audits": audits,
        "registered_generation2_branch_candidates": manifest.get(
            "registered_generation2_branch_candidates", []
        ),
        "interpretation_contract": {
            "association_not_causation": True,
            "activity_is_not_direction": True,
            "no_branch_execution_before_joint_review": True,
            "no_automatic_generation2_approval": True,
            "required_next_step_when_gate_opens": (
                "Interpret all four portability surfaces together against their frozen "
                "controls, then record terminal batch verdicts and the joint Generation-1 "
                "review before freezing any Generation-2 batch."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generation_dir", type=Path)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    generation_dir = args.generation_dir.resolve()
    review = build_review(generation_dir)
    if args.preflight:
        print(json.dumps(review, indent=2, allow_nan=False))
        return 0
    if review["status"] != "review_gate_open":
        print(json.dumps(review, indent=2, allow_nan=False))
        return 2
    output = (
        args.output.resolve()
        if args.output
        else generation_dir / "generation1_review.json"
    )
    atomic_json(review, output)
    print(json.dumps({"status": "written", "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
