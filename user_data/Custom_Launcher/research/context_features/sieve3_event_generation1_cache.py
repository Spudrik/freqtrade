"""Build timestamp-safe targeted event caches and retrospective scoring scopes for G1.

Precursor rows are written only to the scoring-scope artifact.  They are never
written as model features because identifying them requires knowledge of a later
event and would leak the future.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame


TARGET_ENTRY = "mtf_confluence_d1_vp_bos_4h_retest_short"
REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_HANDOVER = (
    REPO_ROOT
    / "ai_guidance_docs"
    / "03_status"
    / "sieve3_v2_freqai_resolved_params_handover_20260807.md"
)
DEFAULT_CONFIG = (
    REPO_ROOT
    / "user_data"
    / "configs"
    / "config_sieve3_event_reaction_freqai.example.json"
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat()


def pair_token(pair: str) -> str:
    return pair.split("/")[0].split(":")[0].lower()


def timeframe_hours(value: str) -> int:
    token = str(value).strip().lower()
    if token.endswith("h"):
        return int(token[:-1])
    if token.endswith("d"):
        return int(token[:-1]) * 24
    raise ValueError(f"Unsupported source timeframe: {value!r}")


def atomic_json(payload: dict[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_parquet(frame: DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary, index=False)
    temporary.replace(path)


def entry_mapping(portability_path: Path) -> DataFrame:
    mapping = pd.read_csv(
        portability_path,
        usecols=["entry_id", "family", "side", "entry_timeframe"],
    ).drop_duplicates()
    duplicates = mapping.duplicated("entry_id", keep=False)
    if duplicates.any():
        conflicting = (
            mapping.loc[duplicates]
            .groupby("entry_id")[["family", "side", "entry_timeframe"]]
            .nunique()
        )
        conflicting = conflicting[(conflicting > 1).any(axis=1)]
        if not conflicting.empty:
            raise ValueError(f"Conflicting entry metadata: {conflicting.index.tolist()}")
        mapping = mapping.drop_duplicates("entry_id")
    mapping["timeframe_hours"] = mapping["entry_timeframe"].map(timeframe_hours)
    return mapping


def episode_members(events: DataFrame) -> DataFrame:
    """Assign family-side episodes without changing or discarding exact onsets."""
    output: list[DataFrame] = []
    grouping = ["pair", "family", "side"]
    for keys, group in events.groupby(grouping, dropna=False, observed=True):
        work = group.sort_values(["decision_time", "timeframe_hours", "entry_id"]).copy()
        unique_times = (
            work.groupby("decision_time", as_index=False)
            .agg(timeframe_hours=("timeframe_hours", "max"))
            .sort_values("decision_time")
        )
        episode_numbers: list[int] = []
        episode = 0
        previous_time: pd.Timestamp | None = None
        previous_hours: int | None = None
        for row in unique_times.itertuples(index=False):
            current_time = row.decision_time
            current_hours = int(row.timeframe_hours)
            if previous_time is not None:
                gap_hours = (current_time - previous_time).total_seconds() / 3600.0
                if gap_hours > max(int(previous_hours or 1), current_hours):
                    episode += 1
            episode_numbers.append(episode)
            previous_time = current_time
            previous_hours = current_hours
        unique_times["episode_number"] = episode_numbers
        unique_times["episode_start"] = unique_times.groupby("episode_number")[
            "decision_time"
        ].transform("min")
        work = work.merge(
            unique_times[["decision_time", "episode_number", "episode_start"]],
            on="decision_time",
            how="left",
            validate="many_to_one",
        )
        prefix = "|".join(str(item) for item in keys)
        work["episode_id"] = work["episode_number"].map(
            lambda number: f"{prefix}|{int(number)}"
        )
        work["is_episode_start"] = work["decision_time"].eq(work["episode_start"])
        output.append(work.drop(columns="episode_number"))
    return pd.concat(output, ignore_index=True) if output else DataFrame()


def scoring_scopes(events: DataFrame) -> DataFrame:
    columns = [
        "pair",
        "family",
        "side",
        "entry_id",
        "entry_timeframe",
        "timeframe_hours",
        "episode_id",
        "episode_start",
        "is_episode_start",
    ]
    records: list[DataFrame] = []
    exact = events[columns].copy()
    exact["event_decision_time"] = events["decision_time"].to_numpy()
    exact["decision_time"] = exact["event_decision_time"]
    exact["scope_type"] = "exact_onset"
    exact["future_derived_scope_only"] = False
    records.append(exact)
    for multiple in (1, 2):
        precursor = events[columns].copy()
        precursor["event_decision_time"] = events["decision_time"].to_numpy()
        precursor["decision_time"] = precursor["event_decision_time"] - pd.to_timedelta(
            precursor["timeframe_hours"] * multiple, unit="h"
        )
        precursor["scope_type"] = f"precursor_{multiple}x_native"
        precursor["future_derived_scope_only"] = True
        records.append(precursor)
    scopes = pd.concat(records, ignore_index=True)
    scopes["decision_time"] = pd.to_datetime(scopes["decision_time"], utc=True)
    scopes["event_decision_time"] = pd.to_datetime(
        scopes["event_decision_time"], utc=True
    )
    return scopes.sort_values(
        ["pair", "family", "side", "decision_time", "scope_type", "entry_id"]
    ).reset_index(drop=True)


def targeted_cache(
    source: DataFrame,
    mapping: DataFrame,
    selected_groups: set[tuple[str, str]],
    selected_entries: set[str],
) -> tuple[DataFrame, dict[str, Any]]:
    output = source[["decision_time", "coverage_present"]].copy()
    selected_mapping = mapping[
        mapping.apply(lambda row: (row["family"], row["side"]) in selected_groups, axis=1)
    ].copy()
    audit_groups: dict[str, Any] = {}
    for (family, side), group in selected_mapping.groupby(
        ["family", "side"], observed=True
    ):
        entries = sorted(group["entry_id"].astype(str).unique())
        onset_columns = [
            f"entry_onset__{entry}" for entry in entries if f"entry_onset__{entry}" in source
        ]
        active_columns = [
            f"entry_active__{entry}" for entry in entries if f"entry_active__{entry}" in source
        ]
        if not onset_columns or len(onset_columns) != len(active_columns):
            raise ValueError(
                f"Incomplete targeted cache columns for {family}:{side}: "
                f"onsets={len(onset_columns)} active={len(active_columns)} entries={len(entries)}"
            )
        token = f"g1_family_{family}_{side}"
        output[f"entry_onset__{token}"] = (
            source[onset_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
        )
        output[f"entry_active__{token}"] = (
            source[active_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
        )
        audit_groups[f"{family}:{side}"] = {
            "entries": entries,
            "entry_count": len(entries),
            "onset_rows": int(output[f"entry_onset__{token}"].gt(0).sum()),
            "active_rows": int(output[f"entry_active__{token}"].gt(0).sum()),
        }
    for entry in sorted(selected_entries):
        onset = f"entry_onset__{entry}"
        active = f"entry_active__{entry}"
        if onset not in source or active not in source:
            raise ValueError(f"Target entry cache columns are missing: {entry}")
        output[onset] = pd.to_numeric(source[onset], errors="coerce").fillna(0.0)
        output[active] = pd.to_numeric(source[active], errors="coerce").fillna(0.0)
    onset_features = [column for column in output if column.startswith("entry_onset__")]
    output["summary__g1_selected_onset_count"] = output[onset_features].sum(axis=1)
    return output, audit_groups


def build_batch(
    batch: str,
    source_cache_dir: Path,
    output_root: Path,
    mapping: DataFrame,
    long_events: DataFrame,
    selected_groups: set[tuple[str, str]],
    selected_entries: set[str],
    pairs: list[str],
) -> dict[str, Any]:
    batch_dir = output_root / batch.lower()
    cache_dir = batch_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    pair_audits: list[dict[str, Any]] = []
    for pair in pairs:
        source_path = source_cache_dir / f"{pair_token(pair)}_sieve3_events_1h.parquet"
        source = pd.read_parquet(source_path)
        source["decision_time"] = pd.to_datetime(source["decision_time"], utc=True)
        if source["decision_time"].duplicated().any():
            raise ValueError(f"Duplicate source decision times: {pair}")
        output, groups = targeted_cache(
            source, mapping, selected_groups, selected_entries
        )
        output_path = cache_dir / source_path.name
        atomic_parquet(output, output_path)
        feature_columns = [
            column
            for column in output
            if column.startswith("entry_onset__")
            or column.startswith("entry_active__")
            or column.startswith("summary__")
        ]
        precursor_features = [
            column for column in feature_columns if "precursor" in column.lower()
        ]
        if precursor_features:
            raise ValueError(
                f"Future-derived precursor features are forbidden: {precursor_features}"
            )
        pair_audits.append(
            {
                "pair": pair,
                "rows": len(output),
                "start": output["decision_time"].min().isoformat(),
                "end": output["decision_time"].max().isoformat(),
                "duplicate_decision_times": int(output["decision_time"].duplicated().sum()),
                "coverage_missing": int(
                    (~pd.to_numeric(output["coverage_present"], errors="coerce").eq(1.0)).sum()
                ),
                "feature_columns": len(feature_columns),
                "groups": groups,
            }
        )
    eligible = long_events[
        long_events["pair"].isin(pairs)
        & long_events["is_onset"].astype(bool)
    ].merge(mapping, on=["entry_id", "side"], how="left", validate="many_to_one")
    eligible = eligible[
        eligible.apply(
            lambda row: (row["family"], row["side"]) in selected_groups
            or row["entry_id"] in selected_entries,
            axis=1,
        )
    ].copy()
    eligible["decision_time"] = pd.to_datetime(eligible["decision_time"], utc=True)
    eligible = eligible.dropna(
        subset=["decision_time", "family", "entry_timeframe", "timeframe_hours"]
    )
    episodes = episode_members(eligible)
    scopes = scoring_scopes(episodes)
    minimum_cache_time = min(pd.Timestamp(item["start"]) for item in pair_audits)
    scopes = scopes[scopes["decision_time"].ge(minimum_cache_time)].copy()
    scope_path = batch_dir / "event_scopes.parquet"
    atomic_parquet(scopes, scope_path)
    audit = {
        "generated_at": now_iso(),
        "batch": batch,
        "source_cache_dir": str(source_cache_dir),
        "cache_dir": str(cache_dir),
        "pairs": pair_audits,
        "event_scope_rows": int(len(scopes)),
        "exact_scope_rows": int(scopes["scope_type"].eq("exact_onset").sum()),
        "precursor_scope_rows": int(scopes["scope_type"].str.startswith("precursor_").sum()),
        "future_derived_model_feature_count": 0,
        "scope_path": str(scope_path),
        "status": "passed",
    }
    atomic_json(audit, batch_dir / "cache_audit.json")
    return audit


def _boolean_column(frame: DataFrame, column: str) -> pd.Series:
    if column not in frame:
        raise ValueError(f"Required component column is missing: {column}")
    return pd.Series(frame[column], index=frame.index).astype("boolean").fillna(False)


def add_b2_component_features(
    *,
    handover: Path,
    config_path: Path,
    timerange: str,
    pairs: list[str],
    batch_dir: Path,
) -> dict[str, Any]:
    """Reconstruct the named event's observable D1/H4 gates at decision time."""
    from freqtrade.commands.optimize_commands import setup_optimize_configuration
    from freqtrade.enums import RunMode
    from freqtrade.optimize.backtesting import Backtesting
    from sieve3_event_reaction_research import (
        apply_locked_parameters,
        parameter_mismatches,
        parse_handover,
        register_archived_entry_sieve_tools,
        strategy_arguments,
    )

    candidates = [
        candidate
        for candidate in parse_handover(handover)
        if candidate.entry_id == TARGET_ENTRY
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one {TARGET_ENTRY} handover candidate, found {len(candidates)}"
        )
    candidate = candidates[0]
    register_archived_entry_sieve_tools()
    catalogue_config = json.loads(config_path.read_text(encoding="utf-8"))
    catalogue_config.setdefault("freqai", {})["enabled"] = False
    catalogue_config_path = batch_dir / "component_catalogue_config.json"
    atomic_json(catalogue_config, catalogue_config_path)
    config = setup_optimize_configuration(
        strategy_arguments(
            candidate,
            config=catalogue_config_path,
            timerange=timerange,
            pairs=tuple(pairs),
        ),
        RunMode.BACKTEST,
    )
    config.setdefault("freqai", {})["enabled"] = False
    backtesting = Backtesting(config)
    audits: list[dict[str, Any]] = []
    component_columns = [
        "component__d1_vp_gate",
        "component__d1_structure_bos_gate",
        "component__d1_prior_level_gate",
        "component__h4_retest_gate",
        "component__h4_structure_gate",
        "component__h4_context_gate",
        "component__d1_gate_count",
        "component__h4_gate_count",
        "component__all_gate_count",
        "component__d1_confluence_pass",
        "component__h4_execution_pass",
        "component__vp_bos_joint",
    ]
    try:
        backtesting._set_strategy(backtesting.strategylist[0])
        strategy = backtesting.strategy
        if strategy.__class__.__name__ != candidate.source_class:
            raise ValueError(
                f"Resolved class {strategy.__class__.__name__} != {candidate.source_class}"
            )
        applied = apply_locked_parameters(strategy, candidate.locked_values)
        mismatches = parameter_mismatches(strategy, candidate.locked_values)
        if mismatches:
            raise ValueError(f"Locked B2 component parameters differ: {mismatches[:8]}")
        data, _ = backtesting.load_bt_data()
        processed = strategy.advise_all_indicators(data)
        native_hours = timeframe_hours(candidate.timeframe)
        min_htf = int(strategy.min_htf_gates.value)
        min_ltf = int(strategy.min_ltf_gates.value)
        for pair in pairs:
            signals = strategy.advise_entry(processed[pair].copy(), {"pair": pair})
            htf_gates = [
                _boolean_column(signals, "mtf_htf_vp_gate"),
                _boolean_column(signals, "mtf_htf_structure_gate"),
                _boolean_column(signals, "mtf_htf_level_gate"),
            ]
            ltf_gates = [gate.astype(bool) for gate in strategy._ltf_gates(signals)]
            if len(ltf_gates) != 3:
                raise ValueError(f"Expected three H4 execution gates for {pair}")
            htf_count = sum(gate.astype("int8") for gate in htf_gates)
            ltf_count = sum(gate.astype("int8") for gate in ltf_gates)
            components = DataFrame(
                {
                    "decision_time": pd.to_datetime(
                        signals["date"], utc=True, errors="coerce"
                    )
                    + pd.Timedelta(hours=native_hours),
                    "component_source_time": pd.to_datetime(
                        signals["date"], utc=True, errors="coerce"
                    )
                    + pd.Timedelta(hours=native_hours),
                    "component__d1_vp_gate": htf_gates[0].astype(float),
                    "component__d1_structure_bos_gate": htf_gates[1].astype(float),
                    "component__d1_prior_level_gate": htf_gates[2].astype(float),
                    "component__h4_retest_gate": ltf_gates[0].astype(float),
                    "component__h4_structure_gate": ltf_gates[1].astype(float),
                    "component__h4_context_gate": ltf_gates[2].astype(float),
                    "component__d1_gate_count": htf_count.astype(float),
                    "component__h4_gate_count": ltf_count.astype(float),
                    "component__all_gate_count": (htf_count + ltf_count).astype(float),
                    "component__d1_confluence_pass": htf_count.ge(min_htf).astype(float),
                    "component__h4_execution_pass": ltf_count.ge(min_ltf).astype(float),
                    "component__vp_bos_joint": (htf_gates[0] & htf_gates[1]).astype(float),
                    "reconstructed_entry_active": pd.to_numeric(
                        signals.get("enter_short", 0.0), errors="coerce"
                    ).fillna(0.0).gt(0.0).astype(float),
                }
            ).dropna(subset=["decision_time"])
            components = components.sort_values("decision_time").drop_duplicates(
                "decision_time", keep="last"
            )
            cache_path = batch_dir / "cache" / f"{pair_token(pair)}_sieve3_events_1h.parquet"
            hourly = pd.read_parquet(cache_path)
            hourly["decision_time"] = pd.to_datetime(
                hourly["decision_time"], utc=True, errors="coerce"
            )
            hourly = hourly.sort_values("decision_time")
            exact = hourly[["decision_time", f"entry_active__{TARGET_ENTRY}"]].merge(
                components[["decision_time", "reconstructed_entry_active"]],
                on="decision_time",
                how="left",
                validate="one_to_one",
            )
            expected = pd.to_numeric(
                exact[f"entry_active__{TARGET_ENTRY}"], errors="coerce"
            ).fillna(0.0)
            reconstructed = pd.to_numeric(
                exact["reconstructed_entry_active"], errors="coerce"
            ).fillna(0.0)
            reproduction_mismatches = int(expected.ne(reconstructed).sum())
            if reproduction_mismatches:
                raise ValueError(
                    f"B2 exact event reconstruction differs for {pair}: "
                    f"{reproduction_mismatches} hourly rows"
                )
            aligned = pd.merge_asof(
                hourly,
                components[["component_source_time", *component_columns]],
                left_on="decision_time",
                right_on="component_source_time",
                direction="backward",
                tolerance=pd.Timedelta(hours=native_hours - 1),
            )
            component_age = (
                aligned["decision_time"] - aligned["component_source_time"]
            ).dt.total_seconds() / 3600.0
            missing_components = int(aligned[component_columns].isna().any(axis=1).sum())
            if missing_components:
                raise ValueError(
                    f"B2 component cache has {missing_components} uncovered hourly rows for {pair}"
                )
            if not component_age.between(0.0, float(native_hours - 1)).all():
                raise ValueError(f"B2 component cache has invalid source ages for {pair}")
            aligned["component__source_age_hours"] = component_age.astype(float)
            aligned = aligned.drop(columns="component_source_time")
            atomic_parquet(aligned, cache_path)
            audits.append(
                {
                    "pair": pair,
                    "rows": int(len(aligned)),
                    "component_columns": len(component_columns) + 1,
                    "source_age_min_hours": float(component_age.min()),
                    "source_age_max_hours": float(component_age.max()),
                    "reproduction_mismatches": reproduction_mismatches,
                    "missing_component_rows": missing_components,
                }
            )
        audit = {
            "generated_at": now_iso(),
            "batch": "G1-B2",
            "entry_id": TARGET_ENTRY,
            "source_path": str(candidate.source_path),
            "source_class": candidate.source_class,
            "catalogue_config": str(catalogue_config_path),
            "locked_parameters_applied": applied,
            "locked_parameters_checked": len(candidate.locked_values),
            "component_semantics": {
                "D1": "three observable higher-timeframe VP, BOS/structure and prior-level gates",
                "H4": "three observable retest/location, structure and context/pressure gates",
                "holding_rule": "Each completed 4h component state is available for the next four 1h decision rows; source age must be 0-3 hours.",
            },
            "pairs": audits,
            "future_derived_model_feature_count": 0,
            "status": "passed",
        }
        atomic_json(audit, batch_dir / "component_cache_audit.json")
        return audit
    finally:
        try:
            backtesting.exchange.close()
        finally:
            Backtesting.cleanup()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generation1_manifest", type=Path)
    parser.add_argument("generation0_base", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--handover", type=Path, default=DEFAULT_HANDOVER)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--timerange", default="20221201-20260629")
    args = parser.parse_args()
    manifest = json.loads(args.generation1_manifest.read_text(encoding="utf-8"))
    generation0_base = args.generation0_base.resolve()
    source_cache_dir = generation0_base / "cache"
    mapping = entry_mapping(generation0_base / "direct_entry_portability_summary.csv")
    long_events = pd.read_parquet(source_cache_dir / "entry_events_long.parquet")
    b1 = next(item for item in manifest["batches"] if item["id"] == "G1-B1")
    b1_groups = {
        tuple(token.rsplit(":", 1)) for token in b1["eligible_family_side_groups"]
    }
    pairs = list(manifest["universe"]["pairs"])
    results = {
        "G1-B1": build_batch(
            "G1-B1",
            source_cache_dir,
            args.output_root.resolve(),
            mapping,
            long_events,
            b1_groups,
            set(),
            pairs,
        ),
        "G1-B2": build_batch(
            "G1-B2",
            source_cache_dir,
            args.output_root.resolve(),
            mapping,
            long_events,
            set(),
            {TARGET_ENTRY},
            pairs,
        ),
    }
    results["G1-B2"]["component_cache"] = add_b2_component_features(
        handover=args.handover.resolve(),
        config_path=args.config.resolve(),
        timerange=args.timerange,
        pairs=pairs,
        batch_dir=args.output_root.resolve() / "g1-b2",
    )
    print(
        json.dumps(
            {
                key: {
                    "event_scope_rows": value["event_scope_rows"],
                    "exact_scope_rows": value["exact_scope_rows"],
                    "precursor_scope_rows": value["precursor_scope_rows"],
                    "status": value["status"],
                }
                for key, value in results.items()
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
