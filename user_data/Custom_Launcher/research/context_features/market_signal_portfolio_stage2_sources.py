"""Assemble outcome-blind, decision-time rows from the frozen portfolio sources.

This is an input ledger, not a trading score.  Local level maps and G17 contacts
are separate materialization lanes and are deliberately not inferred here.
"""

from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import market_signal_portfolio_stage1_preflight as stage1
import pandas as pd


ROOT = stage1.REPO_ROOT
OUTPUT = (
    ROOT
    / "user_data/research_news_data/context_features/event_hierarchy"
    / "market_signal_portfolio_stage2_20260924a"
)
ROWS_PATH = OUTPUT / "source_decision_rows.parquet"
RESULT_PATH = OUTPUT / "source_assembly_result.json"

SCHEMA = [
    "prototype_id",
    "family_id",
    "source_sibling",
    "source_episode_id",
    "parent_episode_ids_json",
    "sample_kind",
    "partition",
    "pair",
    "cohort",
    "horizon",
    "anchor_utc",
    "decision_utc",
    "latest_input_utc",
    "call_state",
    "no_call_reason",
    "signal_direction",
    "signal_value",
    "source_contract",
]


def _iso(value: object) -> str:
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("Missing source timestamp")
    if timestamp.tzinfo is None:
        raise ValueError(f"Source timestamp has no timezone: {value}")
    return timestamp.tz_convert("UTC").isoformat().replace("+00:00", "Z")


def _row(prototype: str, **values: object) -> dict[str, object]:
    meta = next(p for p in stage1.PROTOTYPES if p["prototype_id"] == prototype)
    row: dict[str, object] = {key: "" for key in SCHEMA}
    row.update(prototype_id=prototype, family_id=meta["family_id"])
    row.update(values)
    if row["latest_input_utc"] and pd.Timestamp(row["latest_input_utc"]) > pd.Timestamp(
        row["decision_utc"]
    ):
        raise ValueError(f"Future source input for {prototype}")
    if row["call_state"] not in ("issued", "no_call"):
        raise ValueError(f"Unknown call state for {prototype}: {row['call_state']}")
    return row


def _source_map() -> dict[str, dict[str, str]]:
    result = stage1.validate_existing_result()
    if result["status"] != "stage1_outcome_blind_coverage_preflight_complete":
        raise ValueError("Stage-1 preflight is not complete")
    freeze = json.loads(stage1.FREEZE_PATH.read_text(encoding="utf-8"))
    wanted = {
        "macro_direction_calls",
        "layer2_events",
        "independent_events",
        "layer2_background",
        "layer2_controls",
        "simple_signal_calls",
        "meme_transmission_groups",
    }
    selected = [item for item in freeze["source_artifact_contracts"] if item["name"] in wanted]
    names = [item["name"] for item in selected]
    if set(names) != wanted or len(names) != len(set(names)):
        raise ValueError("Duplicate source contract name; cannot unambiguously assemble")
    return {item["name"]: item for item in selected}


def _path(sources: dict[str, dict[str, str]], name: str) -> Path:
    return stage1._verify_contract(name, sources[name])


def assemble(sources: dict[str, dict[str, str]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    macro = pd.read_parquet(
        _path(sources, "macro_direction_calls"),
        columns=[
            "route_id",
            "expectation_episode_id",
            "source_block",
            "official_event_id",
            "anchor_utc",
            "pair",
            "horizon_minutes",
            "predicted_direction",
            "abstention_reason",
        ],
    )
    for item in macro.loc[macro.route_id.eq("cpi_headline_core_agreement")].itertuples(index=False):
        anchor = _iso(item.anchor_utc)
        reason = str(item.abstention_reason) if pd.notna(item.abstention_reason) else ""
        rows.append(
            _row(
                "cpi_headline_core_agreement_direction",
                source_sibling="CPI_headline_core_agreement",
                source_episode_id=f"official:{item.official_event_id}",
                sample_kind="event",
                partition=item.source_block,
                pair=item.pair,
                horizon=f"{item.horizon_minutes}m",
                anchor_utc=anchor,
                decision_utc=anchor,
                latest_input_utc=anchor,
                call_state="no_call" if reason else "issued",
                no_call_reason=reason,
                signal_direction=int(item.predicted_direction) if not reason else "",
                source_contract="macro_direction_calls",
            )
        )

    events = pd.read_csv(
        _path(sources, "layer2_events"),
        usecols=["event_id", "event_source", "anchor_utc", "whole_event_partition"],
    )
    event_clock = dict(zip(events.event_id, events.anchor_utc, strict=True))
    for item in events.loc[events.event_source.eq("official_fomc")].itertuples(index=False):
        anchor = _iso(item.anchor_utc)
        rows.append(
            _row(
                "fomc_event_activity_warning",
                source_sibling="FOMC_official_clock",
                source_episode_id=f"official:{item.event_id}",
                sample_kind="event",
                partition=item.whole_event_partition,
                anchor_utc=anchor,
                decision_utc=anchor,
                latest_input_utc=anchor,
                call_state="issued",
                signal_value="official_release",
                source_contract="layer2_events",
            )
        )

    independent = pd.read_csv(
        _path(sources, "independent_events"),
        usecols=[
            "family",
            "available_at",
            "anchor_utc",
            "whole_event_partition",
            "source_direction",
            "predicted_direction",
            "is_event",
            "event_id",
            "coverage_ready",
        ],
    )
    for item in independent.loc[independent.is_event.astype(bool)].itertuples(index=False):
        if item.family not in (
            "live_news_activity_spike",
            "live_web_activity_spike",
            "btc_dominance_change",
        ):
            continue
        prototype = (
            "btc_dominance_relative_meme_rotation"
            if item.family == "btc_dominance_change"
            else "live_news_web_activity_warning"
        )
        anchor, available = _iso(item.anchor_utc), _iso(item.available_at)
        ready = bool(item.coverage_ready) and pd.Timestamp(available) <= pd.Timestamp(anchor)
        rows.append(
            _row(
                prototype,
                source_sibling=item.family,
                source_episode_id=f"context:{item.event_id}",
                sample_kind="event",
                partition=item.whole_event_partition,
                anchor_utc=anchor,
                decision_utc=anchor,
                latest_input_utc=available if ready else "",
                call_state="issued" if ready else "no_call",
                no_call_reason="" if ready else "source_unready_or_late",
                signal_direction=int(item.predicted_direction)
                if ready
                and prototype.startswith("btc_dominance")
                and pd.notna(item.predicted_direction)
                else "",
                signal_value="activity_spike"
                if prototype.startswith("live_news")
                else "relative_rotation",
                source_contract="independent_events",
            )
        )

    background = pd.read_parquet(
        _path(sources, "layer2_background"),
        columns=[
            "sample_id",
            "event_id",
            "sample_type",
            "control_type",
            "control_rank",
            "pre_return_30d",
            "background",
            "positive_initial_move",
        ],
    )
    control_map = pd.read_csv(
        _path(sources, "layer2_controls"),
        usecols=["event_id", "control_type", "control_rank", "control_anchor_utc"],
    )
    control_clocks = {
        (item.event_id, item.control_type, int(item.control_rank)): item.control_anchor_utc
        for item in control_map.itertuples(index=False)
    }
    if len(control_clocks) != len(control_map):
        raise ValueError("Duplicate background control clock")
    for item in background.itertuples(index=False):
        sample_clock = (
            event_clock[item.event_id]
            if item.sample_type == "event"
            else control_clocks[(item.event_id, item.control_type, int(item.control_rank))]
        )
        anchor = _iso(sample_clock)
        decision = _iso(pd.Timestamp(anchor) + timedelta(hours=4))
        issued = item.background == "negative_30d_background" and bool(item.positive_initial_move)
        rows.append(
            _row(
                "negative_background_positive_event_fade",
                source_sibling="negative_30d_plus_positive_4h",
                source_episode_id=f"layer2:{item.sample_id}",
                sample_kind=item.sample_type,
                parent_episode_ids_json=json.dumps([f"layer2:{item.event_id}"]),
                anchor_utc=anchor,
                decision_utc=decision,
                latest_input_utc=decision,
                call_state="issued" if issued else "no_call",
                no_call_reason="" if issued else "condition_absent",
                signal_value="fade_watch" if issued else "",
                source_contract="layer2_background",
            )
        )

    volume = pd.read_parquet(
        _path(sources, "simple_signal_calls"),
        columns=[
            "sample_kind",
            "analysis_unit_id",
            "model_anchor_utc",
            "model_period",
            "parent_episode_ids_json",
            "pair",
            "signal_id",
            "signal_issued",
            "latest_contributing_data_utc",
            "call_state",
        ],
    )
    for item in volume.loc[volume.signal_id.eq("recent_volume_persistence")].itertuples(
        index=False
    ):
        anchor, latest = _iso(item.model_anchor_utc), _iso(item.latest_contributing_data_utc)
        rows.append(
            _row(
                "recent_volume_persistence",
                source_sibling="recent_volume",
                source_episode_id=f"market_sample:{item.analysis_unit_id}",
                parent_episode_ids_json=item.parent_episode_ids_json,
                sample_kind=item.sample_kind,
                partition=item.model_period,
                pair=item.pair,
                anchor_utc=anchor,
                decision_utc=anchor,
                latest_input_utc=latest,
                call_state="issued" if bool(item.signal_issued) else "no_call",
                no_call_reason="" if bool(item.signal_issued) else str(item.call_state),
                signal_value="volume_already_high" if bool(item.signal_issued) else "",
                source_contract="simple_signal_calls",
            )
        )

    groups = pd.read_parquet(
        _path(sources, "meme_transmission_groups"),
        columns=[
            "sample_kind",
            "episode_id",
            "model_anchor_utc",
            "model_period",
            "event_families_json",
            "btc_confirmed_reaction",
            "horizon_hours",
            "cohort",
            "complete_cohort",
        ],
    )
    for item in groups.itertuples(index=False):
        anchor = _iso(item.model_anchor_utc)
        decision = _iso(pd.Timestamp(anchor) + timedelta(hours=1))
        issued = bool(item.btc_confirmed_reaction) and bool(item.complete_cohort)
        rows.append(
            _row(
                "confirmed_btc_to_group_activity",
                source_sibling="BTC_confirmed_then_group",
                source_episode_id=f"market_episode:{item.episode_id}",
                sample_kind=item.sample_kind,
                partition=item.model_period,
                cohort=item.cohort,
                horizon=f"{item.horizon_hours}h",
                anchor_utc=anchor,
                decision_utc=decision,
                latest_input_utc=decision,
                call_state="issued" if issued else "no_call",
                no_call_reason="" if issued else "btc_unconfirmed_or_cohort_incomplete",
                signal_value="leader_confirmed" if issued else "",
                source_contract="meme_transmission_groups",
            )
        )

    frame = pd.DataFrame.from_records(rows, columns=SCHEMA)
    key = [
        "prototype_id",
        "source_sibling",
        "source_episode_id",
        "sample_kind",
        "pair",
        "cohort",
        "horizon",
        "decision_utc",
    ]
    duplicates = frame.loc[frame.duplicated(key, keep=False), key]
    if frame.empty or not duplicates.empty:
        raise ValueError(
            f"Missing rows or ambiguous decision rows: {duplicates.head(4).to_dict('records')}"
        )
    frame["signal_direction"] = pd.to_numeric(frame["signal_direction"], errors="coerce").astype(
        "Int64"
    )
    return frame.sort_values(["decision_utc", "prototype_id", "source_episode_id"]).reset_index(
        drop=True
    )


def run() -> dict[str, object]:
    sources = _source_map()
    frame = assemble(sources)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(ROWS_PATH, index=False)
    counts = frame.groupby(["prototype_id", "call_state"], dropna=False).size().to_dict()
    result: dict[str, object] = {
        "status": "stage2_source_decisions_assembled_level_map_pending",
        "outcomes_read": False,
        "market_model_trained": False,
        "trade_rule_promoted": False,
        "decision_row_count": len(frame),
        "prototype_call_counts": {
            f"{key[0]}:{key[1]}": int(value) for key, value in counts.items()
        },
        "missing_lanes": [
            "calculated_area_contact_traffic",
            "local_multitimeframe_support_resistance_map",
        ],
        "rows_path": str(ROWS_PATH),
        "rows_sha256": stage1.g0.sha256_file(ROWS_PATH),
        "source_contracts": {
            name: {"path": contract["path"], "sha256": contract["sha256"]}
            for name, contract in sorted(sources.items())
        },
        "limitation": (
            "Source-native identities are preserved; different event catalogues are "
            "not merged by timestamp. No outcomes or local coordinates were joined."
        ),
    }
    RESULT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", required=True)
    parser.parse_args()
    print(json.dumps(run(), indent=2, sort_keys=True))
