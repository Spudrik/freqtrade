"""Freeze the supported Generation 25 siblings after the outcome-blind coverage gate."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
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
    market_reaction_zone_generation24_joint_review as g24j,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation25_coverage as g25c,
)


ANALYSIS_PATH = Path(__file__).resolve()
DEFAULT_BATCH_ID = "g25_broad_siblings_20260828a"
OUTPUT_ROOT = g25c.OUTPUT_ROOT
FREEZE_PATH = OUTPUT_ROOT.parent / "g25_broad_siblings_freeze_20260828a.json"
COVERAGE_RESULT = (
    g25c.RECORD_ROOT / g25c.DEFAULT_RUN_ID / "g25_coverage_result.json"
)


def artifact(path: Path) -> dict[str, Any]:
    return g24j.artifact(path)


def load_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    review = json.loads(g25c.PARENT_REVIEW.read_text(encoding="utf-8"))
    queue = json.loads(g25c.PARENT_QUEUE.read_text(encoding="utf-8"))
    coverage = json.loads(COVERAGE_RESULT.read_text(encoding="utf-8"))
    if review.get("status") != "completed_generation24_joint_review":
        raise ValueError("Generation 24 review is not terminal.")
    if queue.get("status") != "queued_after_complete_generation24_joint_review":
        raise ValueError("Generation 25 queue is invalid.")
    if coverage.get("status") != "completed_generation25_outcome_blind_coverage_gate":
        raise ValueError("Generation 25 coverage gate is not terminal.")
    if coverage.get("future_reaction_outcomes_read"):
        raise ValueError("Generation 25 coverage gate opened outcomes.")
    return review, queue, coverage


def branch_definitions(
    queue: dict[str, Any], coverage: dict[str, Any]
) -> list[dict[str, Any]]:
    gates = coverage["branch_gates"]
    output: list[dict[str, Any]] = []
    for item in queue["siblings"]:
        branch = dict(item)
        branch_id = str(branch["branch_id"])
        gate = gates.get(branch_id)
        if gate is None:
            branch["status_at_freeze"] = branch.pop("status")
            output.append(branch)
            continue
        branch["coverage_gate"] = gate
        branch.pop("status", None)
        branch["status_at_freeze"] = (
            "frozen_pending_support"
            if gate["advance"]
            else str(gate["status"])
        )
        output.append(branch)
    return output


def freeze(*, overwrite: bool = False) -> dict[str, Any]:
    review, queue, coverage = load_inputs()
    if FREEZE_PATH.is_file() and not overwrite:
        frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        if frozen.get("status") != "frozen_before_generation25_outcomes":
            raise ValueError("Invalid existing Generation 25 freeze.")
        return frozen
    branches = branch_definitions(queue, coverage)
    active = [item for item in branches if item["status_at_freeze"].startswith("frozen_")]
    coverage_parked = [
        item
        for item in branches
        if item["status_at_freeze"].startswith("parked_")
        and item.get("coverage_gate") is not None
    ]
    other_parked = [
        item
        for item in branches
        if item["status_at_freeze"].startswith("parked_")
        and item.get("coverage_gate") is None
    ]
    if len(active) != 3 or len(coverage_parked) != 2:
        raise ValueError("Generation 25 coverage decisions drifted.")
    frozen = {
        "schema_version": 1,
        "generation": 25,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation25_outcomes",
        "batch_id": DEFAULT_BATCH_ID,
        "all_supported_siblings_frozen_together": True,
        "route_portfolio_assessed": 5,
        "active_outcome_siblings": len(active),
        "coverage_parked_siblings": len(coverage_parked),
        "prior_evidence_parked_siblings": len(other_parked),
        "batch_rule": (
            "Build and freeze all three supported sibling surfaces before any Generation "
            "25 outcome. Jointly review all five routes, including the two terminal "
            "coverage findings, before any descendant."
        ),
        "same_holdout_boundary": (
            "Convergence, connected Volume Profile, and market-state attribution reuse "
            "known chronology and remain exploratory. Daily VWAP fresh confirmation and "
            "new external-source modelling stay parked until their coverage gates pass."
        ),
        "branches": branches,
        "research_boundary": {
            "profit_used": False,
            "signed_direction_used": False,
            "no_trading_promotion": True,
            "one_minute_direction_lane_open": False,
            "canonical_indicators_edited": False,
        },
        "source_contracts": {
            "generation24_joint_review": artifact(g25c.PARENT_REVIEW),
            "generation25_queue": artifact(g25c.PARENT_QUEUE),
            "generation25_coverage_gate": artifact(COVERAGE_RESULT),
            "analysis_script": artifact(ANALYSIS_PATH),
        },
        "parent_breadth_assessment": review["breadth_assessment"],
    }
    FREEZE_PATH.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(frozen, FREEZE_PATH)
    return frozen


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(freeze(overwrite=args.overwrite), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
