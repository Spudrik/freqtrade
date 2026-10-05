"""Continue the two supported frozen questions after an outcome-blind support failure.

The original three-route gate stays intact.  This separate record freezes the
support-only reason for parking daily VWAP before either surviving route reads a
future outcome.  It retains the original questions, periods, controls, and code.
"""

from __future__ import annotations

# Bound numerical pools before importing the research modules.
# ruff: noqa: E402
import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYARROW_NUM_THREADS",
):
    os.environ[_name] = "1"

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_fresh_confirmation_freeze as freshz,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_fresh_confirmation_outcomes as fresho,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_fresh_confirmation_support as freshs,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
RECORD_ROOT = freshz.OUTPUT_ROOT / "outcomes" / "two_supported_routes_20260923a"
FREEZE_PATH = RECORD_ROOT / "two_route_freeze.json"
RESULT_PATH = RECORD_ROOT / "two_route_outcome_result.json"
OUTCOME_ROOT = fresho.ARTIFACT_ROOT / "fresh_confirmation_two_route_20260923a"
SUPPORT_ID = "fresh_confirmation_support_20260923a"
SUPPORTED = ("fresh_market_state_activity", "fresh_meme_convergence_crossing")
PARKED = "fresh_daily_current_session_vwap"


def artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


def _verify_artifact(item: dict[str, Any]) -> None:
    path = Path(item["path"])
    if not path.is_file() or g0.sha256_file(path) != item["sha256"]:
        raise ValueError(f"Frozen two-route input changed: {path}")


def _load_supported_inputs() -> tuple[dict[str, Any], dict[str, Any], Path]:
    frozen = freshz.freeze()
    support_path = freshs.support_result_path(SUPPORT_ID)
    support = json.loads(support_path.read_text(encoding="utf-8"))
    if support.get("batch_id") != frozen["batch_id"]:
        raise ValueError("Support belongs to a different frozen question batch.")
    if support.get("status") != "insufficient_outcome_blind_support":
        raise ValueError("The two-route continuation requires the failed support gate.")
    if support.get("future_reaction_outcomes_read") is not False:
        raise ValueError("Support was not outcome blind.")
    if support.get("all_three_support_routes_built_together") is not True:
        raise ValueError("The original three support routes were not assembled together.")
    routes = support.get("routes", {})
    if set(routes) != {*SUPPORTED, PARKED}:
        raise ValueError("Support route names differ from the original frozen questions.")
    if any(routes[name].get("support_gate_pass") is not True for name in SUPPORTED):
        raise ValueError("A retained route lacks its original support gate.")
    if routes[PARKED].get("support_gate_pass") is not False:
        raise ValueError("Daily VWAP did not fail its original support gate.")
    for item in support.get("source_contracts", {}).values():
        _verify_artifact(item)
    for name in SUPPORTED:
        route = routes[name]
        if "causal_cross_asset_context" in route:
            items = [route["causal_cross_asset_context"]]
            items.extend(
                contract
                for row in route["inventory"]
                for contract in (row["feature"], row["label_mapping"])
            )
        else:
            items = [row["support"] for row in route["inventory"]]
        for item in items:
            _verify_artifact(item)
    return frozen, support, support_path


def freeze() -> dict[str, Any]:
    frozen, support, support_path = _load_supported_inputs()
    if FREEZE_PATH.is_file():
        existing = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        if existing.get("status") != "frozen_two_supported_routes_before_outcomes":
            raise ValueError("Existing two-route freeze has an unexpected status.")
        for item in existing["source_contracts"].values():
            _verify_artifact(item)
        return existing
    parked = support["routes"][PARKED]
    result = {
        "schema_version": 1,
        "batch_id": frozen["batch_id"],
        "created_at_utc": g0.utc_now(),
        "status": "frozen_two_supported_routes_before_outcomes",
        "supported_routes": list(SUPPORTED),
        "parked_route": PARKED,
        "park_reason": (
            "The week-shifted VWAP comparison supplied fewer than 10 contacts per "
            "coin/control/period, leaving zero eligible coins in fresh_early and one "
            "in fresh_late. The original 10-per-cell and five-coin gate is unchanged."
        ),
        "parked_eligible_pairs_by_period": parked["eligible_pairs_by_period"],
        "periods": list(frozen["fresh_periods"]),
        "original_questions_and_controls_unchanged": True,
        "future_reaction_outcomes_read": False,
        "future_signed_direction_read": False,
        "profit_read": False,
        "source_contracts": {
            "original_question_freeze": artifact(freshz.FREEZE_PATH),
            "joint_support_result": artifact(support_path),
            "two_route_materializer": artifact(ANALYSIS_PATH),
            "original_outcome_helpers": artifact(fresho.ANALYSIS_PATH),
        },
    }
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(result, FREEZE_PATH)
    return result


def materialize() -> dict[str, Any]:
    if not FREEZE_PATH.is_file():
        raise RuntimeError("Freeze the outcome-blind two-route decision first.")
    freeze()
    if RESULT_PATH.is_file():
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        if result.get("status") == "completed_two_supported_unsigned_outcomes":
            _verify_artifact(result["routes"][SUPPORTED[0]]["target_calibration"])
            for route in result["routes"].values():
                for row in route["inventory"]:
                    for key in (
                        ("actual_target_cache", "shuffled_target_cache", "evaluation_cache")
                        if route["route_id"] == SUPPORTED[0]
                        else ("outcomes",)
                    ):
                        _verify_artifact(row[key])
            return result
    frozen, support, support_path = _load_supported_inputs()
    state = fresho.materialize_state_targets(frozen, support, OUTCOME_ROOT)
    convergence = fresho.materialize_direct_route(
        support["routes"][SUPPORTED[1]],
        cohort="meme",
        horizon=2,
        scope_kind="round_distribution_convergence_pooled",
        scope_value="all_60_frozen_definitions",
        output_root=OUTCOME_ROOT,
    )
    if not state["outcome_gate_pass"] or not convergence["outcome_gate_pass"]:
        raise ValueError("A retained route failed its unchanged outcome support gate.")
    result = {
        "schema_version": 1,
        "batch_id": frozen["batch_id"],
        "created_at_utc": g0.utc_now(),
        "status": "completed_two_supported_unsigned_outcomes",
        "routes": {state["route_id"]: state, convergence["route_id"]: convergence},
        "parked_route": PARKED,
        "source_contracts": {
            "two_route_freeze": artifact(FREEZE_PATH),
            "joint_support_result": artifact(support_path),
        },
        "future_reaction_outcomes_read": True,
        "future_signed_direction_read": False,
        "profit_read": False,
        "next_action": "Score the two unchanged questions against their frozen controls.",
    }
    g0.atomic_write_json(result, RESULT_PATH)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("freeze", "materialize"), default="freeze")
    args = parser.parse_args(argv)
    result = freeze() if args.phase == "freeze" else materialize()
    print(json.dumps({key: result[key] for key in ("status", "batch_id")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
