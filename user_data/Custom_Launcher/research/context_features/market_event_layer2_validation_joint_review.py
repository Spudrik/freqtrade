"""Jointly review every sibling in the frozen Layer 2 validation batch."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_layer2_activity_family_validation as activity_validation,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_layer2_cpi_transmission_validation as cpi_validation,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_layer2_sec_attribution_preflight as sec_preflight,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
OUTPUT_ROOT = activity_validation.validation_freeze.OUTPUT_ROOT / (
    "layer2_validation_joint_review_20260908a"
)
DECISION_PATH = OUTPUT_ROOT / "layer2_validation_joint_decisions.csv"
REPORT_PATH = OUTPUT_ROOT / "layer2_validation_joint_review.md"
RESULT_PATH = OUTPUT_ROOT / "layer2_validation_joint_result.json"


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def load_result(path: Path, expected_status: str) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != expected_status:
        raise ValueError(f"Validation sibling is not terminal: {path}")
    if result.get("profit_used"):
        raise ValueError(f"Validation sibling unexpectedly used profit: {path}")
    return result


def build_joint_decisions(
    activity: dict[str, Any],
    cpi: dict[str, Any],
    sec: dict[str, Any],
) -> list[dict[str, Any]]:
    activity_by_id = {
        str(row["branch_id"]): row for row in activity["decisions"]
    }
    sec_activity = activity_by_id["sec_activity_full_family_randomization"]
    boj = activity_by_id["boj_activity_full_family_randomization"]
    cpi_family = cpi["randomization_decision"]
    cpi_common = cpi["common_event_decision"]
    sec_clock = sec["decision"]
    return [
        {
            "branch_id": "sec_activity_full_family_randomization",
            "plain_question": "Is the earnings-time BTC/ETH activity association robust?",
            "result": "retained_association",
            "key_measure": float(sec_activity["familywise_p_value"]),
            "key_measure_name": "familywide_randomization_probability",
            "decision": (
                "Retain as an unusual 15-minute event-time association, not as cause "
                "or direction."
            ),
        },
        {
            "branch_id": "sec_clock_and_equity_alternative_controls",
            "plain_question": "Can the earnings-time association be causally attributed?",
            "result": "deferred_missing_data",
            "key_measure": float(sec_clock["exact_first_public_clock_coverage"]),
            "key_measure_name": "exact_first_public_clock_coverage",
            "decision": (
                "Defer attribution: no exact independent press-release clock archive "
                "and no overlapping intraday Nasdaq/VIX controls."
            ),
        },
        {
            "branch_id": "cpi_transmission_full_family_randomization",
            "plain_question": "Does CPI BTC/ETH motion lead established coins?",
            "result": "not_retained",
            "key_measure": float(cpi_family["familywise_p_value"]),
            "key_measure_name": "familywide_randomization_probability",
            "decision": (
                "Reject the separate leader claim after accounting for all 60 searched "
                "routes; retain only the older CPI short-reaction finding."
            ),
        },
        {
            "branch_id": "cpi_common_event_and_self_momentum",
            "plain_question": "Does BTC/ETH add more than each follower's own early move?",
            "result": "no_incremental_leader_value",
            "key_measure": float(
                cpi_common["uplift_over_follower_own_on_same_calls"]
            ),
            "key_measure_name": "direction_uplift_over_follower_own_move",
            "decision": (
                "No: the measured four-point uplift missed the frozen five-point floor."
            ),
        },
        {
            "branch_id": "boj_activity_full_family_randomization",
            "plain_question": "Is the one-hour BoJ/ETH activity lead robust?",
            "result": "not_retained",
            "key_measure": float(boj["familywise_p_value"]),
            "key_measure_name": "familywide_randomization_probability",
            "decision": (
                "Reject this narrow route; it failed current data gates and the full-family "
                "randomization."
            ),
        },
    ]


def render_report(decisions: list[dict[str, Any]]) -> str:
    by_id = {row["branch_id"]: row for row in decisions}
    sec = by_id["sec_activity_full_family_randomization"]
    sec_clock = by_id["sec_clock_and_equity_alternative_controls"]
    cpi = by_id["cpi_transmission_full_family_randomization"]
    cpi_self = by_id["cpi_common_event_and_self_momentum"]
    boj = by_id["boj_activity_full_family_randomization"]
    return "\n".join(
        [
            "# Layer 2 Validation Joint Review",
            "",
            "All five frozen siblings were completed before interpreting or branching.",
            "",
            "## Results in plain terms",
            "",
            f"- SEC earnings-time activity: {sec['result']}. Only 7 of 2,000 shuffled "
            "full-family trials looked at least as strong (about 0.4%).",
            f"- SEC cause: {sec_clock['result']}. Exact independent first-public clock "
            "coverage was 0%, and no historical intraday Nasdaq/VIX controls were present.",
            f"- CPI leader transmission: {cpi['result']}. The full-family probability was "
            "about 72.7%, so selecting among the 60 routes readily explains the apparent "
            "64% result.",
            f"- CPI follower self-momentum: {cpi_self['result']}. BTC/ETH agreement added "
            "four percentage points over the followers' own first move, below the frozen "
            "five-point minimum.",
            f"- Bank of Japan activity: {boj['result']}. The full-family probability was "
            "about 74.9%, and the original gates no longer passed on repaired data.",
            "",
            "## Joint decision",
            "",
            "Carry the SEC earnings-time association only as a possible short activity "
            "context feature. Do not call the SEC filing the cause and do not infer "
            "direction. Remove the CPI leader-to-established route and narrow BoJ route "
            "from the positive queue. The separately established CPI short-reaction lead "
            "is unchanged and still needs a true market-expectation input before it can "
            "support a directional claim.",
            "",
            "No profit, entry, exit, or trading rule was tested or promoted.",
            "",
        ]
    )


def run(*, overwrite: bool = False) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        if result.get("status") != "completed_layer2_validation_joint_review":
            raise ValueError("Existing Layer 2 joint review is not terminal")
        return result

    freeze = activity_validation.load_validation_contract()
    activity = load_result(
        activity_validation.RESULT_PATH,
        "completed_layer2_activity_family_validation",
    )
    cpi = load_result(
        cpi_validation.RESULT_PATH,
        "completed_layer2_cpi_transmission_validation",
    )
    sec = load_result(
        sec_preflight.RESULT_PATH,
        "completed_layer2_sec_attribution_preflight",
    )
    decisions = build_joint_decisions(activity, cpi, sec)
    frozen_ids = {str(row["branch_id"]) for row in freeze["routes"]}
    completed_ids = {str(row["branch_id"]) for row in decisions}
    if completed_ids != frozen_ids:
        raise ValueError("Joint review does not cover every frozen validation sibling")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_csv(DataFrame.from_records(decisions), DECISION_PATH)
    REPORT_PATH.write_text(render_report(decisions), encoding="utf-8", newline="\n")
    result = {
        "schema_version": 1,
        "status": "completed_layer2_validation_joint_review",
        "created_at_utc": g0.utc_now(),
        "profit_used": False,
        "all_frozen_siblings_completed": True,
        "frozen_sibling_count": len(frozen_ids),
        "retained_for_later_combinations": [
            "sec_earnings_time_15m_btc_eth_activity_association",
            "previously_retained_cpi_short_reaction",
        ],
        "removed_from_positive_queue": [
            "cpi_btc_eth_to_established_coin_leader_transmission",
            "boj_eth_60m_activity",
        ],
        "deferred_for_missing_data": [
            "sec_first_public_and_intraday_equity_adjusted_attribution",
            "cpi_true_expectation_surprise_direction",
        ],
        "decisions": decisions,
        "artifacts": {
            "decisions": artifact(DECISION_PATH),
            "report": artifact(REPORT_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
            "activity_result": artifact(activity_validation.RESULT_PATH),
            "cpi_result": artifact(cpi_validation.RESULT_PATH),
            "sec_attribution_result": artifact(sec_preflight.RESULT_PATH),
        },
    }
    g0.atomic_write_json(result, RESULT_PATH)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = run(overwrite=args.overwrite)
    print(json.dumps(result, indent=2, default=g0.json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
