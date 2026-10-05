"""Freeze the breadth-balanced validation batch selected by the Layer 2 joint review."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_central_bank_activity_direct as central_bank,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_cpi_links_direct as cpi_links,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_layer2_source_activity_direct as source_activity,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
OUTPUT_ROOT = source_activity.frozen.OUTPUT_ROOT / "layer2_validation_batch_20260907a"
FREEZE_PATH = OUTPUT_ROOT / "layer2_validation_freeze.json"
RESULT_PATH = OUTPUT_ROOT / "layer2_validation_freeze_result.json"
PERMUTATION_ITERATIONS = 2000
PERMUTATION_SEED = 20260907


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def load_result(path: Path, expected_status: str) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") != expected_status:
        raise ValueError(f"Parent result is not terminal: {path}")
    if result.get("profit_used"):
        raise ValueError(f"Parent unexpectedly used profit: {path}")
    return result


def validate_parent_decisions() -> dict[str, Any]:
    source_result = load_result(
        source_activity.RESULT_PATH,
        "completed_layer2_multi_source_activity_review",
    )
    cpi_result = load_result(cpi_links.RESULT_PATH, "completed_cpi_leader_and_meme_link_review")
    bank_result = load_result(
        central_bank.RESULT_PATH, "completed_central_bank_activity_direct_test"
    )
    source = pd.read_csv(source_activity.DECISION_PATH)
    retained_source = source.loc[source["verdict"].str.startswith("retained")]
    if set(retained_source["event_source"]) != {"sec_hyperscaler_earnings"}:
        raise ValueError("Layer 2 source survivor changed before validation freeze")
    if set(retained_source["horizon_minutes"].astype(int)) != {15}:
        raise ValueError("The SEC source survivor is no longer the 15-minute route")
    cpi = pd.read_csv(cpi_links.DECISION_PATH)
    retained_cpi = cpi.loc[cpi["verdict"].str.startswith("retained")]
    if set(retained_cpi["route"]) != {
        "cpi_first_move_to_BNB/USDT:USDT",
        "cpi_first_move_to_ADA/USDT:USDT",
        "cpi_first_move_to_TRX/USDT:USDT",
        "cpi_first_move_to_established_group_median",
    }:
        raise ValueError("CPI transmission survivor set changed before validation freeze")
    meme = cpi.loc[cpi["route"].eq("cpi_conditioned_top10_meme_response")]
    if len(meme) != 1 or meme.iloc[0]["verdict"] != "coverage_limited":
        raise ValueError("The CPI meme route must remain parked in this batch")
    bank = pd.read_csv(central_bank.DECISION_PATH)
    retained_bank = bank.loc[bank["verdict"].str.startswith("retained")]
    if len(retained_bank) != 1:
        raise ValueError("Expected one retained central-bank activity route")
    row = retained_bank.iloc[0]
    if row["bank"] != "BoJ" or row["pair"] != "ETH/USDT:USDT" or int(row["horizon_minutes"]) != 60:
        raise ValueError("The retained central-bank route changed")
    return {
        "source_result": source_result,
        "cpi_result": cpi_result,
        "central_bank_result": bank_result,
    }


def definitions() -> list[dict[str, Any]]:
    return [
        {
            "branch_id": "sec_activity_full_family_randomization",
            "trader_question": (
                "Did earnings-like hyperscaler filing times identify unusual 15-minute "
                "BTC/ETH activity beyond what could arise from searching every Layer 2 "
                "source, asset, horizon, period, and collision view?"
            ),
            "method": (
                "Nominate a matched event or ordinary-time sample for every whole event, "
                "keep correlated BTC/ETH and horizon rows together, recompute every "
                "searched activity route, and compare the retained family with the "
                "largest shuffled route."
            ),
            "permutations": PERMUTATION_ITERATIONS,
            "pass_rule": (
                "Family-adjusted probability <= 0.05 and the unchanged 1.20 median "
                "activity, 55% above-control, two-period, and collision-clean gates still "
                "pass for the connected BTC/ETH 15-minute family."
            ),
        },
        {
            "branch_id": "sec_clock_and_equity_alternative_controls",
            "trader_question": (
                "Is the SEC acceptance clock close enough to first public earnings news, "
                "and is the crypto activity more than a simultaneous technology-equity "
                "or volatility-market move?"
            ),
            "method": (
                "Audit first-public press-release clocks independently of EDGAR; inventory "
                "timestamp-safe Nasdaq/technology and VIX intraday coverage before opening "
                "new outcomes. If minute coverage is unavailable, record the attribution "
                "limit rather than substituting daily data for a 15-minute question."
            ),
            "minimum_first_public_clock_coverage": 0.80,
            "pass_rule": (
                "Attribution requires >=80% exact first-public clocks and a repeated crypto "
                "activity increment after timestamp-matched intraday equity/volatility "
                "controls. Otherwise retain only an event-time association or defer."
            ),
        },
        {
            "branch_id": "cpi_transmission_full_family_randomization",
            "trader_question": (
                "Does the unusually large first CPI move genuinely lead the following "
                "established-coin direction after accounting for all 60 leader, follower, "
                "and window combinations that were searched?"
            ),
            "method": (
                "Shuffle complete first-move blocks among whole CPI events within each "
                "historical period, keep all related leaders and windows joined, and use "
                "the largest apparent advantage among all 60 routes as the null control."
            ),
            "permutations": PERMUTATION_ITERATIONS,
            "pass_rule": (
                "Full-family probability <=0.05 while the unchanged overall >=55%, each "
                "period >=50%, ordinary-control uplift >=5 points, and follower-own-move "
                "comparison remain satisfied."
            ),
        },
        {
            "branch_id": "cpi_common_event_and_self_momentum",
            "trader_question": (
                "On exactly the same complete CPI events, does BTC/ETH add useful direction "
                "beyond simply following each established coin's own first 15 minutes?"
            ),
            "method": (
                "Use one common complete-event set for every candidate; predeclare the "
                "established-group median as primary and individual coins as diagnostic; "
                "compare BTC, ETH, agreement, follower-own continuation, majority direction, "
                "and matched ordinary times without selecting a winner afterwards."
            ),
            "minimum_complete_events": 10,
            "pass_rule": (
                "Primary group result >=55% overall and >=50% in each period, beats ordinary "
                "times by >=5 points, and improves on follower-own continuation by >=5 "
                "points on the same events."
            ),
        },
        {
            "branch_id": "boj_activity_full_family_randomization",
            "trader_question": (
                "Did the one-hour Ethereum activity around scheduled Bank of Japan decisions "
                "survive the fact that 24 bank, asset, and horizon routes were searched?"
            ),
            "method": (
                "Apply whole-event matched-time randomization while preserving related "
                "assets and horizons, then compare the observed BoJ/ETH/60-minute route with "
                "the largest result among the full 24-route family."
            ),
            "permutations": PERMUTATION_ITERATIONS,
            "pass_rule": (
                "Family-adjusted probability <=0.05 and the unchanged 1.20 activity, 55% "
                "above-control, two-period, and collision-clean gates still pass."
            ),
        },
    ]


def freeze(*, overwrite: bool = False) -> dict[str, Any]:
    if RESULT_PATH.is_file() and not overwrite:
        result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        if result.get("status") != "completed_layer2_validation_outcome_blind_freeze":
            raise ValueError("Invalid Layer 2 validation freeze result")
        return result
    validate_parent_decisions()
    routes = definitions()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    frozen = {
        "schema_version": 1,
        "status": "frozen_layer2_validation_before_new_validation_outcomes",
        "created_at_utc": g0.utc_now(),
        "branch_layer": 1,
        "parent_outcomes_reviewed": True,
        "new_validation_outcomes_read": False,
        "profit_used": False,
        "permutation_seed": PERMUTATION_SEED,
        "breadth_rule": (
            "Complete all five siblings and review them jointly before any survivor "
            "creates another branch."
        ),
        "parked_routes": [
            "uk_ons_cpi_clock_only_activity",
            "japan_tankan_clock_only_activity",
            "esma_sovereign_rating_activity",
            "all_hyperscaler_8k_activity",
            "cpi_conditioned_top10_meme_response_until_prospective_coverage",
        ],
        "routes": routes,
        "source_contracts": {
            "source_activity_result": artifact(source_activity.RESULT_PATH),
            "source_activity_decisions": artifact(source_activity.DECISION_PATH),
            "source_activity_rows": artifact(source_activity.OUTCOME_PATH),
            "cpi_result": artifact(cpi_links.RESULT_PATH),
            "cpi_decisions": artifact(cpi_links.DECISION_PATH),
            "cpi_transmission_rows": artifact(cpi_links.TRANSMISSION_ROWS_PATH),
            "central_bank_result": artifact(central_bank.RESULT_PATH),
            "central_bank_decisions": artifact(central_bank.DECISION_PATH),
            "central_bank_rows": artifact(central_bank.OUTCOME_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
        },
    }
    g0.atomic_write_json(frozen, FREEZE_PATH)
    result = {
        "schema_version": 1,
        "status": "completed_layer2_validation_outcome_blind_freeze",
        "created_at_utc": g0.utc_now(),
        "new_validation_outcomes_read": False,
        "profit_used": False,
        "route_count": len(routes),
        "artifacts": {
            "freeze": artifact(FREEZE_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
        },
    }
    g0.atomic_write_json(result, RESULT_PATH)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(freeze(overwrite=args.overwrite), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
