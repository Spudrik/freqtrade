"""Score the frozen meme crossing question on the two fresh periods."""

from __future__ import annotations

# Bound numerical pools before importing the research modules.
# ruff: noqa: E402
import json
import os
import sys
from pathlib import Path
from typing import Any


for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"

import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_fresh_confirmation_two_route as split,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation18_direct_confirmation as g18d,
)


ANALYSIS_PATH = Path(__file__).resolve()
QUESTION_KEYS = ("scope_kind", "scope_value", "metric", "horizon_hours")
PERIODS = {"fresh_early", "fresh_late"}
ROUTE_ID = "fresh_meme_convergence_crossing"
RESULT_PATH = split.RECORD_ROOT / "meme_crossing_review.json"


def _source() -> tuple[dict[str, Any], dict[str, Any], DataFrame]:
    frozen = split.freeze()
    result = json.loads(split.RESULT_PATH.read_text(encoding="utf-8"))
    if result.get("status") != "completed_two_supported_unsigned_outcomes":
        raise ValueError("Both supported routes must have completed materialization.")
    for item in result["source_contracts"].values():
        split._verify_artifact(item)
    route = result["routes"][ROUTE_ID]
    if route.get("outcome_gate_pass") is not True:
        raise ValueError("The meme outcome route did not pass its frozen support gate.")
    question = next(item for item in split.freshz.freeze()["siblings"] if item["id"] == ROUTE_ID)
    if tuple(route["controls"]) != ("actual", *tuple(question["controls"])):
        raise ValueError("The meme control list drifted from the frozen question.")
    if len(route["inventory"]) != 10:
        raise ValueError("The top-ten meme cohort is incomplete.")
    parts = []
    for row in route["inventory"]:
        split._verify_artifact(row["outcomes"])
        parts.append(pd.read_parquet(row["outcomes"]["path"]))
    return frozen, question, pd.concat(parts, ignore_index=True, sort=False)


def score() -> dict[str, Any]:
    if RESULT_PATH.is_file():
        existing = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
        if existing.get("status") == "completed_fresh_meme_crossing_review":
            for item in existing["source_contracts"].values():
                split._verify_artifact(item)
            for item in existing["artifacts"].values():
                split._verify_artifact(item)
            return existing
    _, question, events = _source()
    required = {
        "cohort", "pair", "g18_period", "scope_kind", "scope_value", "control",
        "outcome__crossing_count_h2",
    }
    if missing := required.difference(events.columns):
        raise ValueError(f"Fresh meme outcome columns missing: {sorted(missing)}")
    if set(events["g18_period"]) != PERIODS:
        raise ValueError("Fresh meme periods differ from the frozen pair of periods.")
    if set(events["control"]) != {"actual", *question["controls"]}:
        raise ValueError("Fresh meme controls differ from the frozen control ladder.")
    scope = question["exact_question"]
    if set(events["scope_value"]) != {scope["scope_value"]}:
        raise ValueError("The pooled convergence scope changed.")
    keys = ["cohort", "pair", "g18_period", "scope_kind", "scope_value", "control"]
    summary = (
        events.groupby(keys, observed=True, sort=False)["outcome__crossing_count_h2"]
        .agg(["mean", "size"])
        .reset_index()
        .rename(columns={"mean": "outcome_mean", "size": "event_rows"})
    )
    summary["metric"] = scope["metric"]
    summary["horizon_hours"] = int(scope["horizon_hours"])
    contrasts = g18d.paired_contrasts(summary, question["controls"])
    scores = g18d.period_scores(contrasts, QUESTION_KEYS)
    relevant = scores.loc[
        scores["market_scope"].eq(scope["market_scope"])
        & scores["period"].isin(PERIODS)
    ].copy()
    complete = bool(
        len(relevant) == len(PERIODS) * len(question["controls"])
        and set(relevant["period"]) == PERIODS
        and set(relevant["comparison"]) == set(question["controls"])
        and relevant.groupby("period")["comparison"].nunique().eq(len(question["controls"])).all()
    )
    point = bool(complete and relevant["point_period_pass"].all())
    strict = bool(complete and relevant["strict_period_pass"].all())
    decision = DataFrame.from_records(
        [{
            "route_id": ROUTE_ID,
            "market_scope": scope["market_scope"],
            "fresh_periods": ";".join(sorted(PERIODS)),
            "complete_control_period_ladder": complete,
            "point_confirmation": point,
            "strict_confirmation": strict,
            "status": (
                "strict_fresh_confirmation" if strict else
                "point_only_fresh_lead" if point else "not_confirmed_on_fresh_periods"
            ),
            "minimum_equal_coin_difference": (
                float(relevant["equal_coin_difference"].min()) if len(relevant) else None
            ),
            "minimum_bootstrap_lower_95": (
                float(relevant["bootstrap_lower_95"].min()) if len(relevant) else None
            ),
        }]
    )
    output_files = {
        "summary": split.RECORD_ROOT / "meme_crossing_summary.csv",
        "contrasts": split.RECORD_ROOT / "meme_crossing_pair_contrasts.csv",
        "scores": split.RECORD_ROOT / "meme_crossing_period_scores.csv",
        "decision": split.RECORD_ROOT / "meme_crossing_decision.csv",
    }
    for name, frame in (
        ("summary", summary), ("contrasts", contrasts),
        ("scores", scores), ("decision", decision),
    ):
        g0.atomic_write_csv(frame, output_files[name])
    result = {
        "schema_version": 1,
        "created_at_utc": g0.utc_now(),
        "status": "completed_fresh_meme_crossing_review",
        "decision": decision.iloc[0].to_dict(),
        "event_rows": len(events),
        "pairs": int(events["pair"].nunique()),
        "future_signed_direction_read": False,
        "profit_read": False,
        "source_contracts": {
            "two_route_freeze": split.artifact(split.FREEZE_PATH),
            "two_route_outcomes": split.artifact(split.RESULT_PATH),
            "analysis_script": split.artifact(ANALYSIS_PATH),
        },
        "artifacts": {name: split.artifact(path) for name, path in output_files.items()},
    }
    g0.atomic_write_json(result, RESULT_PATH)
    return result


def main() -> int:
    result = score()
    print(json.dumps({
        "status": result["status"],
        "decision": result["decision"],
        "event_rows": result["event_rows"],
        "pairs": result["pairs"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
