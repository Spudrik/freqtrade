"""Run Generation 25's outcome-blind OHLCV and external-source coverage gate."""

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
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation14_external_readiness as g14x,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation17_joint_review as g17j,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation17_level_source_atlas as g17l,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_joint_review as g24j,
)


ANALYSIS_PATH = Path(__file__).resolve()
DEFAULT_RUN_ID = "g25_outcome_blind_coverage_20260828a"
OUTPUT_ROOT = g24j.REVIEW_ROOT.parent.parent / "g25_broad_siblings"
RECORD_ROOT = OUTPUT_ROOT / "coverage_gate"
PARENT_REVIEW = g24j.REVIEW_ROOT / g24j.DEFAULT_REVIEW_ID / "g24_joint_review.json"
PARENT_QUEUE = (
    g24j.REVIEW_ROOT / g24j.DEFAULT_REVIEW_ID / "g25_sibling_branch_queue.json"
)
G14_EXTERNAL_RESULT = (
    g14x.RECORD_ROOT / g14x.DEFAULT_RUN_ID / "g14_external_readiness_result.json"
)
G17_EXTERNAL_COVERAGE = (
    g17j.REVIEW_ROOT / g17j.DEFAULT_REVIEW_ID / "g17_external_coverage.csv"
)

LIVE_STATUS_PATHS = {
    "context_live_news": REPO_ROOT
    / "user_data"
    / "collector_data"
    / "news"
    / "collector_status.json",
    "context_live_web": REPO_ROOT
    / "user_data"
    / "collector_data"
    / "web"
    / "collector_status.json",
    "context_global_market_macro": REPO_ROOT
    / "user_data"
    / "collector_data"
    / "global_context"
    / "collector_status.json",
    "orderbook_live_multi_venue": REPO_ROOT
    / "user_data"
    / "collector_data"
    / "orderbook"
    / "collector_status.json",
}

FRESH_START = pd.Timestamp("2026-08-20T00:00:00Z")
FRESH_BLOCKS = (
    {
        "id": "g25_fresh_early",
        "start_utc": "2026-08-20T00:00:00Z",
        "end_utc_exclusive": "2026-09-05T00:00:00Z",
    },
    {
        "id": "g25_fresh_late",
        "start_utc": "2026-09-05T00:00:00Z",
        "end_utc_exclusive": "2026-09-20T00:00:00Z",
    },
)


def artifact(path: Path) -> dict[str, Any]:
    return g24j.artifact(path)


def validate_parent() -> None:
    review = json.loads(PARENT_REVIEW.read_text(encoding="utf-8"))
    queue = json.loads(PARENT_QUEUE.read_text(encoding="utf-8"))
    if review.get("status") != "completed_generation24_joint_review":
        raise ValueError("Generation 24 joint review is not terminal.")
    if not review.get("all_five_active_siblings_terminal_before_review"):
        raise ValueError("Generation 24 sibling ordering failed.")
    if queue.get("status") != "queued_after_complete_generation24_joint_review":
        raise ValueError("Generation 25 queue is invalid.")
    if queue.get("active_siblings") != 5:
        raise ValueError("Generation 25 coverage gate lost its five-route portfolio.")


def ohlcv_inventory() -> DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort, manifest_path in g17l.COHORT_MANIFESTS.items():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for pair in manifest["data"]["pairs"]:
            for timeframe in ("1h", "4h", "8h", "1d"):
                path = g0.ohlcv_path(str(pair), timeframe)
                if not path.is_file():
                    rows.append(
                        {
                            "cohort": cohort,
                            "pair": pair,
                            "timeframe": timeframe,
                            "exists": False,
                            "rows": 0,
                            "start_utc": None,
                            "end_utc": None,
                            "duplicate_timestamps": None,
                            "gap_count": None,
                        }
                    )
                    continue
                frame = pd.read_feather(path, columns=["date"])
                dates = pd.to_datetime(frame["date"], utc=True, errors="raise")
                hours = g0.timeframe_hours(timeframe)
                deltas = dates.sort_values().diff().dt.total_seconds().div(3600.0)
                rows.append(
                    {
                        "cohort": cohort,
                        "pair": pair,
                        "timeframe": timeframe,
                        "exists": True,
                        "rows": len(dates),
                        "start_utc": dates.min().isoformat(),
                        "end_utc": dates.max().isoformat(),
                        "duplicate_timestamps": int(dates.duplicated().sum()),
                        "gap_count": int(deltas.gt(hours * 1.5).sum()),
                    }
                )
    return DataFrame.from_records(rows).sort_values(
        ["cohort", "pair", "timeframe"]
    ).reset_index(drop=True)


def fresh_normal_coverage(inventory: DataFrame) -> dict[str, Any]:
    normal = inventory.loc[
        inventory["cohort"].eq("normal") & inventory["timeframe"].eq("1h")
    ].copy()
    ends = pd.to_datetime(normal["end_utc"], utc=True, errors="coerce")
    common_end = ends.min()
    required_end = pd.Timestamp(FRESH_BLOCKS[-1]["end_utc_exclusive"])
    return {
        "fresh_start_utc": FRESH_START.isoformat(),
        "declared_blocks": list(FRESH_BLOCKS),
        "normal_pairs": int(normal["pair"].nunique()),
        "common_latest_1h_candle_utc": (
            common_end.isoformat() if pd.notna(common_end) else None
        ),
        "available_common_fresh_hours": (
            max(0.0, float((common_end - FRESH_START) / pd.Timedelta(hours=1)))
            if pd.notna(common_end)
            else 0.0
        ),
        "required_common_end_utc_exclusive": required_end.isoformat(),
        "two_fresh_blocks_calendar_coverage_pass": bool(
            pd.notna(common_end) and common_end >= required_end
        ),
        "event_and_control_support_still_required_after_calendar_gate": True,
    }


def live_status_audit() -> dict[str, Any]:
    sources: dict[str, Any] = {}
    for source, path in LIVE_STATUS_PATHS.items():
        if not path.is_file():
            sources[source] = {"status_file_present": False, "ready_for_model": False}
            continue
        status = json.loads(path.read_text(encoding="utf-8"))
        sources[source] = {
            "status_file_present": True,
            "collector_status": status.get("status"),
            "started_at": status.get("started_at"),
            "heartbeat_at": status.get("heartbeat_at"),
            "last_error": status.get("last_error"),
            "source_specific_historical_coverage_proven_by_status_file": False,
            "ready_for_model": False,
            "reason": (
                "A live status file proves collection health, not two timestamp-safe "
                "historical evaluation blocks. The live SQLite database was not opened "
                "while its collector was running."
            ),
            "status_artifact": artifact(path),
        }
    return sources


def historical_external_audit() -> dict[str, Any]:
    result = json.loads(G14_EXTERNAL_RESULT.read_text(encoding="utf-8"))
    if result.get("status") != "completed_generation14_external_readiness":
        raise ValueError("Historical external readiness record is not terminal.")
    decisions = {item["source"]: item for item in result["decisions"]}
    coverage = pd.read_csv(G17_EXTERNAL_COVERAGE)
    return {
        "historical_news": {
            "status": decisions["historical_news"]["status"],
            "ready_rows_in_generation17_surface": int(
                pd.to_numeric(
                    coverage.loc[
                        coverage["source"].eq("historical_news"), "ready_rows"
                    ],
                    errors="coerce",
                ).sum()
            ),
            "advance": False,
        },
        "historical_orderbook": {
            "status": decisions["historical_orderbook"]["status"],
            "ready_rows_in_generation17_surface": int(
                pd.to_numeric(
                    coverage.loc[
                        coverage["source"].eq("btc_orderbook"), "ready_rows"
                    ],
                    errors="coerce",
                ).sum()
            ),
            "advance": False,
            "reason": (
                "Older validated windows exist, but broad orderbook conditioning was "
                "already tested without a retained claim and the newest two-block "
                "coverage failed. Do not rerun it without a new retained parent mechanism."
            ),
        },
        "source_contracts": {
            "generation14_external_result": artifact(G14_EXTERNAL_RESULT),
            "generation17_external_coverage": artifact(G17_EXTERNAL_COVERAGE),
        },
    }


def run(run_id: str, *, overwrite: bool = False) -> int:
    validate_parent()
    run_dir = RECORD_ROOT / run_id
    result_path = run_dir / "g25_coverage_result.json"
    inventory_path = run_dir / "g25_ohlcv_inventory.csv"
    if result_path.is_file() and not overwrite:
        print(result_path.read_text(encoding="utf-8"))
        return 0
    inventory = ohlcv_inventory()
    run_dir.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_csv(inventory, inventory_path)
    fresh = fresh_normal_coverage(inventory)
    historical = historical_external_audit()
    live = live_status_audit()
    result = {
        "schema_version": 1,
        "generation": 25,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation25_outcome_blind_coverage_gate",
        "future_reaction_outcomes_read": False,
        "future_signed_direction_read": False,
        "profit_read": False,
        "ohlcv": {
            "rows": len(inventory),
            "missing_files": int((~inventory["exists"].astype(bool)).sum()),
            "duplicate_timestamps": int(
                pd.to_numeric(inventory["duplicate_timestamps"], errors="coerce").sum()
            ),
            "fresh_normal_coverage": fresh,
        },
        "external_sources": {
            "historical": historical,
            "live_status_only": live,
            "live_sqlite_opened": False,
        },
        "branch_gates": {
            "g25a_daily_anchored_vwap_fresh_confirmation": {
                "advance": False,
                "status": "parked_insufficient_fresh_ohlcv",
                "reason": (
                    "Two new 15-16 day blocks do not yet exist after the Generation 24 "
                    "selection boundary."
                ),
            },
            "g25b_convergence_control_representation_repair": {
                "advance": True,
                "status": "ready_for_outcome_blind_support_repair",
            },
            "g25c_connected_volume_profile_zones": {
                "advance": True,
                "status": "ready_for_outcome_blind_support_build",
            },
            "g25d_market_state_only_activity_portability": {
                "advance": True,
                "status": "ready_for_same_holdout_exploratory_attribution",
                "fresh_confirmation": False,
            },
            "g25e_external_source_overlap_gate": {
                "advance": False,
                "status": "parked_no_new_two_block_timestamp_safe_source",
                "reason": (
                    "Historical sources add no new supported window and live status files "
                    "do not prove two complete timestamp-safe blocks."
                ),
            },
        },
        "artifacts": {"ohlcv_inventory": artifact(inventory_path)},
        "source_contracts": {
            "generation24_joint_review": artifact(PARENT_REVIEW),
            "generation25_queue": artifact(PARENT_QUEUE),
            "analysis_script": artifact(ANALYSIS_PATH),
        },
        "result_path": str(result_path.resolve()),
    }
    g0.atomic_write_json(result, result_path)
    print(json.dumps(result, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    return run(args.run_id, overwrite=args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())
