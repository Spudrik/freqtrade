"""Repair only frozen macro-surprise BTC/ETH one-minute outcome gaps."""

from __future__ import annotations

# Numerical pools are bounded before pandas imports.
# ruff: noqa: E402, I001
import argparse
import json
import os
import re
import subprocess
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

import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_macro_expectation_direction_direct as direct,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation4_one_minute_download as g4d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation4_one_minute_repair as g4r,
)


ANALYSIS_PATH = Path(__file__).resolve()
OUTPUT_ROOT = direct.frozen.OUTPUT_ROOT / "one_minute_acquisition_20260911a"
INTERVALS_PATH = OUTPUT_ROOT / "macro_expectation_missing_intervals.csv"
FREEZE_PATH = OUTPUT_ROOT / "macro_expectation_acquisition_freeze.json"
RUN_RECORD_PATH = OUTPUT_ROOT / "macro_expectation_acquisition_run_record.json"
WINDOW_MINUTES = max(direct.frozen.HORIZONS_MINUTES)
MAXIMUM_WORKERS = 1


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def missing_event_windows(catalog: DataFrame) -> DataFrame:
    selected = catalog[catalog["historical_direction_test_eligible"]]
    anchors = selected[["expectation_episode_id", "anchor_utc"]].drop_duplicates()
    records: list[dict[str, Any]] = []
    for pair in direct.frozen.ASSETS:
        target = g0.ohlcv_path(pair, "1m")
        g4r.validate_main_target(target)
        frame = g0.load_ohlcv(target).sort_values("date", kind="stable")
        frame = frame.drop_duplicates("date").reset_index(drop=True)
        positions = direct.prior_direct.position_by_date(frame)
        for row in anchors.itertuples(index=False):
            anchor = pd.Timestamp(row.anchor_utc)
            position = positions.get(anchor)
            if position is None:
                reason = "missing_anchor_candle"
            else:
                _, reason = direct.prior_direct.future_return(
                    frame, position, WINDOW_MINUTES
                )
            if reason is None:
                continue
            records.append(
                {
                    "pair": pair,
                    "expectation_episode_id": row.expectation_episode_id,
                    "prior_coverage_status": reason,
                    "interval_start_utc": anchor,
                    "interval_end_exclusive_utc": anchor
                    + pd.Timedelta(minutes=WINDOW_MINUTES),
                    "target_file": str(target.resolve()),
                }
            )
    return DataFrame.from_records(records)


def pair_spanning_intervals(missing: DataFrame) -> DataFrame:
    columns = [
        "pair",
        "expectation_episode_ids",
        "interval_start_utc",
        "interval_end_exclusive_utc",
        "expected_rows",
        "timerange",
        "target_file",
    ]
    if missing.empty:
        return DataFrame(columns=columns)
    records: list[dict[str, Any]] = []
    for pair, group in missing.groupby("pair", sort=True):
        start = pd.Timestamp(group["interval_start_utc"].min())
        end = pd.Timestamp(group["interval_end_exclusive_utc"].max())
        targets = group["target_file"].unique()
        if len(targets) != 1:
            raise ValueError(f"Pair {pair} maps to several canonical target files")
        records.append(
            {
                "pair": pair,
                "expectation_episode_ids": "|".join(
                    group["expectation_episode_id"].drop_duplicates().astype(str)
                ),
                "interval_start_utc": start,
                "interval_end_exclusive_utc": end,
                "expected_rows": int((end - start) / pd.Timedelta(minutes=1)),
                "timerange": f"{int(start.timestamp())}-{int(end.timestamp())}",
                "target_file": str(targets[0]),
            }
        )
    return DataFrame.from_records(records, columns=columns)


def prepare(*, overwrite: bool = False) -> dict[str, Any]:
    if FREEZE_PATH.is_file() and not overwrite:
        freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        if freeze.get("status") != "frozen_macro_expectation_gap_acquisition":
            raise ValueError("Existing one-minute acquisition freeze is invalid")
        return freeze
    freeze_result, freeze_document, catalog, _ = direct.load_frozen_inputs()
    missing = missing_event_windows(catalog)
    intervals = pair_spanning_intervals(missing)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_csv(intervals, INTERVALS_PATH)
    targets = sorted({Path(path) for path in missing.get("target_file", [])})
    freeze = {
        "schema_version": 1,
        "status": "frozen_macro_expectation_gap_acquisition",
        "created_at_utc": g0.utc_now(),
        "outcome_values_used_for_selection": False,
        "profit_used": False,
        "selection_rule": (
            "Download only complete 60-minute spans for frozen eligible event anchors "
            "whose canonical one-minute file has an anchor or continuity gap."
        ),
        "window_minutes": WINDOW_MINUTES,
        "missing_pair_event_windows": len(missing),
        "pair_spanning_downloads": len(intervals),
        "selected_workers": MAXIMUM_WORKERS,
        "frozen_direction_input": freeze_result,
        "frozen_direction_question": freeze_document["plain_question"],
        "artifacts": {
            "intervals": artifact(INTERVALS_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
            "targets_before": [artifact(path) for path in targets],
        },
    }
    g0.atomic_write_json(freeze, FREEZE_PATH)
    return freeze


def acquire() -> int:
    freeze = prepare(overwrite=False)
    if freeze["artifacts"]["analysis_script"]["sha256"] != g0.sha256_file(
        ANALYSIS_PATH
    ):
        raise ValueError("Acquisition script changed after gap selection was frozen")
    if freeze["artifacts"]["intervals"]["sha256"] != g0.sha256_file(INTERVALS_PATH):
        raise ValueError("Frozen acquisition intervals changed")
    if not g4d.CONTROLLER_PYTHON.is_file():
        raise FileNotFoundError(
            f"Configured controller Python is missing: {g4d.CONTROLLER_PYTHON}"
        )
    if RUN_RECORD_PATH.is_file():
        existing = json.loads(RUN_RECORD_PATH.read_text(encoding="utf-8"))
        if existing.get("status") == "running":
            raise RuntimeError(f"An acquisition is already running: {RUN_RECORD_PATH}")

    intervals = pd.read_csv(INTERVALS_PATH)
    if intervals.empty:
        print("All frozen macro-expectation event windows already pass coverage.")
        return os.EX_OK
    started = g0.utc_now()
    session_id = re.sub(r"[^0-9A-Za-z]+", "", started)
    session_root = (
        g4r.ISOLATED_ROOT
        / "macro_expectation_direction"
        / "20260911a"
        / session_id
    )
    g4r.ensure_isolated_path(session_root)
    session_root.mkdir(parents=True, exist_ok=False)
    jobs: list[dict[str, Any]] = []
    for index, row in enumerate(intervals.to_dict(orient="records"), start=1):
        isolated = session_root / f"job_{index:03d}"
        target = Path(str(row["target_file"])).resolve()
        g4r.validate_main_target(target)
        command = g4r.isolated_download_args(
            str(row["pair"]), str(row["timerange"]), isolated
        )
        jobs.append(
            {
                "job_id": f"{index:03d}",
                "pair": str(row["pair"]),
                "expectation_episode_ids": str(row["expectation_episode_ids"]),
                "timerange": str(row["timerange"]),
                "interval_start_utc": str(row["interval_start_utc"]),
                "interval_end_exclusive_utc": str(
                    row["interval_end_exclusive_utc"]
                ),
                "expected_rows": int(row["expected_rows"]),
                "target_file": str(target),
                "isolated_datadir": str(isolated.resolve()),
                "command": subprocess.list2cmdline(command),
                "log_path": str(
                    (OUTPUT_ROOT / "download_logs" / f"{index:03d}.log").resolve()
                ),
                "status": "queued",
                "pid": None,
                "return_code": None,
            }
        )
    record: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "started_at_utc": started,
        "completed_at_utc": None,
        "controller_pid": os.getpid(),
        "controller_python": str(g4d.CONTROLLER_PYTHON.resolve()),
        "selected_workers": MAXIMUM_WORKERS,
        "jobs": jobs,
        "freeze": artifact(FREEZE_PATH),
    }
    g0.atomic_write_json(record, RUN_RECORD_PATH)

    for job in jobs:
        isolated = Path(str(job["isolated_datadir"]))
        isolated.mkdir(parents=True, exist_ok=False)
        command = g4r.isolated_download_args(
            str(job["pair"]), str(job["timerange"]), isolated
        )
        log_path = Path(str(job["log_path"]))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            job.update(status="running", pid=process.pid)
            g0.atomic_write_json(record, RUN_RECORD_PATH)
            return_code = int(process.wait())
        if return_code:
            job.update(status="download_failed", return_code=return_code)
            g0.atomic_write_json(record, RUN_RECORD_PATH)
            continue
        merge = g4r.merge_isolated_download(job)
        job.update(
            status=(
                "completed" if merge["merge_status"] == "merged" else "source_incomplete"
            ),
            return_code=return_code,
            merge=merge,
        )
        g0.atomic_write_json(record, RUN_RECORD_PATH)
        g4r.cleanup_isolated_job(isolated)

    _, _, catalog, _ = direct.load_frozen_inputs()
    remaining = missing_event_windows(catalog)
    failures = [job for job in jobs if job["status"] != "completed"]
    record.update(
        {
            "status": (
                "completed"
                if not failures and remaining.empty
                else "completed_with_attrition"
            ),
            "completed_at_utc": g0.utc_now(),
            "completed_jobs": len(jobs) - len(failures),
            "failed_jobs": len(failures),
            "remaining_missing_pair_event_windows": len(remaining),
            "targets_after": [
                artifact(Path(str(path)).resolve())
                for path in intervals["target_file"].unique()
            ],
        }
    )
    g0.atomic_write_json(record, RUN_RECORD_PATH)
    g4r.cleanup_empty_session_root(session_root)
    print(json.dumps(record, indent=2))
    return os.EX_OK if record["status"] == "completed" else 1


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    freeze = prepare(overwrite=args.overwrite)
    if args.prepare_only:
        print(json.dumps(freeze, indent=2))
        return os.EX_OK
    return acquire()


if __name__ == "__main__":
    raise SystemExit(main())
