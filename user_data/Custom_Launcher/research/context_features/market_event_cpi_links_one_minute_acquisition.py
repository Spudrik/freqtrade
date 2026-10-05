"""Repair only frozen CPI meme-test BTC one-minute event/control windows."""

from __future__ import annotations

# Bind numerical pools before pandas imports.
# ruff: noqa: E402
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
    market_event_cpi_links_direct as direct,
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
OUTPUT_ROOT = direct.OUTPUT_ROOT / "one_minute_acquisition_20260907a"
INTERVALS_PATH = OUTPUT_ROOT / "cpi_btc_one_minute_intervals.csv"
FREEZE_PATH = OUTPUT_ROOT / "cpi_btc_one_minute_acquisition_freeze.json"
DOWNLOAD_RECORD_PATH = OUTPUT_ROOT / "cpi_btc_one_minute_download_record.json"
TARGET_PAIR = "BTC/USDT:USDT"
WINDOW_MINUTES = 60
MAXIMUM_WORKERS = 1


def artifact(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def load_meme_samples() -> DataFrame:
    _, events = direct.load_frozen_inputs()
    result = json.loads(direct.RESULT_PATH.read_text(encoding="utf-8"))
    if result.get("status") != "completed_cpi_leader_and_meme_link_review":
        raise ValueError("Run the frozen CPI link review before repairing its coverage")
    control_artifact = result.get("artifacts", {}).get("controls", {})
    if Path(
        str(control_artifact.get("path", ""))
    ).resolve() != direct.CONTROL_PATH.resolve() or control_artifact.get(
        "sha256"
    ) != g0.sha256_file(direct.CONTROL_PATH):
        raise ValueError("The CPI control map changed after the direct review")
    controls = pd.read_csv(direct.CONTROL_PATH)
    controls["event_anchor_utc"] = pd.to_datetime(controls["event_anchor_utc"], utc=True)
    controls["control_anchor_utc"] = pd.to_datetime(controls["control_anchor_utc"], utc=True)
    return direct.sample_table(events, controls, eligibility_column="meme_eligible")


def missing_sample_rows(samples: DataFrame, frame: DataFrame) -> DataFrame:
    ordered = frame.sort_values("date", kind="stable").drop_duplicates("date")
    positions = direct.shared.position_by_date(ordered)
    rows: list[dict[str, Any]] = []
    for sample in samples.itertuples(index=False):
        anchor = pd.Timestamp(sample.sample_anchor_utc)
        status, _ = direct.subwindow_metrics(
            ordered,
            positions,
            anchor=anchor,
            start_minute=0,
            end_minute=WINDOW_MINUTES,
        )
        if status == "usable":
            continue
        rows.append(
            {
                "pair": TARGET_PAIR,
                "sample_id": str(sample.sample_id),
                "event_id": str(sample.event_id),
                "sample_type": str(sample.sample_type),
                "control_rank": int(sample.control_rank),
                "prior_coverage_status": status,
                "interval_start_utc": anchor,
                "interval_end_exclusive_utc": anchor + pd.Timedelta(minutes=WINDOW_MINUTES),
            }
        )
    return DataFrame.from_records(rows)


def merge_missing_intervals(missing: DataFrame, target_file: Path) -> DataFrame:
    columns = [
        "pair",
        "sample_ids",
        "event_ids",
        "interval_start_utc",
        "interval_end_exclusive_utc",
        "expected_rows",
        "timerange",
        "target_file",
    ]
    if missing.empty:
        return DataFrame(columns=columns)
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for item in missing.sort_values("interval_start_utc").to_dict(orient="records"):
        start = pd.Timestamp(item["interval_start_utc"])
        end = pd.Timestamp(item["interval_end_exclusive_utc"])
        if current is None or start > current["interval_end_exclusive_utc"]:
            if current is not None:
                rows.append(current)
            current = {
                "pair": TARGET_PAIR,
                "sample_ids": [str(item["sample_id"])],
                "event_ids": [str(item["event_id"])],
                "interval_start_utc": start,
                "interval_end_exclusive_utc": end,
            }
        else:
            current["interval_end_exclusive_utc"] = max(current["interval_end_exclusive_utc"], end)
            current["sample_ids"].append(str(item["sample_id"]))
            current["event_ids"].append(str(item["event_id"]))
    if current is not None:
        rows.append(current)

    output = DataFrame.from_records(rows)
    output["sample_ids"] = output["sample_ids"].map(lambda values: "|".join(values))
    output["event_ids"] = output["event_ids"].map(lambda values: "|".join(dict.fromkeys(values)))
    output["expected_rows"] = (
        (output["interval_end_exclusive_utc"] - output["interval_start_utc"])
        / pd.Timedelta(minutes=1)
    ).astype(int)
    output["timerange"] = [
        f"{int(start.timestamp())}-{int(end.timestamp())}"
        for start, end in zip(
            output["interval_start_utc"],
            output["interval_end_exclusive_utc"],
            strict=True,
        )
    ]
    output["target_file"] = str(target_file.resolve())
    return output.loc[:, columns]


def current_missing() -> tuple[DataFrame, DataFrame, Path]:
    samples = load_meme_samples()
    target = g0.ohlcv_path(TARGET_PAIR, "1m")
    g4r.validate_main_target(target)
    frame = g0.load_ohlcv(target)
    missing = missing_sample_rows(samples, frame)
    return samples, missing, target


def prepare_acquisition(*, overwrite: bool = False) -> dict[str, Any]:
    if FREEZE_PATH.is_file() and not overwrite:
        frozen = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
        if frozen.get("status") != "frozen_before_cpi_btc_one_minute_acquisition":
            raise ValueError("Invalid CPI BTC one-minute acquisition freeze")
        return frozen

    _, missing, target = current_missing()
    intervals = merge_missing_intervals(missing, target)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_csv(intervals, INTERVALS_PATH)
    frozen = {
        "schema_version": 1,
        "status": "frozen_before_cpi_btc_one_minute_acquisition",
        "created_at_utc": g0.utc_now(),
        "outcomes_read_for_selection": False,
        "profit_used": False,
        "target_pair": TARGET_PAIR,
        "window_minutes": WINDOW_MINUTES,
        "missing_frozen_samples": len(missing),
        "merged_download_intervals": len(intervals),
        "selected_workers": MAXIMUM_WORKERS,
        "capacity_reason": "One target pair must merge sequentially; no parallel worker is useful.",
        "artifacts": {
            "intervals": artifact(INTERVALS_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
            "layer2_freeze": artifact(direct.FREEZE_PATH),
            "cpi_events": artifact(direct.EVENTS_PATH),
            "cpi_controls": artifact(direct.CONTROL_PATH),
            "target_before": artifact(target),
        },
    }
    g0.atomic_write_json(frozen, FREEZE_PATH)
    return frozen


def acquire() -> int:
    frozen = prepare_acquisition(overwrite=False)
    if frozen["artifacts"]["analysis_script"]["sha256"] != g0.sha256_file(ANALYSIS_PATH):
        raise ValueError("Acquisition script changed after interval selection was frozen")
    if frozen["artifacts"]["intervals"]["sha256"] != g0.sha256_file(INTERVALS_PATH):
        raise ValueError("Frozen CPI BTC interval list changed")
    if not g4d.CONTROLLER_PYTHON.is_file():
        raise FileNotFoundError(f"Configured controller Python is missing: {g4d.CONTROLLER_PYTHON}")
    if DOWNLOAD_RECORD_PATH.is_file():
        prior = json.loads(DOWNLOAD_RECORD_PATH.read_text(encoding="utf-8"))
        if prior.get("status") == "running":
            raise RuntimeError(f"A CPI BTC acquisition is already running: {DOWNLOAD_RECORD_PATH}")

    intervals = pd.read_csv(INTERVALS_PATH)
    if intervals.empty:
        print("All frozen CPI meme-test BTC one-minute windows already pass coverage.")
        return 0
    started = g0.utc_now()
    session_id = re.sub(r"[^0-9A-Za-z]+", "", started)
    session_root = g4r.ISOLATED_ROOT / "cpi_links" / "cpi_btc_1m_20260907a" / session_id
    g4r.ensure_isolated_path(session_root)
    session_root.mkdir(parents=True, exist_ok=False)
    jobs: list[dict[str, Any]] = []
    for number, row in enumerate(intervals.to_dict(orient="records"), start=1):
        isolated = session_root / f"job_{number:03d}"
        target = Path(str(row["target_file"])).resolve()
        g4r.validate_main_target(target)
        command = g4r.isolated_download_args(str(row["pair"]), str(row["timerange"]), isolated)
        jobs.append(
            {
                "job_id": f"{number:03d}",
                "pair": str(row["pair"]),
                "sample_ids": str(row["sample_ids"]),
                "event_ids": str(row["event_ids"]),
                "timerange": str(row["timerange"]),
                "interval_start_utc": str(row["interval_start_utc"]),
                "interval_end_exclusive_utc": str(row["interval_end_exclusive_utc"]),
                "expected_rows": int(row["expected_rows"]),
                "target_file": str(target),
                "isolated_datadir": str(isolated.resolve()),
                "command": subprocess.list2cmdline(command),
                "log_path": str(
                    (OUTPUT_ROOT / "download_logs" / f"{number:03d}_BTC.log").resolve()
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
        "prelaunch_capacity_check": {
            "logical_processors": os.cpu_count(),
            "reason": "Only one BTC target file is mutated, so jobs run sequentially.",
        },
        "jobs": jobs,
        "source_contracts": {"freeze": artifact(FREEZE_PATH)},
    }
    g0.atomic_write_json(record, DOWNLOAD_RECORD_PATH)

    for job in jobs:
        isolated = Path(str(job["isolated_datadir"]))
        isolated.mkdir(parents=True, exist_ok=False)
        command = g4r.isolated_download_args(str(job["pair"]), str(job["timerange"]), isolated)
        log_path = Path(str(job["log_path"]))
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{g0.utc_now()}] START {subprocess.list2cmdline(command)}\n")
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )
            job.update(status="running", pid=process.pid)
            g0.atomic_write_json(record, DOWNLOAD_RECORD_PATH)
            return_code = int(process.wait())
        if return_code:
            job.update(status="download_failed", return_code=return_code)
            g0.atomic_write_json(record, DOWNLOAD_RECORD_PATH)
            continue
        merge = g4r.merge_isolated_download(job)
        job.update(
            status=("completed" if merge["merge_status"] == "merged" else "source_incomplete"),
            return_code=return_code,
            merge=merge,
        )
        g0.atomic_write_json(record, DOWNLOAD_RECORD_PATH)
        g4r.cleanup_isolated_job(isolated)

    failures = [job for job in jobs if job["status"] != "completed"]
    _, remaining, target = current_missing()
    record.update(
        {
            "status": (
                "completed" if not failures and remaining.empty else "completed_with_failures"
            ),
            "completed_at_utc": g0.utc_now(),
            "completed_jobs": len(jobs) - len(failures),
            "failed_jobs": len(failures),
            "remaining_missing_samples": len(remaining),
            "target_after": artifact(target),
        }
    )
    g0.atomic_write_json(record, DOWNLOAD_RECORD_PATH)
    g4r.cleanup_empty_session_root(session_root)
    return 0 if record["status"] == "completed" else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    frozen = prepare_acquisition(overwrite=args.overwrite)
    if args.prepare_only:
        print(json.dumps(frozen, indent=2))
        return 0
    return acquire()


if __name__ == "__main__":
    raise SystemExit(main())
