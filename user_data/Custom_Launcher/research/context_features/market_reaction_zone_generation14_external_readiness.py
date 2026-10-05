"""Refresh Generation 14 historical news/orderbook readiness without reading outcomes."""

from __future__ import annotations

# ruff: noqa: E402
import os


for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYARROW_NUM_THREADS",
):
    os.environ[_name] = "1"

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
    market_reaction_zone_generation12_freeze as g12z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation13_cache as g13c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation14_freeze as g14z,
)


DEFAULT_RUN_ID = "g14_external_readiness_20260822a"
RECORD_ROOT = (
    g14z.FREEZE_PATH.parent / "g14_broad_combinations" / "external_readiness"
)
SOURCE_COLUMNS = {
    "historical_news": "ready__g11_news_context",
    "historical_orderbook": "ready__g11_orderbook_context",
}


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


def analysis_windows(frame: DataFrame, cohort: str) -> DataFrame:
    output: list[DataFrame] = []
    standard_periods = set(
        g14z.freeze_generation14()["evaluation_windows"][g14z.STANDARD][cohort]
    )
    standard = frame.loc[frame["period"].isin(standard_periods)].copy()
    if not standard.empty:
        standard["window"] = g14z.STANDARD
        standard["analysis_period"] = standard["period"].astype(str)
        output.append(standard)
    if cohort == "normal":
        for definition in g12z.RECENT_PERIODS:
            start = pd.Timestamp(definition["start"])
            stop = pd.Timestamp(definition["stop"])
            selected = frame.loc[
                frame["date"].ge(start) & frame["date"].lt(stop)
            ].copy()
            if selected.empty:
                continue
            selected["window"] = g14z.RECENT
            selected["analysis_period"] = definition["period"]
            output.append(selected)
    if not output:
        return DataFrame(columns=[*frame.columns, "window", "analysis_period"])
    return pd.concat(output, ignore_index=True, sort=False)


def load_readiness(cohort: str) -> tuple[DataFrame, Path]:
    frozen = g14z.freeze_generation14()
    manifest_path = g13c.RECORD_ROOT / f"{cohort}_manifest.json"
    expected = frozen["source_contracts"][f"generation13_{cohort}_cache"]
    if g0.sha256_file(manifest_path) != expected["sha256"]:
        raise ValueError(f"Generation 13 {cohort} cache changed after G14 freeze.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    frames: list[DataFrame] = []
    for item in manifest["inventory"]:
        path = Path(item["mtf_event_path"])
        if g0.sha256_file(path) != item["mtf_event_sha256"]:
            raise ValueError(f"Generation 13 event cache changed: {path}")
        frame = pd.read_parquet(
            path,
            columns=["date", "period", *SOURCE_COLUMNS.values()],
        )
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        frame["pair"] = str(item["pair"])
        frames.append(frame)
    return analysis_windows(pd.concat(frames, ignore_index=True), cohort), manifest_path


def summarize(cohort: str, frame: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for (window, period), cell in frame.groupby(
        ["window", "analysis_period"], observed=True, sort=False
    ):
        for source, column in SOURCE_COLUMNS.items():
            ready = cell.loc[cell[column].fillna(False).astype(bool)]
            counts = ready.groupby("pair", observed=True).size()
            rows.append(
                {
                    "cohort": cohort,
                    "window": window,
                    "analysis_period": period,
                    "source": source,
                    "eligible_rows": len(ready),
                    "eligible_coins": int((counts > 0).sum()),
                    "minimum_ready_pair_rows": int(counts.min()) if len(counts) else 0,
                    "model_gate_passed": bool(len(ready) >= 50 and len(counts) >= 5),
                }
            )
    return DataFrame.from_records(rows)


def run_readiness(run_id: str = DEFAULT_RUN_ID) -> dict[str, Any]:
    frames: list[DataFrame] = []
    sources: dict[str, Any] = {"generation14_freeze": artifact(g14z.FREEZE_PATH)}
    for cohort in g14z.COHORTS:
        frame, manifest_path = load_readiness(cohort)
        frames.append(summarize(cohort, frame))
        sources[f"generation13_{cohort}_cache"] = artifact(manifest_path)
    summary = pd.concat(frames, ignore_index=True)
    source_decisions: list[dict[str, Any]] = []
    for source, cell in summary.groupby("source", observed=True, sort=False):
        all_passed = bool(cell["model_gate_passed"].astype(bool).all())
        source_decisions.append(
            {
                "source": source,
                "status": "supported_for_frozen_model_ladder" if all_passed else "parked_coverage",
                "all_declared_periods_passed": all_passed,
                "minimum_eligible_rows": int(cell["eligible_rows"].min()),
                "minimum_eligible_coins": int(cell["eligible_coins"].min()),
            }
        )
    record_dir = RECORD_ROOT / run_id
    csv_path = record_dir / "external_readiness.csv"
    g0.atomic_write_csv(summary, csv_path)
    result = {
        "schema_version": 1,
        "generation": 14,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation14_external_readiness",
        "reaction_outcomes_read": False,
        "future_signed_direction_read": False,
        "model_gate": "50 eligible rows and five coins in every declared period",
        "decisions": source_decisions,
        "source_contracts": sources,
        "summary": artifact(csv_path),
    }
    result_path = record_dir / "g14_external_readiness_result.json"
    g0.atomic_write_json(result, result_path)
    return {**result, "result_path": str(result_path.resolve())}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit Generation 14 external readiness.")
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    args = parser.parse_args(argv)
    print(json.dumps(run_readiness(args.run_id), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
