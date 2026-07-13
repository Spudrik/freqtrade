from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable
import argparse
import json
import subprocess
import sys

from .builder import default_paths


DEFAULT_REPORT_DIR = Path("C:/FreqTradeStuff/user_data/research_news_data/context_features/reports")
DEFAULT_PHASES = ("inventory", "gdelt_events", "gkg_documents", "gold_features", "quality_gates")
MODULE_BY_PHASE = {
    "inventory": "user_data.Custom_Launcher.research.context_features.gdelt_gkg_raw_inventory",
    "gdelt_events": "user_data.Custom_Launcher.research.context_features.gdelt_event_normalize",
    "gkg_documents": "user_data.Custom_Launcher.research.context_features.gkg_document_normalize",
    "gold_features": "user_data.Custom_Launcher.research.context_features.gdelt_gkg_gold_features",
    "quality_gates": "user_data.Custom_Launcher.research.context_features.gdelt_gkg_quality_gates",
}


@dataclass(frozen=True)
class PipelineStep:
    phase: str
    start: str
    end: str
    command: list[str]
    returncode: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plan or execute GDELT/GKG raw inventory, Silver extraction, Gold formatting, and gates in chunks."
    )
    parser.add_argument("--app-dir", type=Path, default=None)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--feature-db", type=Path, default=None)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--chunk-days", type=int, default=30)
    parser.add_argument("--phases", default=",".join(DEFAULT_PHASES))
    parser.add_argument("--execute", action="store_true", help="Run commands. Omitted means plan only.")
    parser.add_argument("--validate-raw-zip", action="store_true")
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--limit-rows", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--tag-prefix", default="gdelt_gkg_silver_pipeline")
    args = parser.parse_args()

    if args.chunk_days <= 0:
        raise SystemExit("--chunk-days must be positive")
    if args.limit_files is not None and args.limit_files < 0:
        raise SystemExit("--limit-files must be non-negative")
    if args.limit_rows is not None and args.limit_rows < 0:
        raise SystemExit("--limit-rows must be non-negative")

    phases = parse_phases(args.phases)
    start = parse_utc(args.start)
    end = parse_utc(args.end)
    if end <= start:
        raise SystemExit("--end must be after --start")

    paths = default_paths(args.app_dir)
    db_path = args.db or paths.gdelt_db
    feature_db = args.feature_db or paths.feature_db
    steps = build_steps(
        python_exe=Path(sys.executable),
        db_path=db_path,
        feature_db=feature_db,
        output_dir=args.output_dir,
        start=start,
        end=end,
        chunk_days=args.chunk_days,
        phases=phases,
        tag_prefix=args.tag_prefix,
        validate_raw_zip=args.validate_raw_zip,
        limit_files=args.limit_files,
        limit_rows=args.limit_rows,
    )

    manifest = {
        "execute": bool(args.execute),
        "db_path": str(db_path),
        "feature_db": str(feature_db),
        "start": start.isoformat(),
        "end": end.isoformat(),
        "chunk_days": args.chunk_days,
        "phases": list(phases),
        "steps_planned": len(steps),
        "steps": [step.to_dict() for step in steps],
    }
    if args.execute:
        completed = execute_steps(steps)
        manifest["steps"] = [step.to_dict() for step in completed]
        manifest["steps_completed"] = sum(1 for step in completed if step.returncode == 0)
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0 if all(step.returncode == 0 for step in completed) else 2

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


def build_steps(
    *,
    python_exe: Path,
    db_path: Path,
    feature_db: Path,
    output_dir: Path,
    start: datetime,
    end: datetime,
    chunk_days: int,
    phases: tuple[str, ...],
    tag_prefix: str,
    validate_raw_zip: bool,
    limit_files: int | None,
    limit_rows: int | None,
) -> list[PipelineStep]:
    steps: list[PipelineStep] = []
    for chunk_start, chunk_end in iter_chunks(start, end, chunk_days):
        for phase in phases:
            command = command_for_phase(
                phase=phase,
                python_exe=python_exe,
                db_path=db_path,
                feature_db=feature_db,
                output_dir=output_dir,
                start=chunk_start,
                end=chunk_end,
                tag=f"{tag_prefix}_{stamp_tag(chunk_start)}_{stamp_tag(chunk_end)}",
                validate_raw_zip=validate_raw_zip,
                limit_files=limit_files,
                limit_rows=limit_rows,
            )
            steps.append(PipelineStep(phase=phase, start=chunk_start.isoformat(), end=chunk_end.isoformat(), command=command))
    return steps


def command_for_phase(
    *,
    phase: str,
    python_exe: Path,
    db_path: Path,
    feature_db: Path,
    output_dir: Path,
    start: datetime,
    end: datetime,
    tag: str,
    validate_raw_zip: bool,
    limit_files: int | None,
    limit_rows: int | None,
) -> list[str]:
    if phase not in MODULE_BY_PHASE:
        raise ValueError(f"unknown phase: {phase}")
    command = [str(python_exe), "-m", MODULE_BY_PHASE[phase], "--start", start.isoformat(), "--end", end.isoformat()]
    if phase in {"inventory", "gdelt_events", "gkg_documents", "gold_features", "quality_gates"}:
        command.extend(["--db", str(db_path)])
    if phase == "inventory":
        if validate_raw_zip:
            command.append("--validate-zip")
        if limit_files is not None:
            command.extend(["--limit-files", str(limit_files)])
    elif phase in {"gdelt_events", "gkg_documents"}:
        if limit_files is not None:
            command.extend(["--limit-files", str(limit_files)])
        if limit_rows is not None:
            command.extend(["--limit-rows", str(limit_rows)])
    elif phase == "gold_features":
        command.extend(["--output-dir", str(output_dir), "--tag", tag])
    elif phase == "quality_gates":
        command.extend(["--feature-db", str(feature_db), "--output-dir", str(output_dir), "--tag", tag])
    return command


def execute_steps(steps: list[PipelineStep]) -> list[PipelineStep]:
    completed: list[PipelineStep] = []
    for step in steps:
        result = subprocess.run(step.command, check=False)
        completed_step = PipelineStep(
            phase=step.phase,
            start=step.start,
            end=step.end,
            command=step.command,
            returncode=int(result.returncode),
        )
        completed.append(completed_step)
        if result.returncode != 0:
            break
    return completed


def parse_phases(value: str) -> tuple[str, ...]:
    phases = tuple(phase.strip() for phase in value.split(",") if phase.strip())
    unknown = [phase for phase in phases if phase not in MODULE_BY_PHASE]
    if unknown:
        raise SystemExit(f"Unknown phase(s): {', '.join(unknown)}")
    return phases


def iter_chunks(start: datetime, end: datetime, chunk_days: int) -> Iterable[tuple[datetime, datetime]]:
    current = start
    step = timedelta(days=chunk_days)
    while current < end:
        chunk_end = min(current + step, end)
        yield current, chunk_end
        current = chunk_end


def parse_utc(value: str) -> datetime:
    clean = value.strip().replace("Z", "+00:00")
    if len(clean) == 10:
        clean = f"{clean}T00:00:00+00:00"
    parsed = datetime.fromisoformat(clean)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).replace(second=0, microsecond=0)


def stamp_tag(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")


if __name__ == "__main__":
    raise SystemExit(main())
