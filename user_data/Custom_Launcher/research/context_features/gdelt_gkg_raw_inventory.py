from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
import argparse
import json
import re
import sqlite3
import zipfile

from .builder import default_paths
from . import gdelt_gkg_normalized_schema as schema


DEFAULT_EXPORT_RAW_DIR = Path("D:/FreqTradeStuffLargeData/research_news_data/gdelt/raw/export")
DEFAULT_GKG_RAW_DIR = Path("D:/FreqTradeStuffLargeData/research_news_data/gdelt/raw/gkg")

EXPORT_NAME_RE = re.compile(r"^(?P<stamp>\d{14})\.export\.CSV\.zip$", re.IGNORECASE)
GKG_NAME_RE = re.compile(r"^(?P<stamp>\d{14})\.gkg\.csv\.zip$", re.IGNORECASE)


@dataclass(frozen=True)
class RawFileInventory:
    file_kind: str
    gdelt_stamp: str
    local_path: Path
    compressed_bytes: int
    uncompressed_bytes: int | None
    status: str
    parse_error: str | None
    metadata: dict[str, Any]

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["local_path"] = str(self.local_path)
        return payload


@dataclass
class InventorySummary:
    db_path: str
    dry_run: bool
    validate_zip: bool
    export_raw_dir: str
    gkg_raw_dir: str
    start: str | None
    end: str | None
    limit_files: int | None
    directories_scanned: int = 0
    files_seen: int = 0
    files_ignored: int = 0
    files_planned: int = 0
    files_upserted: int = 0
    status_counts: dict[str, int] | None = None
    kind_counts: dict[str, int] | None = None
    first_gdelt_stamp: str | None = None
    last_gdelt_stamp: str | None = None
    files: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["status_counts"] = self.status_counts or {}
        payload["kind_counts"] = self.kind_counts or {}
        payload["files"] = self.files or []
        return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inventory historical-only GDELT export and GKG raw ZIP archives into gdelt_raw_files."
    )
    parser.add_argument("--app-dir", type=Path, default=None)
    parser.add_argument("--db", type=Path, default=None, help="Path to gdelt_context.sqlite.")
    parser.add_argument(
        "--export-raw-dir",
        "--gdelt-raw-dir",
        dest="export_raw_dir",
        type=Path,
        default=DEFAULT_EXPORT_RAW_DIR,
    )
    parser.add_argument("--gkg-raw-dir", type=Path, default=DEFAULT_GKG_RAW_DIR)
    parser.add_argument("--start", default=None, help="Inclusive UTC start stamp/date/time.")
    parser.add_argument("--end", default=None, help="Exclusive UTC end stamp/date/time.")
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-zip", action="store_true")
    args = parser.parse_args()

    if args.limit_files is not None and args.limit_files < 0:
        raise SystemExit("--limit-files must be non-negative")

    start_stamp = parse_filter_stamp(args.start, is_end=False)
    end_stamp = parse_filter_stamp(args.end, is_end=True)
    if start_stamp is not None and end_stamp is not None and end_stamp <= start_stamp:
        raise SystemExit("--end must be after --start")

    paths = default_paths(args.app_dir)
    db_path = args.db or paths.gdelt_db
    summary = inventory_raw_archives(
        db_path=db_path,
        export_raw_dir=args.export_raw_dir,
        gkg_raw_dir=args.gkg_raw_dir,
        start_stamp=start_stamp,
        end_stamp=end_stamp,
        limit_files=args.limit_files,
        dry_run=args.dry_run,
        validate_zip=args.validate_zip,
    )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0


def inventory_raw_archives(
    *,
    db_path: Path,
    export_raw_dir: Path = DEFAULT_EXPORT_RAW_DIR,
    gkg_raw_dir: Path = DEFAULT_GKG_RAW_DIR,
    start_stamp: str | None = None,
    end_stamp: str | None = None,
    limit_files: int | None = None,
    dry_run: bool = False,
    validate_zip: bool = False,
) -> InventorySummary:
    files_seen = 0
    files_ignored = 0
    directories_scanned = 0
    candidates: list[tuple[str, str, Path]] = []

    for raw_dir, expected_kind in ((export_raw_dir, "export"), (gkg_raw_dir, "gkg")):
        if raw_dir.exists():
            directories_scanned += 1
        for path in _iter_zip_candidates(raw_dir):
            files_seen += 1
            inferred = infer_file_identity(path.name)
            if inferred is None or inferred[0] != expected_kind:
                files_ignored += 1
                continue
            file_kind, gdelt_stamp = inferred
            if start_stamp is not None and gdelt_stamp < start_stamp:
                continue
            if end_stamp is not None and gdelt_stamp >= end_stamp:
                continue
            candidates.append((gdelt_stamp, file_kind, path))

    candidates.sort(key=lambda item: (item[0], item[1], str(item[2])))
    if limit_files is not None:
        candidates = candidates[:limit_files]

    entries = [
        build_inventory_entry(path, file_kind, gdelt_stamp, validate_zip=validate_zip)
        for gdelt_stamp, file_kind, path in candidates
    ]

    files_upserted = 0
    if not dry_run:
        files_upserted = upsert_inventory(db_path, entries)

    status_counts = _count_by(entries, "status")
    kind_counts = _count_by(entries, "file_kind")
    stamps = [entry.gdelt_stamp for entry in entries]
    return InventorySummary(
        db_path=str(db_path),
        dry_run=dry_run,
        validate_zip=validate_zip,
        export_raw_dir=str(export_raw_dir),
        gkg_raw_dir=str(gkg_raw_dir),
        start=start_stamp,
        end=end_stamp,
        limit_files=limit_files,
        directories_scanned=directories_scanned,
        files_seen=files_seen,
        files_ignored=files_ignored,
        files_planned=len(entries),
        files_upserted=files_upserted,
        status_counts=status_counts,
        kind_counts=kind_counts,
        first_gdelt_stamp=min(stamps) if stamps else None,
        last_gdelt_stamp=max(stamps) if stamps else None,
        files=[entry.to_json_dict() for entry in entries],
    )


def infer_file_identity(file_name: str) -> tuple[str, str] | None:
    export_match = EXPORT_NAME_RE.fullmatch(file_name)
    if export_match:
        return "export", export_match.group("stamp")
    gkg_match = GKG_NAME_RE.fullmatch(file_name)
    if gkg_match:
        return "gkg", gkg_match.group("stamp")
    return None


def build_inventory_entry(path: Path, file_kind: str, gdelt_stamp: str, *, validate_zip: bool) -> RawFileInventory:
    compressed_bytes = path.stat().st_size
    metadata: dict[str, Any] = {
        "inventory_version": "gdelt_gkg_raw_inventory_v1",
        "file_name_pattern": f"{file_kind}_standard_zip",
        "zip_validation": "not_requested",
    }
    uncompressed_bytes: int | None = None
    status = "inventoried"
    parse_error: str | None = None

    if validate_zip:
        validation = validate_zip_file(path)
        metadata.update(validation["metadata"])
        uncompressed_bytes = validation["uncompressed_bytes"]
        status = validation["status"]
        parse_error = validation["parse_error"]

    return RawFileInventory(
        file_kind=file_kind,
        gdelt_stamp=gdelt_stamp,
        local_path=path,
        compressed_bytes=compressed_bytes,
        uncompressed_bytes=uncompressed_bytes,
        status=status,
        parse_error=parse_error,
        metadata=metadata,
    )


def validate_zip_file(path: Path) -> dict[str, Any]:
    try:
        with zipfile.ZipFile(path) as archive:
            members = [info for info in archive.infolist() if not info.is_dir()]
            if not members:
                return {
                    "status": "zip_empty",
                    "parse_error": "empty zip",
                    "uncompressed_bytes": 0,
                    "metadata": {"zip_validation": "empty", "zip_members": 0},
                }
            bad_member = archive.testzip()
            if bad_member:
                return {
                    "status": "zip_error",
                    "parse_error": f"corrupt zip member: {bad_member}",
                    "uncompressed_bytes": sum(info.file_size for info in members),
                    "metadata": {"zip_validation": "corrupt_member", "zip_members": len(members)},
                }
            return {
                "status": "zip_valid",
                "parse_error": None,
                "uncompressed_bytes": sum(info.file_size for info in members),
                "metadata": {"zip_validation": "ok", "zip_members": len(members)},
            }
    except zipfile.BadZipFile as exc:
        return {
            "status": "zip_error",
            "parse_error": f"bad zip: {exc}",
            "uncompressed_bytes": None,
            "metadata": {"zip_validation": "bad_zip", "zip_members": 0},
        }
    except OSError as exc:
        return {
            "status": "zip_error",
            "parse_error": f"read error: {exc}",
            "uncompressed_bytes": None,
            "metadata": {"zip_validation": "read_error", "zip_members": 0},
        }


def upsert_inventory(db_path: Path, entries: Iterable[RawFileInventory]) -> int:
    schema.init_db(db_path)
    count = 0
    with schema.connect_db(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            for entry in entries:
                schema.upsert_raw_file_metadata(
                    conn,
                    file_kind=entry.file_kind,
                    gdelt_stamp=entry.gdelt_stamp,
                    local_path=entry.local_path,
                    compressed_bytes=entry.compressed_bytes,
                    uncompressed_bytes=entry.uncompressed_bytes,
                    status=entry.status,
                    parse_error=entry.parse_error,
                    metadata=entry.metadata,
                )
                count += 1
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    return count


def parse_filter_stamp(value: str | None, *, is_end: bool) -> str | None:
    if value is None:
        return None
    text = value.strip()
    if not text:
        return None
    if re.fullmatch(r"\d{14}", text):
        return text
    if re.fullmatch(r"\d{8}", text):
        return f"{text}000000"

    clean = text.replace("Z", "+00:00")
    if len(clean) == 10:
        clean = f"{clean}T00:00:00+00:00"
    try:
        parsed = datetime.fromisoformat(clean)
    except ValueError as exc:
        boundary = "end" if is_end else "start"
        raise SystemExit(f"--{boundary} must be YYYYMMDDHHMMSS, YYYYMMDD, or ISO datetime") from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc).replace(microsecond=0)
    return parsed.strftime("%Y%m%d%H%M%S")


def _iter_zip_candidates(raw_dir: Path) -> Iterable[Path]:
    if not raw_dir.exists():
        return []
    return sorted(
        (path for path in raw_dir.iterdir() if path.is_file() and path.name.lower().endswith(".zip")),
        key=lambda path: path.name,
    )


def _count_by(entries: Iterable[RawFileInventory], field_name: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        value = str(getattr(entry, field_name))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    raise SystemExit(main())
