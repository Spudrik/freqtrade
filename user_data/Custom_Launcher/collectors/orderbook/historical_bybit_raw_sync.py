from __future__ import annotations

import argparse
import json
import re
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests


APP_DIR = Path(__file__).resolve().parents[2]
USER_DATA_DIR = APP_DIR.parent
DEFAULT_RAW_DIR = Path("D:/FreqTradeStuffLargeData/orderbook_data/historical_bybit/spot_raw")
DEFAULT_BASE_URL = "https://quote-saver.bycsi.com/orderbook/spot"
DEFAULT_MANIFEST = USER_DATA_DIR / "orderbook_data" / "historical_bybit" / "bybit_ob200_raw_manifest.json"


@dataclass(frozen=True)
class ArchiveItem:
    name: str
    url: str
    day: date


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync free Bybit spot OB200 archive ZIPs into the local raw cache.")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--depth-pattern", default="ob200", help="Archive depth token regex, for example ob200 or ob\\d+.")
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--start-date", default="", help="Inclusive UTC date, YYYY-MM-DD. Defaults to first listed archive.")
    parser.add_argument("--end-date", default="", help="Inclusive UTC date, YYYY-MM-DD. Defaults to last listed archive.")
    parser.add_argument("--max-files", type=int, default=0, help="Optional cap for one sync run. 0 means no cap.")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    summary = sync_archives(
        symbol=args.symbol.upper(),
        raw_dir=args.raw_dir,
        base_url=args.base_url.rstrip("/"),
        manifest_path=args.manifest,
        depth_pattern=args.depth_pattern,
        start=date.fromisoformat(args.start_date) if args.start_date else None,
        end=date.fromisoformat(args.end_date) if args.end_date else None,
        max_files=max(0, int(args.max_files)),
        retries=max(1, int(args.retries)),
        dry_run=args.dry_run,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def sync_archives(
    *,
    symbol: str,
    raw_dir: Path,
    base_url: str,
    manifest_path: Path,
    depth_pattern: str,
    start: date | None,
    end: date | None,
    max_files: int,
    retries: int,
    dry_run: bool,
) -> dict[str, Any]:
    items = _list_available_archives(base_url, symbol, depth_pattern)
    if start is not None:
        items = [item for item in items if item.day >= start]
    if end is not None:
        items = [item for item in items if item.day <= end]

    raw_dir.mkdir(parents=True, exist_ok=True)
    existing = {path.name for path in raw_dir.glob(f"*_{symbol}_*.data.zip")}
    missing = [item for item in items if item.name not in existing]
    if max_files:
        missing = missing[:max_files]

    summary: dict[str, Any] = {
        "symbol": symbol,
        "base_url": base_url,
        "raw_dir": str(raw_dir),
        "manifest": str(manifest_path),
        "depth_pattern": depth_pattern,
        "dry_run": dry_run,
        "available_files": len(items),
        "available_start": items[0].day.isoformat() if items else None,
        "available_end": items[-1].day.isoformat() if items else None,
        "existing_files": len(existing),
        "missing_files_planned": len(missing),
        "downloaded_files": 0,
        "downloaded_bytes": 0,
        "failed": [],
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    if dry_run:
        summary["planned_first"] = missing[0].name if missing else None
        summary["planned_last"] = missing[-1].name if missing else None
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        return summary

    for item in missing:
        output = raw_dir / item.name
        try:
            print(f"downloading {item.name}", flush=True)
            bytes_written = _download_archive(item.url, output, retries=retries)
            _validate_zip(output)
        except Exception as exc:
            output.unlink(missing_ok=True)
            output.with_suffix(output.suffix + ".tmp").unlink(missing_ok=True)
            summary["failed"].append({"name": item.name, "error": str(exc)})
            continue
        summary["downloaded_files"] += 1
        summary["downloaded_bytes"] += bytes_written
        print(f"downloaded {item.name} {bytes_written}", flush=True)

    summary["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        **summary,
        "local_files": sorted(path.name for path in raw_dir.glob(f"*_{symbol}_*.data.zip")),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _list_available_archives(base_url: str, symbol: str, depth_pattern: str) -> list[ArchiveItem]:
    listing_url = f"{base_url.rstrip('/')}/{symbol}/"
    response = requests.get(listing_url, timeout=60)
    response.raise_for_status()
    pattern = re.compile(rf'(\d{{4}}-\d{{2}}-\d{{2}}_{re.escape(symbol)}_{depth_pattern}\.data\.zip)')
    items: list[ArchiveItem] = []
    for name in sorted(set(pattern.findall(response.text))):
        items.append(
            ArchiveItem(
                name=name,
                url=urljoin(listing_url, name),
                day=date.fromisoformat(name[:10]),
            )
        )
    return items


def _download_archive(url: str, output: Path, *, retries: int) -> int:
    temp = output.with_suffix(output.suffix + ".tmp")
    last_error: Exception | None = None
    for _attempt in range(retries):
        bytes_written = 0
        temp.unlink(missing_ok=True)
        try:
            with requests.get(url, stream=True, timeout=(15, 60)) as response:
                response.raise_for_status()
                with temp.open("wb") as handle:
                    for chunk in response.iter_content(1024 * 1024):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        bytes_written += len(chunk)
            temp.replace(output)
            return bytes_written
        except Exception as exc:
            last_error = exc
            temp.unlink(missing_ok=True)
    if last_error:
        raise last_error
    raise RuntimeError(f"Download failed for {url}")


def _validate_zip(path: Path) -> None:
    with zipfile.ZipFile(path) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"ZIP validation failed at member {bad}")
        if not archive.namelist():
            raise ValueError("ZIP archive is empty")


if __name__ == "__main__":
    raise SystemExit(main())
