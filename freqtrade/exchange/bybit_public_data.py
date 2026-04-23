"""
Fetch daily Bybit historical order book archives.

Bybit exposes ob500 ZIP archives through a CDN used by their historical data page.
This helper intentionally stores the raw vendor ZIPs; downstream processing can
derive compact feature files without losing the original data.
"""

import asyncio
import logging
from datetime import date, timedelta
from pathlib import Path

import aiohttp

from freqtrade.exceptions import OperationalException
from freqtrade.misc import chunks
from freqtrade.util.datetime_helpers import dt_from_ts, dt_now


logger = logging.getLogger(__name__)

BYBIT_ORDERBOOK_BASE_URL = "https://quote-saver.bycsi.com/orderbook"
SUPPORTED_ORDERBOOK_CATEGORIES = ["linear"]


class BybitArchiveHttpError(Exception):
    pass


def bybit_orderbook_zip_name(symbol: str, day: date, depth: int) -> str:
    return f"{day:%Y-%m-%d}_{symbol}_ob{depth}.data.zip"


def bybit_orderbook_zip_url(category: str, symbol: str, day: date, depth: int) -> str:
    return (
        f"{BYBIT_ORDERBOOK_BASE_URL}/{category}/{symbol}/"
        f"{bybit_orderbook_zip_name(symbol, day, depth)}"
    )


def bybit_orderbook_archive_dir(datadir: Path, category: str, symbol: str) -> Path:
    return datadir / "orderbook" / category / symbol


def bybit_orderbook_availability_filename(
    datadir: Path, category: str, symbol: str, depth: int
) -> Path:
    return bybit_orderbook_archive_dir(datadir, category, symbol) / f"availability_ob{depth}.json"


def list_orderbook_archive_files(
    datadir: Path,
    symbol: str,
    *,
    category: str,
    depth: int,
    since_ms: int | None = None,
    until_ms: int | None = None,
) -> list[Path]:
    archive_dir = bybit_orderbook_archive_dir(datadir, category, symbol)
    if not archive_dir.exists():
        return []

    files = sorted(archive_dir.glob(f"*_ob{depth}.data.zip"))
    if since_ms is None and until_ms is None:
        return files

    start = dt_from_ts(since_ms).date() if since_ms is not None else None
    end = dt_from_ts(until_ms).date() if until_ms is not None else None
    filtered = []
    for path in files:
        try:
            file_date = date.fromisoformat(path.name.split("_", 1)[0])
        except ValueError:
            continue
        if start and file_date < start:
            continue
        if end and file_date > end:
            continue
        filtered.append(path)
    return filtered


def date_range(start: date, end: date):
    day = start
    while day <= end:
        yield day
        day += timedelta(days=1)


async def probe_daily_orderbook_archive(
    symbol: str,
    category: str,
    day: date,
    depth: int,
    session: aiohttp.ClientSession,
    *,
    retry_count: int = 3,
) -> tuple[date, bool]:
    url = bybit_orderbook_zip_url(category, symbol, day, depth)
    retry = 0
    while True:
        try:
            async with session.head(url, allow_redirects=True) as resp:
                if resp.status == 200:
                    return day, True
                if resp.status == 404:
                    return day, False
                if resp.status in (400, 403, 405):
                    async with session.get(
                        url, allow_redirects=True, headers={"Range": "bytes=0-0"}
                    ) as get_resp:
                        if get_resp.status in (200, 206):
                            return day, True
                        if get_resp.status == 404:
                            return day, False
                        raise BybitArchiveHttpError(
                            f"{get_resp.status} - {get_resp.reason}: {url}"
                        )
                raise BybitArchiveHttpError(f"{resp.status} - {resp.reason}: {url}")
        except Exception:
            retry += 1
            if retry > retry_count:
                raise
            await asyncio.sleep(retry)


async def scan_archive_orderbook_availability(
    pair: str,
    symbol: str,
    *,
    category: str,
    depth: int,
    since_ms: int,
    until_ms: int | None,
    concurrency: int = 20,
) -> tuple[str, dict]:
    if category not in SUPPORTED_ORDERBOOK_CATEGORIES:
        raise OperationalException(
            f"Bybit historical order book category '{category}' is not supported. "
            f"Supported categories: {', '.join(SUPPORTED_ORDERBOOK_CATEGORIES)}."
        )
    if depth != 500:
        raise OperationalException("Bybit historical order book downloads currently support ob500.")

    start = dt_from_ts(since_ms).date()
    end_dt = dt_from_ts(until_ms) if until_ms else dt_now() - timedelta(days=1)
    end = min(end_dt.date(), (dt_now() - timedelta(days=1)).date())
    if start > end:
        report = {
            "pair": pair,
            "symbol": symbol,
            "category": category,
            "depth": depth,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "available_dates": [],
            "missing_dates": [],
            "available_count": 0,
            "missing_count": 0,
            "first_available": None,
            "last_available": None,
        }
        return pair, report

    all_days = list(date_range(start, end))
    available_dates: list[str] = []
    missing_dates: list[str] = []

    connector = aiohttp.TCPConnector(limit=concurrency)
    async with aiohttp.ClientSession(connector=connector, trust_env=True) as session:
        for day_chunk in chunks(all_days, concurrency):
            tasks = [
                asyncio.create_task(
                    probe_daily_orderbook_archive(symbol, category, day, depth, session)
                )
                for day in day_chunk
            ]
            for day_result, is_available in await asyncio.gather(*tasks):
                if is_available:
                    available_dates.append(day_result.isoformat())
                else:
                    missing_dates.append(day_result.isoformat())

    report = {
        "pair": pair,
        "symbol": symbol,
        "category": category,
        "depth": depth,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "available_dates": available_dates,
        "missing_dates": missing_dates,
        "available_count": len(available_dates),
        "missing_count": len(missing_dates),
        "first_available": available_dates[0] if available_dates else None,
        "last_available": available_dates[-1] if available_dates else None,
    }
    return pair, report


async def get_daily_orderbook_archive(
    symbol: str,
    category: str,
    day: date,
    depth: int,
    target_dir: Path,
    session: aiohttp.ClientSession,
    *,
    erase: bool = False,
    retry_count: int = 3,
) -> Path | None:
    await asyncio.to_thread(target_dir.mkdir, parents=True, exist_ok=True)
    filename = target_dir / bybit_orderbook_zip_name(symbol, day, depth)
    if await asyncio.to_thread(filename.exists) and not erase:
        logger.info("Bybit order book archive already exists: %s", filename)
        return filename

    url = bybit_orderbook_zip_url(category, symbol, day, depth)
    retry = 0
    while True:
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    await asyncio.to_thread(filename.write_bytes, await resp.read())
                    logger.info("Downloaded Bybit order book archive: %s", filename)
                    return filename
                if resp.status == 404:
                    logger.info("Bybit order book archive not found: %s", url)
                    return None
                raise BybitArchiveHttpError(f"{resp.status} - {resp.reason}: {url}")
        except Exception:
            retry += 1
            if retry > retry_count:
                raise
            await asyncio.sleep(retry)


async def download_archive_orderbook(
    pair: str,
    symbol: str,
    *,
    datadir: Path,
    category: str,
    depth: int,
    since_ms: int,
    until_ms: int | None,
    erase: bool = False,
    available_days: list[date] | None = None,
) -> tuple[str, list[Path]]:
    """
    Download Bybit order book ZIP archives for a single pair.

    :param pair: Freqtrade pair name, used for reporting
    :param symbol: Exchange symbol, e.g. BTCUSDT
    :param datadir: Exchange data directory
    :param category: Bybit archive category. Currently only linear is known.
    :param depth: Archive depth, usually 500 for ob500 files.
    :param since_ms: Start timestamp in milliseconds
    :param until_ms: End timestamp in milliseconds, or None for latest available
    :param erase: Redownload files even when they already exist
    :return: Tuple of pair and downloaded/existing file paths
    """
    if available_days is None:
        _pair, report = await scan_archive_orderbook_availability(
            pair,
            symbol,
            category=category,
            depth=depth,
            since_ms=since_ms,
            until_ms=until_ms,
        )
        available_days = [date.fromisoformat(day) for day in report["available_dates"]]
    if not available_days:
        return pair, []

    target_dir = bybit_orderbook_archive_dir(datadir, category, symbol)
    downloaded: list[Path] = []

    connector = aiohttp.TCPConnector(limit=10)
    async with aiohttp.ClientSession(connector=connector, trust_env=True) as session:
        for day in available_days:
            result = await get_daily_orderbook_archive(
                symbol,
                category,
                day,
                depth,
                target_dir,
                session,
                erase=erase,
            )
            if result is not None:
                downloaded.append(result)

    return pair, downloaded
