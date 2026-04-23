from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from freqtrade.exceptions import OperationalException
from freqtrade.exchange.bybit_public_data import (
    bybit_orderbook_archive_dir,
    bybit_orderbook_availability_filename,
    bybit_orderbook_zip_name,
    bybit_orderbook_zip_url,
    download_archive_orderbook,
    list_orderbook_archive_files,
    scan_archive_orderbook_availability,
)
from freqtrade.util.datetime_helpers import dt_ts


def test_bybit_orderbook_paths(tmp_path: Path):
    day = date(2024, 12, 1)

    assert bybit_orderbook_zip_name("XRPUSDT", day, 500) == (
        "2024-12-01_XRPUSDT_ob500.data.zip"
    )
    assert bybit_orderbook_zip_url("linear", "XRPUSDT", day, 500) == (
        "https://quote-saver.bycsi.com/orderbook/linear/XRPUSDT/"
        "2024-12-01_XRPUSDT_ob500.data.zip"
    )
    assert bybit_orderbook_archive_dir(tmp_path, "linear", "XRPUSDT") == (
        tmp_path / "orderbook" / "linear" / "XRPUSDT"
    )
    assert bybit_orderbook_availability_filename(tmp_path, "linear", "XRPUSDT", 500) == (
        tmp_path / "orderbook" / "linear" / "XRPUSDT" / "availability_ob500.json"
    )


async def test_download_archive_orderbook(mocker, tmp_path: Path):
    async def mock_get_daily(symbol, category, day, depth, target_dir, session, *, erase=False):
        return target_dir / bybit_orderbook_zip_name(symbol, day, depth)

    daily_mock = mocker.patch(
        "freqtrade.exchange.bybit_public_data.get_daily_orderbook_archive",
        side_effect=mock_get_daily,
    )

    pair, files = await download_archive_orderbook(
        "XRP/USDT:USDT",
        "XRPUSDT",
        datadir=tmp_path,
        category="linear",
        depth=500,
        since_ms=dt_ts(datetime(2024, 12, 1, tzinfo=UTC)),
        until_ms=dt_ts(datetime(2024, 12, 3, tzinfo=UTC)),
        available_days=[date(2024, 12, 1), date(2024, 12, 2), date(2024, 12, 3)],
    )

    assert pair == "XRP/USDT:USDT"
    assert daily_mock.call_count == 3
    assert files == [
        tmp_path / "orderbook" / "linear" / "XRPUSDT" / "2024-12-01_XRPUSDT_ob500.data.zip",
        tmp_path / "orderbook" / "linear" / "XRPUSDT" / "2024-12-02_XRPUSDT_ob500.data.zip",
        tmp_path / "orderbook" / "linear" / "XRPUSDT" / "2024-12-03_XRPUSDT_ob500.data.zip",
    ]


async def test_scan_archive_orderbook_availability(mocker):
    async def mock_probe(symbol, category, day, depth, session, *, retry_count=3):
        return day, day.day != 2

    mocker.patch(
        "freqtrade.exchange.bybit_public_data.probe_daily_orderbook_archive",
        side_effect=mock_probe,
    )
    pair, report = await scan_archive_orderbook_availability(
        "XRP/USDT:USDT",
        "XRPUSDT",
        category="linear",
        depth=500,
        since_ms=dt_ts(datetime(2024, 12, 1, tzinfo=UTC)),
        until_ms=dt_ts(datetime(2024, 12, 3, tzinfo=UTC)),
    )

    assert pair == "XRP/USDT:USDT"
    assert report["available_dates"] == ["2024-12-01", "2024-12-03"]
    assert report["missing_dates"] == ["2024-12-02"]
    assert report["available_count"] == 2
    assert report["missing_count"] == 1
    assert report["first_available"] == "2024-12-01"
    assert report["last_available"] == "2024-12-03"


@pytest.mark.parametrize(
    ("category", "depth", "match"),
    [
        ("spot", 500, "category 'spot' is not supported"),
        ("linear", 50, "currently support ob500"),
    ],
)
async def test_download_archive_orderbook_unsupported(tmp_path: Path, category, depth, match):
    with pytest.raises(OperationalException, match=match):
        await download_archive_orderbook(
            "XRP/USDT:USDT",
            "XRPUSDT",
            datadir=tmp_path,
            category=category,
            depth=depth,
            since_ms=dt_ts(datetime(2024, 12, 1, tzinfo=UTC)),
            until_ms=dt_ts(datetime(2024, 12, 1, tzinfo=UTC)),
        )


def test_list_orderbook_archive_files(tmp_path: Path):
    archive_dir = tmp_path / "orderbook" / "linear" / "XRPUSDT"
    archive_dir.mkdir(parents=True)
    for day in ("2024-12-01", "2024-12-02", "2024-12-03"):
        (archive_dir / f"{day}_XRPUSDT_ob500.data.zip").write_bytes(b"x")

    files = list_orderbook_archive_files(
        tmp_path,
        "XRPUSDT",
        category="linear",
        depth=500,
        since_ms=dt_ts(datetime(2024, 12, 2, tzinfo=UTC)),
        until_ms=dt_ts(datetime(2024, 12, 3, tzinfo=UTC)),
    )
    assert [path.name for path in files] == [
        "2024-12-02_XRPUSDT_ob500.data.zip",
        "2024-12-03_XRPUSDT_ob500.data.zip",
    ]
