import json
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
from pandas.testing import assert_frame_equal

from freqtrade.data.converter import (
    convert_bybit_orderbook_archive_to_features,
    load_orderbook_features,
    orderbook_feature_filename,
    store_orderbook_features,
)
from freqtrade.strategy import merge_orderbook_features
from tests.conftest import generate_test_data


def _write_orderbook_zip(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(path, "w") as zipf:
        filename = path.stem
        content = "\n".join(json.dumps(row) for row in rows) + "\n"
        zipf.writestr(filename, content)


def test_convert_bybit_orderbook_archive_to_features(tmp_path: Path):
    archive = tmp_path / "2024-12-01_XRPUSDT_ob500.data.zip"
    rows = [
        {
            "topic": "orderbook.500.XRPUSDT",
            "type": "snapshot",
            "ts": 1733011200000,
            "cts": 1733011200000,
            "data": {
                "s": "XRPUSDT",
                "b": [["100", "1"], ["99", "2"], ["98", "3"]],
                "a": [["101", "1.5"], ["102", "1"], ["103", "0.5"]],
                "u": 1,
                "seq": 1,
            },
        },
        {
            "topic": "orderbook.500.XRPUSDT",
            "type": "delta",
            "ts": 1733011800000,
            "cts": 1733011800000,
            "data": {
                "s": "XRPUSDT",
                "b": [["100", "2"], ["97", "4"]],
                "a": [["101", "0"], ["102", "2"]],
                "u": 2,
                "seq": 2,
            },
        },
        {
            "topic": "orderbook.500.XRPUSDT",
            "type": "delta",
            "ts": 1733015100000,
            "cts": 1733015100000,
            "data": {
                "s": "XRPUSDT",
                "b": [["101", "1.5"], ["100", "0"]],
                "a": [["102", "1"], ["103", "1.5"]],
                "u": 3,
                "seq": 3,
            },
        },
    ]
    _write_orderbook_zip(archive, rows)

    df = convert_bybit_orderbook_archive_to_features([archive], "1h")

    assert list(df.columns) == [
        "date",
        "mid_open",
        "mid_high",
        "mid_low",
        "mid_close",
        "spread_bps_mean",
        "spread_bps_max",
        "microprice_mean",
        "best_bid_size_mean",
        "best_ask_size_mean",
        "updates",
        "bid_depth_5_mean",
        "ask_depth_5_mean",
        "imbalance_5_mean",
        "bid_depth_25_mean",
        "ask_depth_25_mean",
        "imbalance_25_mean",
        "bid_depth_100_mean",
        "ask_depth_100_mean",
        "imbalance_100_mean",
    ]
    assert len(df) == 2
    assert df.iloc[0]["date"] == pd.Timestamp("2024-12-01 00:00:00+0000", tz="UTC")
    assert df.iloc[0]["updates"] == 2
    assert df.iloc[0]["mid_open"] == 100.5
    assert df.iloc[0]["mid_close"] == 101.0
    assert df.iloc[1]["date"] == pd.Timestamp("2024-12-01 01:00:00+0000", tz="UTC")
    assert df.iloc[1]["updates"] == 1
    assert round(df.iloc[0]["imbalance_5_mean"], 6) == round((1 / 3 + 17 / 27) / 2, 6)


def test_orderbook_feature_storage_roundtrip(tmp_path: Path):
    df = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-12-01 00:00:00+0000", tz="UTC")],
            "mid_open": [100.5],
            "updates": [10],
        }
    )
    filename = store_orderbook_features(tmp_path, "XRP/USDT:USDT", "1h", df)
    assert filename == orderbook_feature_filename(tmp_path, "XRP/USDT:USDT", "1h")
    loaded = load_orderbook_features(tmp_path, "XRP/USDT:USDT", "1h")
    assert_frame_equal(df, loaded)


def test_merge_orderbook_features():
    data = generate_test_data("15m", 20)
    features = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2022-12-31 00:00:00+0000", "2022-12-31 01:00:00+0000"], utc=True
            ),
            "mid_open": [100.0, 101.0],
            "imbalance_5_mean": [0.2, 0.3],
        }
    )

    result = merge_orderbook_features(data, features, "15m", "1h")

    assert "date_ob" in result.columns
    assert "mid_open_ob" in result.columns
    assert "imbalance_5_mean_ob" in result.columns
