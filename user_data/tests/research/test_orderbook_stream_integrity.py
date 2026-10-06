from __future__ import annotations

# ruff: noqa: S101
import json

import pandas as pd

from user_data.Custom_Launcher.collectors.orderbook.collector import (
    BOOK_FRESHNESS_SECONDS,
    _book_is_fresh,
    _invalidate_book_state,
)
from user_data.Custom_Launcher.collectors.orderbook.feature_compaction import (
    RAW_NUMERIC_COLUMNS,
    _market_tick_features,
)
from user_data.Custom_Launcher.collectors.orderbook.markets import MARKET_PROFILES
from user_data.Custom_Launcher.collectors.orderbook.metrics import calculate_orderbook_metrics
from user_data.Custom_Launcher.collectors.orderbook.streams import apply_book_update, parse_stream_message


def test_hourly_features_use_valid_books_and_keep_raw_row_diagnostics() -> None:
    rows = pd.DataFrame({column: [99.0] * 3600 for column in RAW_NUMERIC_COLUMNS})
    rows["ts"] = pd.date_range("2026-10-06", periods=3600, freq="s", tz="UTC")
    rows["book_valid"] = 0
    rows["message_count_interval"] = 99
    for column in (
        "strong_bid_pressure", "strong_ask_pressure", "extreme_bid_pressure", "extreme_ask_pressure"
    ):
        rows[column] = 0
    for side in ("bid", "ask"):
        rows[f"{side}_wall_candidates_json"] = "[]"
        rows[f"{side}_liquidity_zones_json"] = "[]"
    rows.loc[0, ["book_valid", "message_count_interval", "strong_bid_pressure"]] = 1
    rows.loc[0, "spread_bps"] = 2.0
    features = _market_tick_features(rows, "bybit_linear", min_coverage_ratio=0.5)
    prefix = "ob1h_bybit_linear"
    assert features[f"{prefix}_tick_rows"] == 3600
    assert features[f"{prefix}_valid_tick_rows"] == 1
    assert features[f"{prefix}_valid_book_ratio"] == 1 / 3600
    assert features[f"{prefix}_coverage_ratio"] == 1 / 3600
    assert features[f"{prefix}_ready"] == 0
    assert features[f"{prefix}_spread_bps_mean"] == 2
    assert features[f"{prefix}_strong_bid_pressure_ratio"] == 1
    assert features[f"{prefix}_message_count_sum"] == 1

    rows.loc[3599, "book_valid"] = 1
    features = _market_tick_features(rows, "bybit_linear", min_coverage_ratio=0.5)
    assert features[f"{prefix}_max_gap_seconds"] == 3599
    rows["book_valid"] = 0
    features = _market_tick_features(rows, "bybit_linear", min_coverage_ratio=0.5)
    assert features[f"{prefix}_tick_rows"] == 3600
    assert features[f"{prefix}_coverage_ratio"] == features[f"{prefix}_ready"] == 0
    assert pd.isna(features[f"{prefix}_spread_bps_mean"])


def test_stale_book_is_gated_without_requiring_a_bybit_resnapshot() -> None:
    state = {
        "market_key": "bybit_linear",
        "bids": [(100.0, 1.0)],
        "asks": [(101.0, 1.0)],
        "last_book_update_monotonic": 10.0,
        "awaiting_snapshot": False,
    }

    assert _book_is_fresh(state, 10.0 + BOOK_FRESHNESS_SECONDS)
    assert not _book_is_fresh(state, 10.0 + BOOK_FRESHNESS_SECONDS + 0.01)
    assert state["bids"] and state["asks"]
    assert not state["awaiting_snapshot"]

    _invalidate_book_state(state)
    assert state["bids"] == state["asks"] == []
    assert state["awaiting_snapshot"]


def test_bybit_requires_snapshot_and_keeps_subscribed_depth_before_metric_trim() -> None:
    profile = MARKET_PROFILES["bybit_linear"]
    state = {"bids": [], "asks": [], "awaiting_snapshot": True}
    delta = json.dumps({"topic": "orderbook.50.BTCUSDT", "type": "delta", "data": {"s": "BTCUSDT", "b": [["100", "2"]], "a": []}})
    symbol, bids, asks, update_type = parse_stream_message(profile, delta)
    assert symbol == "BTCUSDT"
    assert not apply_book_update(state, "bybit_linear", bids, asks, update_type, 50)
    assert state["bids"] == []

    snapshot_bids = [[str(100 - index), "1"] for index in range(50)]
    snapshot_asks = [[str(101 + index), "1"] for index in range(50)]
    snapshot = json.dumps({"topic": "orderbook.50.BTCUSDT", "type": "snapshot", "data": {"s": "BTCUSDT", "b": snapshot_bids, "a": snapshot_asks}})
    _, bids, asks, update_type = parse_stream_message(profile, snapshot)
    assert apply_book_update(state, "bybit_linear", bids, asks, update_type, 50)
    assert len(state["bids"]) == len(state["asks"]) == 50

    delta_delete = json.dumps({"topic": "orderbook.50.BTCUSDT", "type": "delta", "data": {"s": "BTCUSDT", "b": [["100", "0"]], "a": []}})
    _, bids, asks, update_type = parse_stream_message(profile, delta_delete)
    assert apply_book_update(state, "bybit_linear", bids, asks, update_type, 50)
    assert 100.0 not in {price for price, _ in state["bids"]}
    assert len(state["bids"]) == 49
    metrics = calculate_orderbook_metrics("BTC/USDT:USDT", "BTCUSDT", state["bids"][:20], state["asks"][:20], {"depth_levels": 20}, 1)
    assert metrics["book_valid"] == 1


def test_control_frames_empty_replacement_sides_and_invalid_books() -> None:
    profile = MARKET_PROFILES["bybit_spot"]
    ack = json.dumps({"success": True, "ret_msg": "", "op": "subscribe"})
    assert parse_stream_message(profile, ack) == (None, [], [], "")

    state = {"bids": [(100.0, 1.0)], "asks": [(101.0, 1.0)], "awaiting_snapshot": False}
    snapshot = json.dumps({"topic": "orderbook.50.BTCUSDT", "type": "snapshot", "data": {"s": "BTCUSDT", "b": [], "a": [["101", "1"]]}})
    _, bids, asks, update_type = parse_stream_message(profile, snapshot)
    assert apply_book_update(state, "bybit_spot", bids, asks, update_type, 50)
    assert state["bids"] == []

    crossed = calculate_orderbook_metrics("BTC/USDT", "BTCUSDT", [(101.0, 1.0)], [(101.0, 1.0)], {}, 1)
    nonfinite = calculate_orderbook_metrics("BTC/USDT", "BTCUSDT", [(float("inf"), 1.0)], [(102.0, 1.0)], {}, 1)
    assert crossed["book_valid"] == nonfinite["book_valid"] == 0
    try:
        parse_stream_message(profile, json.dumps({"topic": "orderbook.50.BTCUSDT", "type": "snapshot", "data": {"s": "BTCUSDT", "b": [["NaN", "1"]], "a": [["102", "1"]]}}))
    except ValueError:
        pass
    else:
        raise AssertionError("snapshot containing a nonfinite level was accepted")


def test_bybit_malformed_delta_rejected_instead_of_becoming_empty_update() -> None:
    profile = MARKET_PROFILES["bybit_linear"]
    malformed_levels = [
        [["NaN", "1"]],
        [["100", "Infinity"]],
        [["100", "-1"]],
        [["100"]],
        [["100", "1", "extra"]],
        "not-a-level-list",
    ]
    for levels in malformed_levels:
        message = json.dumps({"topic": "orderbook.50.BTCUSDT", "type": "delta", "data": {"s": "BTCUSDT", "b": levels}})
        try:
            parse_stream_message(profile, message)
        except ValueError:
            pass
        else:
            raise AssertionError(f"malformed delta was accepted: {levels!r}")

    # A valid delta may omit one side or provide an explicitly empty side.
    symbol, bids, asks, update_type = parse_stream_message(
        profile,
        json.dumps({"topic": "orderbook.50.BTCUSDT", "type": "delta", "data": {"s": "BTCUSDT", "b": [["100", "2"]]}}),
    )
    assert (symbol, bids, asks, update_type) == ("BTCUSDT", [(100.0, 2.0)], [], "delta")
    assert parse_stream_message(
        profile,
        json.dumps({"topic": "orderbook.50.BTCUSDT", "type": "delta", "data": {"s": "BTCUSDT", "b": [], "a": []}}),
    ) == ("BTCUSDT", [], [], "delta")


def test_bybit_reset_update_id_is_applied_as_a_fresh_snapshot() -> None:
    profile = MARKET_PROFILES["bybit_linear"]
    state = {"bids": [(99.0, 1.0)], "asks": [(102.0, 1.0)], "awaiting_snapshot": True}
    reset = json.dumps(
        {
            "topic": "orderbook.50.BTCUSDT",
            "type": "delta",
            "data": {"s": "BTCUSDT", "u": 1, "b": [["100", "2"]], "a": [["101", "3"]]},
        }
    )
    _, bids, asks, update_type = parse_stream_message(profile, reset)
    assert update_type == "snapshot"
    assert apply_book_update(state, "bybit_linear", bids, asks, update_type, 50)
    assert state == {"bids": [(100.0, 2.0)], "asks": [(101.0, 3.0)], "awaiting_snapshot": False}

    missing_snapshot_side = json.dumps(
        {"topic": "orderbook.50.BTCUSDT", "type": "snapshot", "data": {"s": "BTCUSDT", "b": []}}
    )
    try:
        parse_stream_message(profile, missing_snapshot_side)
    except ValueError:
        pass
    else:
        raise AssertionError("snapshot missing its ask side was accepted")
