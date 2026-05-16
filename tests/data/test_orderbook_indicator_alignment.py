from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[2]
INDICATOR_PATH = ROOT / "user_data" / "Indicators" / "work_in_progress" / "external_orderbook_context_features.py"


def _load_indicator_module():
    spec = importlib.util.spec_from_file_location("external_orderbook_context_features_test", INDICATOR_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _create_orderbook_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        CREATE TABLE orderbook_metric_bars (
            id INTEGER,
            ts_start TEXT,
            ts_end TEXT,
            timeframe_seconds INTEGER,
            stream_id TEXT,
            market_key TEXT,
            venue TEXT,
            exchange TEXT,
            market_type TEXT,
            margin_type TEXT,
            quote_asset TEXT,
            canonical_pair TEXT,
            pair TEXT,
            symbol TEXT,
            valid_samples INTEGER,
            expected_samples INTEGER,
            spread_bps_mean REAL,
            spread_bps_max REAL,
            microprice_offset_bps_mean REAL,
            imbalance_top20_mean REAL,
            imbalance_top20_min REAL,
            imbalance_top20_max REAL,
            imbalance_10bps_mean REAL,
            imbalance_25bps_mean REAL,
            bid_pressure_seconds REAL,
            ask_pressure_seconds REAL,
            bid_pressure_ratio REAL,
            ask_pressure_ratio REAL,
            max_bid_pressure_streak_seconds REAL,
            max_ask_pressure_streak_seconds REAL,
            nearest_bid_wall_min_distance_bps REAL,
            nearest_ask_wall_min_distance_bps REAL,
            strongest_bid_wall_score REAL,
            strongest_ask_wall_score REAL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE market_context_ticks (
            id INTEGER,
            ts TEXT,
            source_ts TEXT,
            market_key TEXT,
            venue TEXT,
            market_type TEXT,
            margin_type TEXT,
            quote_asset TEXT,
            canonical_pair TEXT,
            symbol TEXT,
            funding_rate REAL,
            open_interest REAL,
            long_ratio REAL,
            short_ratio REAL,
            long_short_ratio REAL,
            taker_buy_volume REAL,
            taker_sell_volume REAL,
            taker_buy_sell_ratio REAL
        )
        """
    )
    conn.executemany(
        "INSERT INTO orderbook_metric_bars VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            _bar_row(1, "2026-01-01T12:00:00+00:00", "2026-01-01T12:30:00+00:00", 10.0, 900.0, 0.0, 5.0, 20.0),
            _bar_row(2, "2026-01-01T12:30:00+00:00", "2026-01-01T13:00:00+00:00", 20.0, 0.0, 900.0, 2.0, 8.0),
            _bar_row(3, "2026-01-01T13:00:00+00:00", "2026-01-01T13:30:00+00:00", 99.0, 1800.0, 0.0, 1.0, 1.0),
        ],
    )
    conn.commit()
    conn.close()


def _bar_row(
    row_id: int,
    ts_start: str,
    ts_end: str,
    spread: float,
    bid_seconds: float,
    ask_seconds: float,
    bid_wall_distance: float,
    ask_wall_distance: float,
) -> tuple[object, ...]:
    return (
        row_id,
        ts_start,
        ts_end,
        1800,
        "stream",
        "binance_usdm_futures",
        "binance",
        "binance",
        "futures",
        "usdm",
        "USDT",
        "BTC/USDT",
        "BTC/USDT:USDT",
        "BTCUSDT",
        1800,
        1800,
        spread,
        spread + 1.0,
        1.0,
        0.5,
        0.2,
        0.8,
        0.4,
        0.6,
        bid_seconds,
        ask_seconds,
        bid_seconds / 1800.0,
        ask_seconds / 1800.0,
        bid_seconds,
        ask_seconds,
        bid_wall_distance,
        ask_wall_distance,
        2.0,
        3.0,
    )


def test_orderbook_indicator_summarizes_closed_window_without_lookahead(tmp_path: Path) -> None:
    indicator = _load_indicator_module()
    db_path = tmp_path / "orderbook.sqlite"
    _create_orderbook_db(db_path)

    frame = pd.DataFrame({"date": pd.to_datetime(["2026-01-01T13:00:00Z"])})
    result = indicator.add_orderbook_context_features(
        frame,
        pair="BTC/USDT:USDT",
        db_path=db_path,
        market_keys=("binance_usdm_futures",),
        primary_market_key="binance_usdm_futures",
        comparison_pairs=(),
        bar_timeframe_seconds=1800,
        include_market_context=False,
        min_coverage_ratio=0.0,
        include_diagnostics=True,
        short_window=2,
        medium_window=2,
        long_window=2,
        persistence_window=2,
    )

    assert result.at[0, "obctx_source_bar_timeframe_seconds"] == 1800.0
    assert result.at[0, "obctx_summary_window_seconds"] == 3600.0
    assert result.at[0, "obctx_summary_lag_seconds"] == 0.0
    assert result.at[0, "obctx_coverage_ratio"] == 1.0
    assert result.at[0, "obctx_ready"] == 1.0
    assert result.at[0, "obctx_spread_bps_mean"] == pytest.approx(15.0)
    assert result.at[0, "obctx_spread_bps_max_roll_medium"] == pytest.approx(21.0)
    assert result.at[0, "obctx_bid_pressure_ratio"] == pytest.approx(0.25)
    assert result.at[0, "obctx_ask_pressure_ratio"] == pytest.approx(0.25)
    assert result.at[0, "obctx_wall_support_score"] == pytest.approx(0.8)
    assert result.at[0, "obctx_bar_age_seconds"] == 0.0


def test_orderbook_indicator_default_output_is_strategy_facing(tmp_path: Path) -> None:
    indicator = _load_indicator_module()
    db_path = tmp_path / "orderbook.sqlite"
    _create_orderbook_db(db_path)

    frame = pd.DataFrame({"date": pd.to_datetime(["2026-01-01T13:00:00Z"])})
    result = indicator.add_orderbook_context_features(
        frame,
        pair="BTC/USDT:USDT",
        db_path=db_path,
        market_keys=("binance_usdm_futures",),
        primary_market_key="binance_usdm_futures",
        comparison_pairs=(),
        bar_timeframe_seconds=1800,
        min_coverage_ratio=0.0,
        short_window=2,
        medium_window=2,
        long_window=2,
        persistence_window=2,
    )

    orderbook_columns = [column for column in result.columns if column.startswith("obctx_")]

    assert orderbook_columns == [f"obctx_{suffix}" for suffix in indicator.STRATEGY_OUTPUT_SUFFIXES]
    assert "obctx_spread_bps" in result.columns
    assert "obctx_risk_score" in result.columns
    assert "obctx_long_context_ok" not in result.columns
    assert "obctx_bid_wall1_price" in result.columns
    assert "obctx_ask_liq_high1_lower_price" in result.columns
    assert "obctx_binance_usdm_futures_spread_bps_mean" not in result.columns
    assert "obctx_spread_bps_mean" not in result.columns
    assert result.at[0, "obctx_ready"] == 1.0
    assert result.at[0, "obctx_quality_score"] == pytest.approx(1.0)


def test_orderbook_indicator_flattens_wall_and_liquidity_blocks(tmp_path: Path) -> None:
    indicator = _load_indicator_module()
    db_path = tmp_path / "orderbook.sqlite"
    _create_orderbook_db(db_path)
    with sqlite3.connect(db_path) as conn:
        for column in indicator.BAR_JSON_COLUMNS:
            conn.execute(f"ALTER TABLE orderbook_metric_bars ADD COLUMN {column} TEXT")
        conn.execute(
            """
            UPDATE orderbook_metric_bars
            SET bid_wall_blocks_json = ?,
                ask_wall_blocks_json = ?,
                bid_liquidity_zones_json = ?,
                ask_liquidity_zones_json = ?
            WHERE id IN (1, 2)
            """,
            (
                json.dumps([
                    {
                        "slot": 1,
                        "price": 79900.0,
                        "distance_bps": 15.0,
                        "score": 8.0,
                        "notional": 120000.0,
                        "persistence": 0.75,
                        "age_seconds": 900.0,
                        "resilience_score": 0.82,
                    }
                ]),
                json.dumps([
                    {
                        "slot": 1,
                        "price": 80150.0,
                        "distance_bps": 18.0,
                        "score": 9.0,
                        "notional": 140000.0,
                        "persistence": 0.65,
                        "age_seconds": 700.0,
                        "resilience_score": 0.78,
                    }
                ]),
                json.dumps([
                    {
                        "kind": "high",
                        "slot": 1,
                        "bucket_index": 1,
                        "lower_price": 79880.0,
                        "upper_price": 79920.0,
                        "mid_price": 79900.0,
                        "distance_bps": 20.0,
                        "score": 2.5,
                        "notional": 100000.0,
                        "persistence": 0.70,
                    }
                ]),
                json.dumps([
                    {
                        "kind": "low",
                        "slot": 1,
                        "bucket_index": 2,
                        "lower_price": 80180.0,
                        "upper_price": 80220.0,
                        "mid_price": 80200.0,
                        "distance_bps": 25.0,
                        "score": 0.2,
                        "notional": 5000.0,
                        "persistence": 0.60,
                    }
                ]),
            ),
        )
        conn.commit()

    frame = pd.DataFrame({"date": pd.to_datetime(["2026-01-01T13:00:00Z"])})
    result = indicator.add_orderbook_context_features(
        frame,
        pair="BTC/USDT:USDT",
        db_path=db_path,
        market_keys=("binance_usdm_futures",),
        primary_market_key="binance_usdm_futures",
        comparison_pairs=(),
        bar_timeframe_seconds=1800,
        min_coverage_ratio=0.0,
        short_window=2,
        medium_window=2,
        long_window=2,
        persistence_window=2,
    )

    assert result.at[0, "obctx_bid_wall1_price"] == pytest.approx(79900.0)
    assert result.at[0, "obctx_bid_wall1_resilience_score"] == pytest.approx(0.82)
    assert result.at[0, "obctx_ask_wall1_score"] == pytest.approx(9.0)
    assert result.at[0, "obctx_bid_liq_high1_mid_price"] == pytest.approx(79900.0)
    assert result.at[0, "obctx_ask_liq_low1_score"] == pytest.approx(0.2)
