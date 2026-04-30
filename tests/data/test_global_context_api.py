from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LAUNCHER_DIR = ROOT / "user_data" / "Custom_Launcher"
if str(LAUNCHER_DIR) not in sys.path:
    sys.path.insert(0, str(LAUNCHER_DIR))

from launcher_v2.services.global_context_service import GlobalContextService
from research.collectors.global_context_collector import (
    normalize_coingecko_global,
    normalize_coingecko_markets,
    normalize_defillama_chains,
    normalize_defillama_stablecoins,
    normalize_fear_greed,
)
from research.collectors.global_context_store import (
    connect_db,
    fetch_latest_context_rows,
    fetch_source_health,
    init_db,
    insert_context_tick,
    upsert_source_status,
)


def test_global_context_normalizers_score_and_note_payloads() -> None:
    fear = normalize_fear_greed(
        _source("alternative_fear_greed", "alternative_fng", "sentiment"),
        {"data": [{"value": "72", "value_classification": "Greed", "timestamp": "1767052800"}]},
        store_raw=False,
    )
    assert fear["score"] == 72.0
    assert fear["signal"] == "risk_on"
    assert fear["value"] == 72.0
    assert fear["source_ts"]

    global_row = normalize_coingecko_global(
        _source("coingecko_global", "coingecko_global", "market"),
        {
            "data": {
                "total_market_cap": {"usd": 3_100_000_000_000},
                "total_volume": {"usd": 95_000_000_000},
                "market_cap_percentage": {"btc": 51.2},
                "market_cap_change_percentage_24h_usd": 2.0,
            }
        },
        store_raw=False,
    )
    assert global_row["score"] == 60.0
    assert "BTC dom 51.20%" in global_row["notes"]

    markets = normalize_coingecko_markets(
        _source("coingecko_btc_eth", "coingecko_markets", "market"),
        [
            {"symbol": "btc", "current_price": 100000, "price_change_percentage_24h": 1.0, "price_change_percentage_7d_in_currency": 5.0},
            {"symbol": "eth", "current_price": 4000, "price_change_percentage_24h": 3.0, "price_change_percentage_7d_in_currency": 1.0},
        ],
        store_raw=False,
    )
    assert markets["value"] == 2.0
    assert markets["score"] == 62.5
    assert "BTC" in markets["notes"]


def test_defillama_normalizers_build_liquidity_context() -> None:
    stablecoins = normalize_defillama_stablecoins(
        _source("defillama_stablecoins", "defillama_stablecoins", "defi"),
        {
            "peggedAssets": [
                {
                    "symbol": "USDT",
                    "circulating": {"peggedUSD": 100.0},
                    "circulatingPrevDay": {"peggedUSD": 99.0},
                    "circulatingPrevWeek": {"peggedUSD": 95.0},
                    "circulatingPrevMonth": {"peggedUSD": 90.0},
                },
                {
                    "symbol": "USDC",
                    "circulating": {"peggedUSD": 50.0},
                    "circulatingPrevDay": {"peggedUSD": 50.0},
                    "circulatingPrevWeek": {"peggedUSD": 50.0},
                    "circulatingPrevMonth": {"peggedUSD": 45.0},
                },
            ]
        },
        store_raw=False,
    )
    assert stablecoins["metric_key"] == "stablecoin_supply_change_7d"
    assert round(float(stablecoins["value"]), 6) == round((150.0 - 145.0) / 145.0 * 100.0, 6)
    assert "supply" in stablecoins["notes"]

    chains = normalize_defillama_chains(
        _source("defillama_chains", "defillama_chains", "defi"),
        [
            {"name": "Ethereum", "tvl": 1000.0, "change_1d": 1.0, "change_7d": 2.0},
            {"name": "Solana", "tvl": 500.0, "change_1d": -1.0, "change_7d": 4.0},
        ],
        store_raw=False,
    )
    assert chains["metric_key"] == "defi_tvl_weighted_change_7d"
    assert round(float(chains["value"]), 6) == round(((1000.0 * 2.0) + (500.0 * 4.0)) / 1500.0, 6)
    assert "Ethereum" in chains["notes"]


def test_store_keeps_latest_success_when_later_failure_occurs(tmp_path: Path) -> None:
    db_path = tmp_path / "global_context.sqlite"
    init_db(db_path)
    source = _source("alternative_fear_greed", "alternative_fng", "sentiment")
    row = normalize_fear_greed(source, {"data": [{"value": "25", "value_classification": "Fear"}]}, store_raw=False)
    with connect_db(db_path) as conn:
        insert_context_tick(conn, row)
        upsert_source_status(conn, _status_payload(source, row=row))
        upsert_source_status(conn, _status_payload(source, error="temporary API failure"))
        conn.commit()
        latest = fetch_latest_context_rows(conn)
        health = fetch_source_health(conn)

    assert latest[0]["source_id"] == "alternative_fear_greed"
    assert health[0]["last_score"] == 25.0
    assert health[0]["last_error"] == "temporary API failure"
    assert health[0]["last_success_at"] is not None


def test_service_command_and_latest_rows(tmp_path: Path) -> None:
    service = GlobalContextService(LAUNCHER_DIR, sys.executable)
    state = {
        "data_dir": str(tmp_path),
        "db_path": str(tmp_path / "global_context.sqlite"),
        "interval_minutes": "30",
        "once": True,
    }
    command = service.build_command(state)
    assert command[:4] == [sys.executable, "-u", "-m", "research.collectors.global_context_collector"]
    assert "--once" in command

    db_path = service.paths(state)["db"]
    init_db(db_path)
    with connect_db(db_path) as conn:
        insert_context_tick(
            conn,
            {
                "ts": "2026-04-30T00:00:00+00:00",
                "source_id": "coingecko_global",
                "source_group": "market",
                "source_type": "coingecko_global",
                "metric_key": "global_market_cap_change_24h",
                "score": 55.0,
                "signal": "neutral",
                "value": 1.0,
                "unit": "percent",
                "notes": "test row",
            },
        )
        conn.commit()

    rows = service.latest_context_rows(state)
    assert rows[0][:7] == ("coingecko_global", "market", "neutral", "55", "1", "percent", "test row")


def _source(source_id: str, source_type: str, group: str) -> dict[str, object]:
    return {
        "id": source_id,
        "type": source_type,
        "source_group": group,
        "market_relevance": "medium",
        "url": "https://example.test",
    }


def _status_payload(source: dict[str, object], row: dict[str, object] | None = None, error: str | None = None) -> dict[str, object | None]:
    payload: dict[str, object | None] = {
        "source_id": source["id"],
        "source_group": source["source_group"],
        "source_type": source["type"],
        "enabled": 1,
        "market_relevance": source["market_relevance"],
        "url": source["url"],
        "last_success_at": None,
        "last_failure_at": "2026-04-30T01:00:00+00:00" if error else None,
        "last_error": error,
        "last_score": None,
        "last_signal": None,
        "last_value": None,
        "last_unit": None,
        "last_notes": None,
        "updated_at": "2026-04-30T00:00:00+00:00",
    }
    if row:
        payload.update(
            {
                "last_success_at": row["ts"],
                "last_score": row["score"],
                "last_signal": row["signal"],
                "last_value": row["value"],
                "last_unit": row["unit"],
                "last_notes": row["notes"],
            }
        )
    return payload
