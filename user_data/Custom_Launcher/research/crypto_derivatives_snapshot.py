"""Compact public crypto-derivatives context for the existing paper snapshot.

This module has no trading authority and writes no files. Raw exchange responses
are reduced immediately to bounded features with separate observation/fetch times.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from http.client import HTTPException
import json
from math import isfinite
from statistics import median
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "DOGEUSDT", "1000PEPEUSDT")
BINANCE = "https://fapi.binance.com"
BYBIT = "https://api.bybit.com"
DERIBIT = "https://www.deribit.com/api/v2"
BREADTH_24H_COUNT = 40
BREADTH_1H_COUNT = 16
MIN_BREADTH_COVERAGE = 0.75
EXCLUDED_BASES = {
    "USDC", "FDUSD", "TUSD", "USDE", "USDS", "DAI", "PYUSD", "EUR",
    "WBTC", "WETH", "STETH", "WSTETH", "WEETH", "CBETH", "CBBTC", "RETH",
}
CRYPTO_UNDERLYING_TYPES = {"COIN", "CRYPTO"}
NETWORK_ERRORS = (HTTPError, URLError, TimeoutError, ConnectionError, OSError, HTTPException)


def get_json(base: str, path: str, params: dict | None = None, timeout: int = 8):
    url = base + path
    if params:
        url += "?" + urlencode(params)
    request = Request(url, headers={"User-Agent": "Freq-paper-trial/1.0"})
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def safe_get(base: str, path: str, params: dict | None = None):
    try:
        return get_json(base, path, params), None
    except (*NETWORK_ERRORS, json.JSONDecodeError, ValueError, TypeError) as exc:
        return None, f"{type(exc).__name__}: {str(exc)[:160]}"


def _float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if isfinite(value) else None


def _int(value):
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return None


def _iso_ms(value):
    milliseconds = _int(value)
    if milliseconds is None:
        return None
    try:
        return datetime.fromtimestamp(milliseconds / 1000, timezone.utc).isoformat()
    except (OSError, OverflowError, ValueError):
        return None


def _age_minutes(observation_at_utc, now: datetime):
    if not observation_at_utc:
        return None
    try:
        observed = datetime.fromisoformat(observation_at_utc.replace("Z", "+00:00"))
        return (now-observed.astimezone(timezone.utc)).total_seconds()/60.0
    except (TypeError, ValueError):
        return None


def _fresh(observation_at_utc, now: datetime, max_minutes: float) -> bool:
    age = _age_minutes(observation_at_utc, now)
    return age is not None and -2.0 <= age <= max_minutes


def _valid_series_rows(rows, numeric_keys: tuple[str, ...]) -> list[dict]:
    """Return timestamped rows whose fields needed by a metric are actually usable."""
    valid = {}
    if not isinstance(rows, list):
        return []
    for row in rows:
        if not isinstance(row, dict):
            continue
        timestamp = _int(row.get("timestamp"))
        if timestamp is None:
            continue
        values = {key: _float(row.get(key)) for key in numeric_keys}
        if any(value is None or value < 0 for value in values.values()):
            continue
        valid[timestamp] = {"timestamp": timestamp, **values}
    return [valid[key] for key in sorted(valid)]


def _window_meta(rows: list[dict], required_samples: int, interval_minutes: int = 5) -> dict:
    sample_count = min(len(rows), required_samples)
    coverage = sample_count / required_samples if required_samples else 0.0
    selected = rows[-required_samples:] if len(rows) >= required_samples else rows
    continuous = len(selected) == required_samples
    if continuous and len(selected) > 1:
        expected_ms = interval_minutes * 60_000
        gaps = [selected[i]["timestamp"] - selected[i-1]["timestamp"] for i in range(1, len(selected))]
        continuous = all(0.7 * expected_ms <= gap <= 1.3 * expected_ms for gap in gaps)
    return {
        "sample_count": sample_count,
        "required_samples": required_samples,
        "coverage": coverage,
        "complete": bool(continuous),
    }


def _change_window(rows: list[dict], key: str, steps: int) -> tuple[float | None, dict]:
    meta = _window_meta(rows, steps + 1)
    if not meta["complete"]:
        return None, meta
    selected = rows[-(steps+1):]
    previous, latest = selected[0][key], selected[-1][key]
    if previous == 0:
        return None, meta
    return latest / previous - 1.0, meta


def _taker_window(rows: list[dict], count: int) -> tuple[float | None, dict]:
    meta = _window_meta(rows, count)
    if not meta["complete"]:
        return None, meta
    selected = rows[-count:]
    buys = sum(row["buyVol"] for row in selected)
    sells = sum(row["sellVol"] for row in selected)
    total = buys + sells
    return (None if total <= 0 else (buys - sells) / total), meta


def _last_timestamp(rows: list[dict]):
    return _iso_ms(rows[-1]["timestamp"]) if rows else None


def _derivatives_regime(oi_1h, taker_1h):
    if oi_1h is None or taker_1h is None:
        return "unknown"
    if oi_1h >= 0.01 and taker_1h >= 0.05:
        return "oi_expansion_buying"
    if oi_1h >= 0.01 and taker_1h <= -0.05:
        return "oi_expansion_selling"
    if oi_1h <= -0.01:
        return "deleveraging"
    return "mixed"


def _exchange_metadata():
    payload, error = safe_get(BINANCE, "/fapi/v1/exchangeInfo")
    rows = payload.get("symbols") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return {}, error or "invalid exchangeInfo response"
    metadata = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = row.get("symbol")
        if isinstance(symbol, str) and symbol:
            metadata[symbol] = row
    return metadata, error


def _eligible_crypto_perpetual(symbol: str, metadata: dict[str, dict]) -> bool:
    row = metadata.get(symbol)
    if not isinstance(row, dict):
        return False
    base = row.get("baseAsset")
    return (
        row.get("quoteAsset") == "USDT"
        and row.get("contractType") == "PERPETUAL"
        and row.get("status") == "TRADING"
        and row.get("underlyingType") in CRYPTO_UNDERLYING_TYPES
        and isinstance(base, str)
        and base not in EXCLUDED_BASES
    )


def _mad(values: list[float]) -> float:
    centre = median(values)
    return median(abs(value-centre) for value in values)


def _breadth_stats(values: dict[str, float], requested: int, observation_at_utc: str | None, now: datetime) -> dict:
    sample = len(values)
    coverage = 0.0 if requested <= 0 else min(1.0, sample / requested)
    if not values:
        return {
            "status": "unavailable", "sample": 0, "requested": requested,
            "coverage": coverage, "observation_at_utc": observation_at_utc,
            "age_minutes": _age_minutes(observation_at_utc, now),
        }
    returns = list(values.values())
    centre = median(returns)
    share = sum(value > 0 for value in returns) / sample
    return {
        "status": "observed" if coverage >= MIN_BREADTH_COVERAGE else "degraded",
        "sample": sample,
        "requested": requested,
        "coverage": coverage,
        "share_advancing": share,
        "median_pct": centre,
        "dispersion_mad_pct": _mad(returns),
        "btc_pct": values.get("BTCUSDT"),
        "median_minus_btc_pct": None if values.get("BTCUSDT") is None else centre-values["BTCUSDT"],
        "observation_at_utc": observation_at_utc,
        "age_minutes": _age_minutes(observation_at_utc, now),
    }


def _completed_1h_change(symbol: str, now_ms: int):
    rows, error = safe_get(BINANCE, "/fapi/v1/klines", {"symbol": symbol, "interval": "1h", "limit": 4})
    if not isinstance(rows, list):
        return symbol, None, None, error or "invalid kline response"
    completed = []
    for row in rows:
        if not isinstance(row, list) or len(row) <= 6:
            continue
        close_time, close = _int(row[6]), _float(row[4])
        if close_time is None or close is None or close_time >= now_ms:
            continue
        completed.append((close_time, close))
    if len(completed) < 2:
        return symbol, None, None, "insufficient valid completed 1h klines"
    completed.sort()
    if completed[-1][0] - completed[-2][0] != 3_600_000:
        return symbol, None, None, "non-contiguous completed 1h klines"
    if not 0 <= now_ms - completed[-1][0] <= 3_900_000:
        return symbol, None, None, "stale completed 1h klines"
    previous, latest = completed[-2][1], completed[-1][1]
    if previous == 0:
        return symbol, None, None, "invalid completed 1h closes"
    return symbol, 100.0 * (latest / previous - 1.0), completed[-1][0], None


def _premium_map():
    rows, error = safe_get(BINANCE, "/fapi/v1/premiumIndex")
    if not isinstance(rows, list):
        return {}, error or "invalid premium response"
    output = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = row.get("symbol")
        if isinstance(symbol, str) and symbol in SYMBOLS:
            output[symbol] = row
    return output, error


def _binance_symbol(symbol: str, premium: dict | None, now: datetime) -> dict:
    errors = {}
    oi_raw, error = safe_get(BINANCE, "/futures/data/openInterestHist", {"symbol": symbol, "period": "5m", "limit": 13})
    if error:
        errors["open_interest"] = error
    taker_raw, error = safe_get(BINANCE, "/futures/data/takerlongshortRatio", {"symbol": symbol, "period": "5m", "limit": 13})
    if error:
        errors["taker_flow"] = error
    crowd_raw, error = safe_get(BINANCE, "/futures/data/globalLongShortAccountRatio", {"symbol": symbol, "period": "5m", "limit": 13})
    if error:
        errors["positioning"] = error

    oi_rows = _valid_series_rows(oi_raw, ("sumOpenInterest",))
    taker_rows = _valid_series_rows(taker_raw, ("buyVol", "sellVol"))
    crowd_rows = _valid_series_rows(crowd_raw, ("longShortRatio",))

    oi_5m, oi_5m_meta = _change_window(oi_rows, "sumOpenInterest", 1)
    oi_1h, oi_1h_meta = _change_window(oi_rows, "sumOpenInterest", 12)
    taker_5m, taker_5m_meta = _taker_window(taker_rows, 1)
    taker_1h, taker_1h_meta = _taker_window(taker_rows, 12)
    positioning_1h, positioning_1h_meta = _change_window(crowd_rows, "longShortRatio", 12)

    mark = _float(premium.get("markPrice")) if isinstance(premium, dict) else None
    index = _float(premium.get("indexPrice")) if isinstance(premium, dict) else None
    funding = _float(premium.get("lastFundingRate")) if isinstance(premium, dict) else None
    premium_observed = _iso_ms(premium.get("time")) if isinstance(premium, dict) else None
    mark_index_premium = None if mark is None or index in (None, 0.0) else mark/index-1.0
    positioning = crowd_rows[-1]["longShortRatio"] if crowd_rows else None

    freshness = {
        "open_interest": _fresh(_last_timestamp(oi_rows), now, 15),
        "taker_flow": _fresh(_last_timestamp(taker_rows), now, 15),
        "positioning": _fresh(_last_timestamp(crowd_rows), now, 15),
        "premium": _fresh(premium_observed, now, 10),
    }
    # Unavailable and stale are descriptive source states, never neutral market votes.
    if not freshness["open_interest"]:
        oi_5m = oi_1h = None
    if not freshness["taker_flow"]:
        taker_5m = taker_1h = None
    if not freshness["positioning"]:
        positioning = positioning_1h = None
    if not freshness["premium"]:
        funding = mark_index_premium = None
    usable = {
        "oi_change_5m": oi_5m is not None,
        "oi_change_1h": oi_1h is not None,
        "taker_imbalance_5m": taker_5m is not None,
        "taker_imbalance_1h": taker_1h is not None,
        "funding_rate": funding is not None and premium_observed is not None,
        "mark_index_premium": mark_index_premium is not None and premium_observed is not None,
        "global_long_short_ratio": positioning is not None and _last_timestamp(crowd_rows) is not None,
        "global_long_short_ratio_change_1h": positioning_1h is not None,
    }
    usable_count = sum(usable.values())
    advertised_count = len(usable)
    coverage = usable_count / advertised_count
    status = "observed" if usable_count == advertised_count else "degraded" if usable_count else "unavailable"
    partial = [name for name, is_usable in usable.items() if not is_usable]

    return {
        "status": status,
        "coverage": coverage,
        "usable_metric_count": usable_count,
        "advertised_metric_count": advertised_count,
        "oi_change_5m": oi_5m,
        "oi_change_1h": oi_1h,
        "oi_observation_at_utc": _last_timestamp(oi_rows),
        "oi_age_minutes": _age_minutes(_last_timestamp(oi_rows), now),
        "taker_imbalance_5m": taker_5m,
        "taker_imbalance_1h": taker_1h,
        "taker_observation_at_utc": _last_timestamp(taker_rows),
        "taker_age_minutes": _age_minutes(_last_timestamp(taker_rows), now),
        "funding_rate": funding,
        "mark_index_premium": mark_index_premium,
        "premium_observation_at_utc": premium_observed,
        "premium_age_minutes": _age_minutes(premium_observed, now),
        "global_long_short_ratio": positioning,
        "global_long_short_ratio_change_1h": positioning_1h,
        "positioning_observation_at_utc": _last_timestamp(crowd_rows),
        "positioning_age_minutes": _age_minutes(_last_timestamp(crowd_rows), now),
        "fresh_inputs": freshness,
        "fixed_window_coverage": {
            "oi_change_5m": oi_5m_meta,
            "oi_change_1h": oi_1h_meta,
            "taker_imbalance_5m": taker_5m_meta,
            "taker_imbalance_1h": taker_1h_meta,
            "global_long_short_ratio_change_1h": positioning_1h_meta,
        },
        "regime": _derivatives_regime(oi_1h, taker_1h),
        "unavailable_or_partial_metrics": partial,
        "errors": errors,
    }


def _binance_breadth(now: datetime, now_ms: int, metadata: dict[str, dict], metadata_error: str | None) -> dict:
    if not metadata:
        return {"status": "unavailable", "error": metadata_error or "Binance contract metadata unavailable"}
    tickers, error = safe_get(BINANCE, "/fapi/v1/ticker/24hr")
    if not isinstance(tickers, list):
        return {"status": "unavailable", "error": error or "invalid ticker response"}

    eligible = []
    for row in tickers:
        if not isinstance(row, dict):
            continue
        symbol = row.get("symbol")
        if not isinstance(symbol, str) or not _eligible_crypto_perpetual(symbol, metadata):
            continue
        quote_volume = _float(row.get("quoteVolume"))
        if quote_volume is None:
            continue
        eligible.append((quote_volume, row))
    eligible.sort(key=lambda item: item[0], reverse=True)
    eligible_rows = [row for _, row in eligible]

    day_rows = eligible_rows[:BREADTH_24H_COUNT]
    day_values = {}
    day_times = []
    for row in day_rows:
        symbol = row.get("symbol")
        value = _float(row.get("priceChangePercent"))
        close_time = _int(row.get("closeTime"))
        if (isinstance(symbol, str) and value is not None and close_time is not None
                and _fresh(_iso_ms(close_time), now, 10)):
            day_values[symbol] = value
            day_times.append(close_time)

    hour_symbols = [str(row["symbol"]) for row in eligible_rows[:BREADTH_1H_COUNT] if isinstance(row.get("symbol"), str)]
    if "BTCUSDT" not in hour_symbols and _eligible_crypto_perpetual("BTCUSDT", metadata):
        if len(hour_symbols) >= BREADTH_1H_COUNT:
            hour_symbols[-1] = "BTCUSDT"
        else:
            hour_symbols.append("BTCUSDT")
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda symbol: _completed_1h_change(symbol, now_ms), hour_symbols))
    hour_values = {symbol: value for symbol, value, _, _ in rows if value is not None}
    hour_times = [ts for _, _, ts, _ in rows if ts is not None]
    errors = {symbol: err for symbol, _, _, err in rows if err}

    one_hour = _breadth_stats(hour_values, BREADTH_1H_COUNT, _iso_ms(min(hour_times, default=None)), now)
    twenty_four_hour = _breadth_stats(day_values, BREADTH_24H_COUNT, _iso_ms(min(day_times, default=None)), now)
    statuses = {one_hour["status"], twenty_four_hour["status"]}
    status = "observed" if statuses == {"observed"} else "unavailable" if statuses == {"unavailable"} else "degraded"
    return {
        "status": status,
        "universe": "Binance metadata-confirmed trading USDT crypto perpetuals; stable/wrapped bases excluded",
        "eligible_contracts_found": len(eligible_rows),
        "one_hour": one_hour,
        "twenty_four_hour": twenty_four_hour,
        "errors": errors,
    }


def _bybit_crosscheck(now: datetime) -> dict:
    payload, error = safe_get(BYBIT, "/v5/market/tickers", {"category": "linear"})
    result = payload.get("result") if isinstance(payload, dict) else None
    rows = result.get("list", []) if isinstance(result, dict) else []
    if not isinstance(rows, list):
        rows = []
    by_symbol = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("symbol"), str):
            by_symbol[row["symbol"]] = row
    observed_at = _iso_ms(payload.get("time")) if isinstance(payload, dict) else None
    fresh = _fresh(observed_at, now, 10)
    symbols = {}
    for symbol in SYMBOLS:
        row = by_symbol.get(symbol)
        if not row:
            symbols[symbol] = {"status": "unavailable"}
            continue
        mark, index = _float(row.get("markPrice")), _float(row.get("indexPrice"))
        funding = _float(row.get("fundingRate")) if fresh else None
        premium = None if mark is None or index in (None, 0.0) else mark/index-1.0
        if not fresh:
            premium = None
        usable = sum(value is not None for value in (funding, premium))
        symbols[symbol] = {
            "status": "observed" if usable == 2 else "degraded" if usable else "unavailable",
            "funding_rate": funding,
            "mark_index_premium": premium,
        }
    usable_symbols = sum(row["status"] != "unavailable" for row in symbols.values())
    return {
        "status": "observed" if usable_symbols == len(SYMBOLS) else "degraded" if usable_symbols else "unavailable",
        "observation_at_utc": observed_at,
        "age_minutes": _age_minutes(observed_at, now),
        "symbols": symbols,
        "error": error,
    }


def _deribit_dvol(currency: str, now: datetime) -> dict:
    payload, error = safe_get(DERIBIT, "/public/get_volatility_index_data", {
        "currency": currency,
        "start_timestamp": int((now-timedelta(hours=24)).timestamp()*1000),
        "end_timestamp": int(now.timestamp()*1000),
        "resolution": "3600",
    })
    result = payload.get("result") if isinstance(payload, dict) else None
    rows = result.get("data", []) if isinstance(result, dict) else []
    if not isinstance(rows, list):
        rows = []
    for row in reversed(rows):
        if not isinstance(row, list) or len(row) <= 4:
            continue
        timestamp, close = _int(row[0]), _float(row[4])
        if timestamp is None or close is None:
            continue
        observed_at = _iso_ms(timestamp)
        return {
            "status": "observed" if _fresh(observed_at, now, 125) else "stale",
            "close": close,
            "observation_at_utc": observed_at,
            "age_minutes": _age_minutes(observed_at, now),
        }
    return {"status": "unavailable", "error": error or "no valid DVOL rows"}


def collect_crypto_derivatives_snapshot(now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    premium, premium_error = _premium_map()
    metadata, metadata_error = _exchange_metadata()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {symbol: pool.submit(_binance_symbol, symbol, premium.get(symbol), now) for symbol in SYMBOLS}
        symbols = {symbol: future.result() for symbol, future in futures.items()}
    observed = sum(row["status"] == "observed" for row in symbols.values())
    degraded = sum(row["status"] == "degraded" for row in symbols.values())
    status = "observed" if observed == len(SYMBOLS) else "degraded" if observed or degraded else "unavailable"
    return {
        "status": status,
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
        "sources": {"binance": BINANCE, "bybit": BYBIT, "deribit": DERIBIT},
        "binance": {
            "premium_error": premium_error,
            "exchange_info_error": metadata_error,
            "symbols": symbols,
            "breadth": _binance_breadth(now, int(now.timestamp()*1000), metadata, metadata_error),
        },
        "bybit_crosscheck": _bybit_crosscheck(now),
        "deribit_dvol": {currency: _deribit_dvol(currency, now) for currency in ("BTC", "ETH")},
    }


def main() -> int:
    print(json.dumps(collect_crypto_derivatives_snapshot(), indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

