from __future__ import annotations

import math
from typing import Any
from urllib.parse import urlencode

from .common import (
    clamp,
    compact_usd,
    fetch_public_json,
    float_or_none,
    fmt_pct,
    mean,
    nested_float,
    result,
    score_label,
    score_signal,
    unix_to_iso,
)


def fetch_coingecko_markets(source: dict[str, Any], config: dict[str, Any]) -> Any:
    url = str(source.get("url") or "").strip()
    params = {
        "vs_currency": str(source.get("vs_currency") or "usd"),
        "ids": ",".join(str(value) for value in source.get("coin_ids") or ["bitcoin", "ethereum"]),
        "sparkline": "false",
        "price_change_percentage": "1h,24h,7d",
    }
    separator = "&" if "?" in url else "?"
    return fetch_public_json(
        url + separator + urlencode(params),
        timeout_seconds=int(config.get("request_timeout_seconds", 20)),
        user_agent=str(config.get("user_agent") or "FreQ-GlobalContextCollector/1.0"),
    )


def fetch_deribit_options(source: dict[str, Any], config: dict[str, Any]) -> Any:
    url = str(source.get("url") or "").strip()
    params = {"currency": str(source.get("currency") or "BTC").upper(), "kind": "option"}
    separator = "&" if "?" in url else "?"
    return fetch_public_json(
        url + separator + urlencode(params),
        timeout_seconds=int(config.get("request_timeout_seconds", 20)),
        user_agent=str(config.get("user_agent") or "FreQ-GlobalContextCollector/1.0"),
    )


def normalize_deribit_options(
    source: dict[str, Any], payload: Any, *, store_raw: bool = True
) -> list[dict[str, Any]]:
    from datetime import datetime, timedelta, timezone

    envelope = payload if isinstance(payload, dict) else {}
    rows = envelope.get("result")
    if envelope.get("error") or not isinstance(rows, list):
        raise ValueError("Malformed Deribit options response")
    now = datetime.now(timezone.utc)
    max_age = int(source.get("max_age_seconds", 1800))
    max_clock_gap = int(source.get("max_clock_gap_seconds", 300))
    currency = str(source.get("currency") or "BTC").upper()
    groups: dict[str, list[dict[str, Any]]] = {}
    seen_instruments: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Malformed Deribit options instrument row {index}: expected an object")
        instrument_name = row.get("instrument_name")
        if not isinstance(instrument_name, str) or not instrument_name.strip():
            raise ValueError(f"Malformed Deribit options instrument row {index}: missing instrument_name")
        normalized_name = instrument_name.strip().upper()
        parts = normalized_name.split("-")
        if len(parts) != 4 or parts[0] != currency:
            raise ValueError(f"Malformed Deribit options instrument row {index}: invalid instrument_name")
        try:
            expiry = datetime.strptime(parts[1], "%d%b%y").date().isoformat()
        except ValueError:
            raise ValueError(f"Malformed Deribit options instrument row {index}: invalid expiry") from None
        strike = _finite_number(parts[2])
        if strike is None or strike <= 0:
            raise ValueError(f"Malformed Deribit options instrument row {index}: invalid strike")
        side = parts[3]
        if side not in {"C", "P"}:
            raise ValueError(f"Malformed Deribit options instrument row {index}: invalid option side")
        if normalized_name in seen_instruments:
            raise ValueError(f"Duplicate Deribit options instrument row: {instrument_name.strip()}")
        seen_instruments.add(normalized_name)
        item = dict(row)
        item["_side"] = side
        item["_clock"] = _deribit_clock(row.get("creation_timestamp"))
        groups.setdefault(expiry, []).append(item)

    output: list[dict[str, Any]] = []
    for expiry in sorted(groups)[: int(source.get("max_expiries", 6))]:
        expiry_rows = groups[expiry]
        side_values: dict[str, dict[str, Any]] = {}
        for side, label in (("C", "call"), ("P", "put")):
            selected = [row for row in expiry_rows if row["_side"] == side]
            clocks = [clock for clock in (row["_clock"] for row in selected) if clock is not None]
            clock = max(clocks) if clocks else None
            age_seconds = (now - clock).total_seconds() if clock else None
            fresh = (
                bool(selected)
                and len(clocks) == len(selected)
                and age_seconds is not None
                and all(0 <= (now - item).total_seconds() <= max_age for item in clocks)
            )
            if clocks and (max(clocks) - min(clocks)).total_seconds() > max_clock_gap:
                fresh = False
            open_interest = [_finite_number(row.get("open_interest")) for row in selected]
            volume = [_finite_number(row.get("volume")) for row in selected]
            oi_complete = bool(selected) and all(
                value is not None and value >= 0 for value in open_interest
            )
            volume_complete = bool(selected) and all(
                value is not None and value >= 0 for value in volume
            )
            oi_total = _finite_number(sum(open_interest)) if fresh and oi_complete else None
            volume_total = _finite_number(sum(volume)) if fresh and volume_complete else None
            side_values[label] = {
                "oi": oi_total,
                "volume": volume_total,
                "clock": clock.isoformat() if clock else None,
                "fresh": fresh,
                "oi_complete": oi_complete and oi_total is not None,
                "volume_complete": volume_complete and volume_total is not None,
                "unit": currency,
            }
            for field, metric, description in (
                ("oi", "open_interest", "Open interest"),
                ("volume", "volume_24h", "24-hour volume"),
            ):
                entry = side_values[label]
                output.append(
                    result(
                        source,
                        metric_key=f"deribit_option_{metric}_{label}_{expiry}",
                        score=None,
                        signal="",
                        value=entry[field],
                        unit=currency,
                        notes=(
                            f"{description} summed across {label}s for {currency} expiry {expiry}; "
                            "values are in underlying coin units. Positioning/hedging context only, "
                            "not directional by itself."
                            if entry["fresh"] and entry[f"{field}_complete"]
                            else f"Unknown: incomplete, stale, or clock-inconsistent {label} observations "
                            f"for {currency} expiry {expiry}."
                        ),
                        source_ts=entry["clock"],
                        raw=payload if store_raw else None,
                    )
                )
        for metric, field in (("open_interest", "oi"), ("volume_24h", "volume")):
            calls, puts = side_values["call"], side_values["put"]
            aligned = (
                calls["fresh"]
                and puts["fresh"]
                and calls[f"{field}_complete"]
                and puts[f"{field}_complete"]
            )
            clocks = [
                datetime.fromisoformat(item["clock"]) for item in (calls, puts) if item["clock"]
            ]
            aligned = (
                aligned
                and len(clocks) == 2
                and (max(clocks) - min(clocks)) <= timedelta(seconds=max_clock_gap)
            )
            denominator = calls[field]
            ratio = (
                _finite_number(puts[field] / denominator)
                if aligned and denominator is not None and denominator > 0
                else None
            )
            output.append(
                result(
                    source,
                    metric_key=f"deribit_option_put_call_{metric}_ratio_{expiry}",
                    score=None,
                    signal="",
                    value=ratio,
                    unit="ratio",
                    notes=(
                        f"Put/call {metric.replace('_', ' ')} ratio for matched {currency} expiry {expiry}; "
                        "descriptive positioning/hedging context, not a bullish/bearish verdict."
                        if ratio is not None
                        else f"Unknown: no positive call denominator or calls/puts are stale, missing, or clock-misaligned "
                        f"for {currency} expiry {expiry}."
                    ),
                    source_ts=max(clocks).isoformat() if clocks else None,
                    raw=payload if store_raw else None,
                )
            )
    if not output:
        raise ValueError("Deribit options response contained no parseable instrument rows")
    return output


def normalize_coingecko_relative_strength(
    source: dict[str, Any], payload: Any, *, store_raw: bool = True
) -> list[dict[str, Any]]:
    from datetime import datetime, timezone

    rows = payload if isinstance(payload, list) else []
    by_id = {str(row.get("id") or "").lower(): row for row in rows if isinstance(row, dict)}
    btc = by_id.get("bitcoin", {})
    eth = by_id.get("ethereum", {})
    btc_change = _finite_number(btc.get("price_change_percentage_24h"))
    eth_change = _finite_number(eth.get("price_change_percentage_24h"))
    btc_price = _finite_number(btc.get("current_price"))
    eth_price = _finite_number(eth.get("current_price"))
    btc_clock = _iso_clock(btc.get("last_updated"))
    eth_clock = _iso_clock(eth.get("last_updated"))
    max_age = int(source.get("max_age_seconds", 3600))
    max_clock_gap = int(source.get("max_clock_gap_seconds", 300))
    now = datetime.now(timezone.utc)
    clocks_aligned = (
        btc_clock is not None
        and eth_clock is not None
        and 0 <= (now - btc_clock).total_seconds() <= max_age
        and 0 <= (now - eth_clock).total_seconds() <= max_age
        and abs((btc_clock - eth_clock).total_seconds()) <= max_clock_gap
    )
    valid_changes = clocks_aligned and btc_change is not None and eth_change is not None
    difference = _finite_number(eth_change - btc_change) if valid_changes else None
    ratio = (
        _finite_number(eth_price / btc_price)
        if clocks_aligned
        and eth_price is not None
        and eth_price > 0
        and btc_price is not None
        and btc_price > 0
        else None
    )
    source_ts = max(btc_clock, eth_clock).isoformat() if clocks_aligned else None
    shared_notes = (
        "BTC and ETH values came from the same CoinGecko response. Relative strength is ETH 24h percent change "
        "minus BTC 24h percent change; it is not the existing average BTC/ETH return. Changes use the provider's "
        "rolling 24h window; matched provider last_updated clocks are preserved."
    )
    return [
        result(
            source,
            metric_key="eth_minus_btc_change_24h_pct",
            score=None,
            signal="",
            value=difference,
            unit="percentage_points",
            notes=shared_notes
            if difference is not None
            else "Unknown: BTC/ETH provider clocks are missing, stale, future, misaligned, or a 24h change is malformed.",
            source_ts=source_ts,
            raw=payload if store_raw else None,
        ),
        result(
            source,
            metric_key="eth_btc_price_ratio",
            score=None,
            signal="",
            value=ratio,
            unit="BTC_per_ETH",
            notes="ETH current USD price divided by BTC current USD price from the same response; descriptive ratio, not a return. Unknown unless both provider clocks are fresh and aligned.",
            source_ts=source_ts,
            raw=None,
        ),
    ]


def _finite_number(value: Any) -> float | None:
    parsed = float_or_none(value)
    return parsed if parsed is not None and math.isfinite(parsed) else None


def _deribit_clock(value: Any):
    from datetime import datetime, timezone

    parsed = _finite_number(value)
    if parsed is None:
        return None
    try:
        if parsed > 10_000_000_000:
            parsed /= 1000.0
        return datetime.fromtimestamp(parsed, timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None


def _iso_clock(value: Any):
    from datetime import datetime, timezone

    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return None
        return parsed.astimezone(timezone.utc)
    except (OverflowError, TypeError, ValueError):
        return None


def normalize_fear_greed(
    source: dict[str, Any], payload: Any, *, store_raw: bool = True
) -> dict[str, Any]:
    rows = payload.get("data") if isinstance(payload, dict) else []
    row = rows[0] if isinstance(rows, list) and rows and isinstance(rows[0], dict) else {}
    value = float_or_none(row.get("value"))
    classification = str(row.get("value_classification") or "")
    score = clamp(value if value is not None else 50.0)
    notes = classification or score_label(score)
    return result(
        source,
        metric_key="fear_greed_index",
        score=score,
        source_score=score,
        signal=score_signal(score),
        value=value,
        unit="index",
        notes=notes,
        source_ts=unix_to_iso(row.get("timestamp")),
        raw=payload if store_raw else None,
    )


def normalize_coingecko_global(
    source: dict[str, Any], payload: Any, *, store_raw: bool = True
) -> list[dict[str, Any]]:
    data = payload.get("data") if isinstance(payload, dict) else {}
    if not isinstance(data, dict):
        data = {}
    market_cap = nested_float(data, "total_market_cap", "usd")
    volume = nested_float(data, "total_volume", "usd")
    btc_dom = nested_float(data, "market_cap_percentage", "btc")
    change_24h = float_or_none(data.get("market_cap_change_percentage_24h_usd"))
    score = clamp(50.0 + (change_24h or 0.0) * 5.0)
    notes = f"cap ${compact_usd(market_cap)}, volume ${compact_usd(volume)}, BTC dom {fmt_pct(btc_dom)}, 24h cap {fmt_pct(change_24h)}"
    primary = result(
        source,
        metric_key="global_market_cap_change_24h",
        score=score,
        calc_score=score,
        signal=score_signal(score),
        value=change_24h,
        unit="percent",
        notes=notes,
        raw=payload if store_raw else None,
    )
    rows = [primary]
    if btc_dom is not None:
        dominance_score = _btc_dominance_score(btc_dom)
        rows.append(
            result(
                source,
                metric_key="btc_dominance_pct",
                score=dominance_score,
                calc_score=dominance_score,
                signal=_btc_dominance_signal(dominance_score),
                value=btc_dom,
                unit="percent",
                notes="BTC market-cap dominance from CoinGecko global endpoint. Score maps low dominance to Weak and high dominance to Strong.",
                raw=None,
            )
        )
    return rows


def normalize_coingecko_markets(
    source: dict[str, Any], payload: Any, *, store_raw: bool = True
) -> dict[str, Any]:
    rows = payload if isinstance(payload, list) else []
    changes_24h: list[float] = []
    changes_7d: list[float] = []
    note_parts: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        symbol = str(row.get("symbol") or row.get("id") or "").upper()
        price = float_or_none(row.get("current_price"))
        change_24h = float_or_none(row.get("price_change_percentage_24h"))
        change_7d = float_or_none(row.get("price_change_percentage_7d_in_currency"))
        if change_24h is not None:
            changes_24h.append(change_24h)
        if change_7d is not None:
            changes_7d.append(change_7d)
        if symbol:
            note_parts.append(
                f"{symbol} ${compact_usd(price)} 24h {fmt_pct(change_24h)} 7d {fmt_pct(change_7d)}"
            )
    avg_24h = mean(changes_24h)
    avg_7d = mean(changes_7d)
    score = clamp(50.0 + (avg_24h or 0.0) * 4.0 + (avg_7d or 0.0) * 1.5)
    return result(
        source,
        metric_key="btc_eth_avg_change_24h",
        score=score,
        calc_score=score,
        signal=score_signal(score),
        value=avg_24h,
        unit="percent",
        notes="; ".join(note_parts) or "No coin rows returned.",
        raw=payload if store_raw else None,
    )


def _btc_dominance_score(value: float) -> float:
    return clamp(50.0 + (value - 50.0) * 2.0)


def _btc_dominance_signal(score: float) -> str:
    if score >= 60:
        return "Strong"
    if score <= 40:
        return "Weak"
    return "Balanced"
