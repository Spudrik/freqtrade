from __future__ import annotations

import json
import math
import re
import time
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from .common import fetch_public_json, result


_EVM_ADDRESS_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
_ETH_HASH_RE = re.compile(r"^0x[0-9a-fA-F]{64}$")
_BTC_TXID_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_BTC_BECH32_RE = re.compile(r"^bc1[qpzry9x8gf2tvdw0s3jn54khce6mua7l]{59}$")
_HYPERLIQUID_SAMPLE_NOTE = (
    "Frozen outcome-blind sample of official Hyperliquid public BTC/ETH/HYPE trades over 45 seconds; "
    "the first eight lexicographic addresses among 18 with fills >= $10k were selected without PnL selection. "
    "These anonymous accounts are not representative whales and do not imply position direction."
)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if math.isfinite(parsed) else None


def _decimal(value: Any) -> Decimal | None:
    if isinstance(value, bool) or value is None or value == "":
        return None
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return parsed if parsed.is_finite() else None


def _finite_float_from_decimal(value: Decimal | None) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (OverflowError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return parsed if parsed > 0 and str(parsed) == str(value).strip() else None


def _evm_address(value: Any, *, label: str) -> str:
    address = str(value or "").strip()
    if not _EVM_ADDRESS_RE.fullmatch(address):
        raise ValueError(
            f"{label} must be exactly 20 bytes (0x plus 40 hexadecimal characters); "
            "refusing to guess or repair it"
        )
    return address


def _millis_clock(value: Any) -> datetime | None:
    millis = _finite_number(value)
    if millis is None or millis <= 0:
        return None
    try:
        return datetime.fromtimestamp(millis / 1000.0, UTC)
    except (OverflowError, OSError, ValueError):
        return None


def _iso_clock(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _clock_within_age(clock: datetime | None, max_age_seconds: Any) -> bool:
    if clock is None:
        return False
    max_age = _finite_number(max_age_seconds)
    if max_age is None or max_age < 0:
        return False
    age = (datetime.now(UTC) - clock).total_seconds()
    return -2 <= age <= max_age


def fetch_hyperliquid_clearinghouse(source: dict[str, Any], config: dict[str, Any]) -> Any:
    wallet = _evm_address(source.get("wallet"), label="Hyperliquid wallet")
    url = str(source.get("url") or "https://api.hyperliquid.xyz/info")
    parsed_url = urlparse(url)
    if (
        parsed_url.scheme != "https"
        or parsed_url.netloc.lower() != "api.hyperliquid.xyz"
        or parsed_url.path != "/info"
    ):
        raise ValueError(
            "Hyperliquid clearinghouse POST is restricted to its documented HTTPS /info endpoint"
        )
    body = json.dumps({"type": "clearinghouseState", "user": wallet}).encode("utf-8")
    request = Request(  # noqa: S310 - URL scheme and host are restricted above.
        url,
        data=body,
        headers={
            "User-Agent": str(config.get("user_agent") or "FreQ-GlobalContextCollector/1.0"),
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=int(config.get("request_timeout_seconds", 20))) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _hyperliquid_position_index(
    positions: Any,
) -> tuple[dict[str, dict[str, Any]], set[str], bool]:
    if not isinstance(positions, list):
        return {}, set(), False
    by_coin: dict[str, dict[str, Any]] = {}
    invalid_coins: set[str] = set()
    coverage_ok = True
    for item in positions:
        position = item.get("position") if isinstance(item, dict) else None
        coin = str(position.get("coin") or "").strip().upper() if isinstance(position, dict) else ""
        if not coin:
            coverage_ok = False
            continue
        if coin in by_coin:
            invalid_coins.add(coin)
        else:
            by_coin[coin] = position
    return by_coin, invalid_coins, coverage_ok


def _hyperliquid_signed_position(position: dict[str, Any]) -> tuple[float | None, str]:
    signed_size = _finite_number(position.get("szi"))
    absolute_value = _finite_number(position.get("positionValue"))
    if (
        signed_size is None
        or absolute_value is None
        or absolute_value < 0
        or (signed_size == 0 and absolute_value > 0)
    ):
        return None, "malformed szi or absolute positionValue; notional unknown"
    direction = 1.0 if signed_size > 0 else -1.0 if signed_size < 0 else 0.0
    risk_fields = {
        "szi": position.get("szi"),
        "entryPx": position.get("entryPx"),
        "leverage": position.get("leverage"),
        "unrealizedPnl": position.get("unrealizedPnl"),
        "liquidationPx": position.get("liquidationPx"),
    }
    risk = "position risk facts=" + json.dumps(risk_fields, separators=(",", ":"), sort_keys=True)
    return absolute_value * direction, risk


def normalize_hyperliquid_clearinghouse(
    source: dict[str, Any], payload: Any, *, store_raw: bool = True
) -> list[dict[str, Any]]:
    if not isinstance(payload, dict):
        raise ValueError("Malformed Hyperliquid clearinghouse response")
    symbols_raw = source.get("symbols")
    symbols = (
        [str(item).strip().upper() for item in symbols_raw] if isinstance(symbols_raw, list) else []
    )
    if not symbols or len(symbols) != len(set(symbols)):
        raise ValueError(
            "Hyperliquid source requires a non-empty, duplicate-free configured symbol list"
        )

    source_clock = _millis_clock(payload.get("time"))
    clock_ok = _clock_within_age(source_clock, source.get("max_age_seconds", 300))
    source_ts = source_clock.isoformat() if clock_ok else None
    margin = payload.get("marginSummary")
    account_value = _finite_number(margin.get("accountValue")) if isinstance(margin, dict) else None
    if account_value is not None and account_value < 0:
        account_value = None
    by_coin, invalid_coins, coverage_ok = _hyperliquid_position_index(payload.get("assetPositions"))

    shared = _HYPERLIQUID_SAMPLE_NOTE
    if not clock_ok:
        shared += " Provider time is missing, stale, future, or malformed; values are unknown."
    if not coverage_ok:
        shared += " Position list coverage is malformed; configured coin notionals are unknown."
    account_valid = clock_ok and account_value is not None
    output = [
        result(
            source,
            metric_key="hyperliquid_account_value_usd",
            score=None,
            signal="",
            value=account_value if account_valid else None,
            unit="USD",
            notes=(
                shared + " Clearinghouse accountValue from marginSummary."
                if account_valid
                else shared
                + " Unknown account value: missing or malformed marginSummary.accountValue."
            ),
            source_ts=source_ts,
            raw=payload if store_raw else None,
        )
    ]
    for coin in symbols:
        position = by_coin.get(coin)
        if not clock_ok:
            value, risk = None, "provider clock invalid; configured coin notional unknown"
        elif not coverage_ok:
            value, risk = None, "malformed position list; configured coin notional unknown"
        elif coin in invalid_coins:
            value, risk = None, "duplicate position rows for this coin; notional unknown"
        elif position is None:
            value, risk = 0.0, "no open position for this configured coin in the valid response"
        else:
            value, risk = _hyperliquid_signed_position(position)
        output.append(
            result(
                source,
                metric_key=f"hyperliquid_position_signed_notional_usd_{coin.lower()}",
                score=None,
                signal="",
                value=value,
                unit="USD",
                notes=(
                    f"Signed USD notional for {coin}: positive is long, negative is short; {risk}. "
                    + shared
                ),
                source_ts=source_ts,
                raw=None,
            )
        )
    return output


def fetch_blockscout_wallet_page(source: dict[str, Any], config: dict[str, Any]) -> Any:
    _evm_address(source.get("wallet"), label="Blockscout wallet")
    return fetch_public_json(
        str(source.get("url") or ""),
        timeout_seconds=int(config.get("request_timeout_seconds", 20)),
        user_agent=str(config.get("user_agent") or "FreQ-GlobalContextCollector/1.0"),
    )


def _blockscout_page_notes(
    payload: dict[str, Any], items: list[Any], page_size: int, response_count: int
) -> str:
    cursor = payload.get("next_page_params")
    cursor_present = isinstance(cursor, dict) and bool(cursor)
    truncated = cursor_present or response_count >= page_size
    if cursor_present:
        cursor_text = json.dumps(cursor, sort_keys=True, separators=(",", ":"))[:600]
        cursor_note = f" next_page_params cursor present but not followed: {cursor_text}."
    elif truncated:
        cursor_note = (
            " Page reached configured row limit; additional history may exist, "
            "but no cursor was followed."
        )
    else:
        cursor_note = " No next_page_params cursor was returned."
    note = (
        f"One Blockscout page only ({len(items)} of {response_count} response rows); "
        f"truncated_or_more_possible={str(truncated).lower()}.{cursor_note} "
        "Individual wallet transfers are not buys/sells; "
        "internal exchange-wallet movements are possible, "
        "and these rows do not establish complete current-hour flow."
    )
    return note


def _page_items(
    payload: Any, *, page_size: int, provider: str
) -> tuple[dict[str, Any], list[Any], int]:
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
        raise ValueError(f"Malformed {provider} page response: expected an items list")
    items = payload["items"]
    return payload, items[:page_size], len(items)


def normalize_bybit_eth_native(
    source: dict[str, Any], payload: Any, *, store_raw: bool = True
) -> list[dict[str, Any]]:
    wallet = _evm_address(source.get("wallet"), label="Blockscout wallet")
    page_size = max(1, min(int(source.get("page_size", 50)), 50))
    page, items, response_count = _page_items(payload, page_size=page_size, provider="Blockscout")
    page_note = _blockscout_page_notes(page, items, page_size, response_count)
    seen: set[str] = set()
    valid: list[dict[str, Any]] = []
    malformed = duplicates = 0
    for item in items:
        if not isinstance(item, dict):
            malformed += 1
            continue
        tx_hash = str(item.get("hash") or "").strip()
        if not _ETH_HASH_RE.fullmatch(tx_hash):
            malformed += 1
            continue
        normalized_hash = tx_hash.lower()
        if normalized_hash in seen:
            duplicates += 1
            continue
        seen.add(normalized_hash)
        block_number = _positive_int(item.get("block_number"))
        clock = _iso_clock(item.get("timestamp"))
        amount_wei = _decimal(item.get("value"))
        from_row, to_row = item.get("from"), item.get("to")
        from_hash = str(from_row.get("hash") or "").casefold() if isinstance(from_row, dict) else ""
        to_hash = str(to_row.get("hash") or "").casefold() if isinstance(to_row, dict) else ""
        wallet_key = wallet.casefold()
        if (
            block_number is None
            or clock is None
            or amount_wei is None
            or amount_wei < 0
            or not from_hash
            or not to_hash
            or (from_hash != wallet_key and to_hash != wallet_key)
        ):
            malformed += 1
            continue
        if from_hash == wallet_key and to_hash == wallet_key:
            signed_value = 0.0
        elif from_hash == wallet_key:
            signed_value = _finite_float_from_decimal(-amount_wei / Decimal(10**18))
        else:
            signed_value = _finite_float_from_decimal(amount_wei / Decimal(10**18))
        if signed_value is None:
            malformed += 1
            continue
        valid.append(
            {
                "tx_hash": normalized_hash,
                "clock": clock,
                "value": signed_value,
                "raw": item,
            }
        )
    partial = (
        f" Parsed {len(valid)} transaction(s), omitted {malformed} malformed row(s), "
        f"deduplicated {duplicates}."
        if malformed or duplicates
        else ""
    )
    note = page_note + partial
    latest = max((row["clock"] for row in valid), default=None)
    output = [
        result(
            source,
            metric_key="bybit_eth_native_parsed_page_transaction_count",
            score=None,
            signal="",
            value=float(len(valid)),
            unit="transactions",
            notes=(
                note + " Native ETH is wei converted to ETH; "
                "no USD conversion or price-direction inference."
            ),
            source_ts=latest.isoformat() if latest else None,
            raw=page if store_raw else None,
        )
    ]
    for row in valid:
        output.append(
            result(
                source,
                metric_key=f"bybit_eth_native_wallet_net_flow_eth_{row['tx_hash']}",
                score=None,
                signal="",
                value=row["value"],
                unit="ETH",
                notes=(
                    note + " Negative is a native ETH outflow from the selected wallet; "
                    "positive is an inflow."
                ),
                source_ts=row["clock"].isoformat(),
                raw=row["raw"] if store_raw else None,
            )
        )
    return output


def normalize_bybit_eth_erc20(
    source: dict[str, Any], payload: Any, *, store_raw: bool = True
) -> list[dict[str, Any]]:
    wallet = _evm_address(source.get("wallet"), label="Blockscout wallet")
    page_size = max(1, min(int(source.get("page_size", 50)), 50))
    page, items, response_count = _page_items(payload, page_size=page_size, provider="Blockscout")
    page_note = _blockscout_page_notes(page, items, page_size, response_count)
    seen: set[tuple[str, int]] = set()
    valid: list[dict[str, Any]] = []
    malformed = duplicates = 0
    wallet_key = wallet.casefold()
    for item in items:
        try:
            parsed = _parse_erc20_transfer(item, wallet_key)
        except ValueError:
            malformed += 1
            continue
        identity = (parsed["tx_hash"], parsed["log_index"])
        if identity in seen:
            duplicates += 1
            continue
        seen.add(identity)
        valid.append(parsed)
    partial = (
        f" Parsed {len(valid)} transfer log(s), omitted {malformed} malformed row(s), "
        f"deduplicated {duplicates}."
        if malformed or duplicates
        else ""
    )
    count_coverage = (
        " Transfer-log count is unknown because one or more response rows were malformed; "
        "do not interpret the parsed subset as a zero or complete count."
        if malformed
        else ""
    )
    note = page_note + partial + count_coverage
    latest = max((row["clock"] for row in valid), default=None)
    output = [
        result(
            source,
            metric_key="bybit_eth_erc20_parsed_page_transfer_log_count",
            score=None,
            signal="",
            value=None if malformed else float(len(valid)),
            unit="transfer_logs",
            notes=(
                note + " ERC-20 raw token units use provider decimals; "
                "no USD conversion or buys/sells inference."
            ),
            source_ts=latest.isoformat() if latest else None,
            raw=page if store_raw else None,
        )
    ]
    for row in valid:
        output.append(
            result(
                source,
                metric_key=(
                    f"bybit_eth_erc20_wallet_net_transfer_{row['token_address']}_"
                    f"{row['tx_hash']}_log_{row['log_index']}"
                ),
                score=None,
                signal="",
                value=row["value"],
                unit=row["unit"],
                notes=(
                    note + " Negative is a selected-wallet token outflow; positive is an inflow. "
                    "Transfers may be internal wallet movements."
                ),
                source_ts=row["clock"].isoformat(),
                raw=row["raw"] if store_raw else None,
            )
        )
    return output


def _parse_erc20_transfer(item: Any, wallet_key: str) -> dict[str, Any]:
    if not isinstance(item, dict):
        raise ValueError("Malformed ERC-20 transfer row")
    tx_hash = str(item.get("transaction_hash") or "").strip()
    token = item.get("token")
    # Blockscout v2 token-transfer rows identify the token with address_hash.
    # Do not fall back to similarly named fields from other provider schemas.
    token_address = str(token.get("address_hash") or "").strip() if isinstance(token, dict) else ""
    try:
        log_index = int(item.get("log_index"))
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Malformed ERC-20 log index")
    if log_index < 0 or str(log_index) != str(item.get("log_index")).strip():
        raise ValueError("Malformed ERC-20 log index")
    if not _ETH_HASH_RE.fullmatch(tx_hash) or not _EVM_ADDRESS_RE.fullmatch(token_address):
        raise ValueError("Malformed ERC-20 transaction or token address")
    block_number = _positive_int(item.get("block_number"))
    clock = _iso_clock(item.get("timestamp"))
    from_row, to_row = item.get("from"), item.get("to")
    from_hash = str(from_row.get("hash") or "").casefold() if isinstance(from_row, dict) else ""
    to_hash = str(to_row.get("hash") or "").casefold() if isinstance(to_row, dict) else ""
    total = item.get("total")
    raw_value = _decimal(total.get("value")) if isinstance(total, dict) else None
    raw_decimals = _decimal(total.get("decimals")) if isinstance(total, dict) else None
    decimals = None
    if (
        raw_decimals is not None
        and raw_decimals == raw_decimals.to_integral_value()
        and 0 <= raw_decimals <= 255
    ):
        decimals = int(raw_decimals)
    if (
        block_number is None
        or clock is None
        or raw_value is None
        or raw_value < 0
        or raw_value != raw_value.to_integral_value()
        or decimals is None
        or not 0 <= decimals <= 255
        or not from_hash
        or not to_hash
        or (from_hash != wallet_key and to_hash != wallet_key)
    ):
        raise ValueError("Malformed or unrelated ERC-20 transfer")
    direction = (
        0
        if from_hash == wallet_key and to_hash == wallet_key
        else -1
        if from_hash == wallet_key
        else 1
    )
    try:
        scaled_value = raw_value.scaleb(-decimals) * Decimal(direction)
    except (InvalidOperation, OverflowError):
        raise ValueError("Unrepresentable ERC-20 amount")
    signed_value = _finite_float_from_decimal(scaled_value)
    if signed_value is None:
        raise ValueError("Unrepresentable ERC-20 amount")
    symbol = str(token.get("symbol") or "").strip()[:32] if isinstance(token, dict) else ""
    unit = symbol if symbol else f"token:{token_address.lower()}"
    return {
        "tx_hash": tx_hash.lower(),
        "token_address": token_address.lower(),
        "log_index": log_index,
        "clock": clock,
        "value": signed_value,
        "unit": unit,
        "raw": item,
    }


def fetch_bybit_btc_address_page(source: dict[str, Any], config: dict[str, Any]) -> Any:
    address = str(source.get("address") or "").strip()
    if not _BTC_BECH32_RE.fullmatch(address):
        raise ValueError(
            "Configured Blockstream address must be a 62-character mainnet bc1 address; "
            "refusing to guess or repair it"
        )
    url = str(source.get("url") or "").format(address=address)
    return fetch_public_json(
        url,
        timeout_seconds=int(config.get("request_timeout_seconds", 20)),
        user_agent=str(config.get("user_agent") or "FreQ-GlobalContextCollector/1.0"),
    )


def _btc_wallet_input_sats(inputs: Any, address: str) -> int | None:
    if not isinstance(inputs, list) or not inputs:
        return None
    spent = 0
    for item in inputs:
        if not isinstance(item, dict):
            return None
        if item.get("is_coinbase") is True:
            continue
        prevout = item.get("prevout")
        if not isinstance(prevout, dict):
            return None
        value = _positive_or_zero_sats(prevout.get("value"))
        if value is None:
            return None
        prevout_address = prevout.get("scriptpubkey_address")
        if not isinstance(prevout_address, str) or not prevout_address:
            return None
        if prevout_address == address:
            spent += value
    return spent


def _btc_wallet_output_sats(outputs: Any, address: str) -> int | None:
    if not isinstance(outputs, list) or not outputs:
        return None
    received = 0
    for item in outputs:
        if not isinstance(item, dict):
            return None
        value = _positive_or_zero_sats(item.get("value"))
        if value is None:
            return None
        output_address = item.get("scriptpubkey_address")
        if output_address is None and value == 0:
            continue
        if not isinstance(output_address, str) or not output_address:
            return None
        if output_address == address:
            received += value
    return received


def _btc_transaction_net_sats(tx: dict[str, Any], address: str) -> int | None:
    spent = _btc_wallet_input_sats(tx.get("vin"), address)
    received = _btc_wallet_output_sats(tx.get("vout"), address)
    if spent is None or received is None:
        return None
    if spent == 0 and received == 0:
        return None
    return received - spent


def _positive_or_zero_sats(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if parsed < 0 or str(parsed) != str(value).strip():
        return None
    return parsed


def normalize_bybit_btc_address_page(
    source: dict[str, Any], payload: Any, *, store_raw: bool = True
) -> list[dict[str, Any]]:
    address = str(source.get("address") or "").strip()
    if not _BTC_BECH32_RE.fullmatch(address):
        raise ValueError(
            "Configured Blockstream address must be a 62-character mainnet bc1 address; "
            "refusing to guess or repair it"
        )
    if not isinstance(payload, list):
        raise ValueError("Malformed Blockstream address response: expected transaction list")
    page_size = max(1, min(int(source.get("page_size", 25)), 25))
    items = payload[:page_size]
    page_note = (
        f"One Blockstream latest-address page only ({len(items)} of {len(payload)} "
        "response transaction(s)); "
        f"page_limit={page_size}. A full page may have older history; no pagination was requested, "
        "so this is not complete current-hour or total exchange flow. "
        "Only confirmed transactions are projected. "
        "This is one dated public Bybit-attributed wallet, not total Bybit flow; "
        "transfers are not buys/sells."
    )
    seen: set[str] = set()
    valid: list[dict[str, Any]] = []
    pending = malformed = duplicates = 0
    for tx in items:
        if not isinstance(tx, dict):
            malformed += 1
            continue
        txid = str(tx.get("txid") or "").strip()
        if not _BTC_TXID_RE.fullmatch(txid):
            malformed += 1
            continue
        txid = txid.lower()
        if txid in seen:
            duplicates += 1
            continue
        seen.add(txid)
        status = tx.get("status")
        if isinstance(status, dict) and status.get("confirmed") is False:
            pending += 1
            continue
        if not isinstance(status, dict) or status.get("confirmed") is not True:
            malformed += 1
            continue
        height = _positive_int(status.get("block_height"))
        block_time = _positive_int(status.get("block_time"))
        net_sats = _btc_transaction_net_sats(tx, address)
        if height is None or block_time is None or net_sats is None:
            malformed += 1
            continue
        try:
            clock = datetime.fromtimestamp(block_time, UTC)
        except (OverflowError, OSError, ValueError):
            malformed += 1
            continue
        valid.append({"txid": txid, "clock": clock, "net_sats": net_sats, "raw": tx})
    partial = (
        f" Parsed {len(valid)} confirmed transaction(s); "
        f"excluded {pending} pending transaction(s), "
        f"omitted {malformed} malformed/unknown row(s), deduplicated {duplicates}."
    )
    latest = max((row["clock"] for row in valid), default=None)
    output = [
        result(
            source,
            metric_key="bybit_btc_confirmed_parsed_page_transaction_count",
            score=None,
            signal="",
            value=float(len(valid)),
            unit="transactions",
            notes=page_note + partial,
            source_ts=latest.isoformat() if latest else None,
            raw=payload if store_raw else None,
        )
    ]
    for row in valid:
        output.append(
            result(
                source,
                metric_key=f"bybit_btc_wallet_net_confirmed_flow_sats_{row['txid']}",
                score=None,
                signal="",
                value=float(row["net_sats"]),
                unit="satoshis",
                notes=page_note
                + partial
                + " Per-transaction net is received minus spent satoshis for this address.",
                source_ts=row["clock"].isoformat(),
                raw=row["raw"] if store_raw else None,
            )
        )
    return output


def _decode_ws_message(raw: Any) -> dict[str, Any]:
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("Bybit liquidation socket delivered malformed JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Bybit liquidation socket delivered a non-object JSON frame")
    return payload


def _is_bybit_pong(payload: dict[str, Any]) -> bool:
    # The public linear channel documents success=true, ret_msg=pong, op=ping.
    return (
        payload.get("success") is True
        and payload.get("ret_msg") == "pong"
        and payload.get("op") == "ping"
    )


def _parse_liquidation_message(
    message: dict[str, Any], *, symbols: set[str], received_at: datetime, max_age_seconds: int
) -> list[dict[str, Any]]:
    topic = str(message.get("topic") or "")
    prefix = "allLiquidation."
    if not topic.startswith(prefix):
        raise ValueError("Unexpected Bybit liquidation topic")
    topic_symbol = topic[len(prefix) :].upper()
    if topic_symbol not in symbols:
        raise ValueError("Bybit liquidation message topic is outside the configured symbol set")
    data = message.get("data")
    events = data if isinstance(data, list) else [data] if isinstance(data, dict) else None
    if events is None:
        raise ValueError("Malformed Bybit liquidation event data")
    parsed: list[dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Malformed Bybit liquidation event row")
        symbol = str(event.get("s") or "").upper()
        side = str(event.get("S") or "")
        clock = _millis_clock(event.get("T"))
        size = _finite_number(event.get("v"))
        bankruptcy_price = _finite_number(event.get("p"))
        if (
            symbol != topic_symbol
            or symbol not in symbols
            or side not in {"Buy", "Sell"}
            or clock is None
            or size is None
            or size <= 0
            or bankruptcy_price is None
            or bankruptcy_price <= 0
        ):
            raise ValueError("Malformed Bybit liquidation event fields")
        age = (received_at - clock).total_seconds()
        if age < -2 or age > max_age_seconds:
            raise ValueError("Bybit liquidation provider clock is future-dated or stale")
        parsed.append(
            {
                "symbol": symbol,
                "side": side,
                "source_ts": clock.isoformat(),
                "size": size,
                "bankruptcy_price": bankruptcy_price,
            }
        )
    return parsed


def _await_bybit_subscription_ack(
    socket_client: Any, timeout_error: type[Exception], topics: list[str], timeout_seconds: int
) -> None:
    socket_client.send(json.dumps({"op": "subscribe", "args": topics}, separators=(",", ":")))
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            payload = _decode_ws_message(socket_client.recv())
        except timeout_error:
            continue
        if payload.get("op") == "subscribe":
            accepted = payload.get("success") is True and payload.get("ret_msg") in {
                "",
                "subscribe",
            }
            if accepted:
                return
            raise RuntimeError("Bybit liquidation subscription was not acknowledged successfully")
        if "topic" in payload:
            raise RuntimeError("Bybit liquidation data arrived before subscription acknowledgement")
    raise RuntimeError("Bybit liquidation subscription acknowledgement timed out")


def _receive_bybit_observation(
    socket_client: Any,
    timeout_error: type[Exception],
    *,
    symbols: list[str],
    duration: int,
    max_events: int,
    max_event_age: int,
) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    ping_count = pong_count = 0
    event_limit_reached = False
    pending_ping_deadline: float | None = None
    observation_started_at = datetime.now(UTC)
    started_monotonic = time.monotonic()
    deadline = started_monotonic + duration
    next_ping_at = started_monotonic + 20.0
    while time.monotonic() < deadline:
        now_monotonic = time.monotonic()
        if pending_ping_deadline is not None and now_monotonic > pending_ping_deadline:
            raise RuntimeError("Bybit liquidation connection gap: JSON ping received no pong")
        if now_monotonic >= next_ping_at:
            socket_client.send(json.dumps({"op": "ping"}, separators=(",", ":")))
            ping_count += 1
            pending_ping_deadline = now_monotonic + 10.0
            next_ping_at += 20.0
        try:
            payload = _decode_ws_message(socket_client.recv())
        except timeout_error:
            continue
        received_at = datetime.now(UTC)
        if _is_bybit_pong(payload):
            if pending_ping_deadline is None:
                raise RuntimeError("Bybit liquidation received an unsolicited JSON pong")
            pong_count += 1
            pending_ping_deadline = None
            continue
        if payload.get("op") in {"pong", "ping"}:
            raise RuntimeError("Bybit liquidation JSON ping/pong response was malformed")
        if "topic" not in payload:
            continue
        events.extend(
            _parse_liquidation_message(
                payload,
                symbols=set(symbols),
                received_at=received_at,
                max_age_seconds=max_event_age,
            )
        )
        if len(events) >= max_events:
            events = events[:max_events]
            event_limit_reached = True
            break
    observation_ended_at = datetime.now(UTC)
    observation_seconds = min(float(duration), max(0.0, time.monotonic() - started_monotonic))
    if pending_ping_deadline is not None:
        raise RuntimeError("Bybit liquidation connection gap: last JSON ping was not answered")
    if ping_count < 1 or pong_count < 1:
        raise RuntimeError("Bybit liquidation window lacked a verified JSON ping/pong heartbeat")
    return {
        "acknowledged": True,
        "observation_started_at": observation_started_at.isoformat(),
        "observation_ended_at": observation_ended_at.isoformat(),
        "observation_seconds": observation_seconds,
        "ping_count": ping_count,
        "pong_count": pong_count,
        "event_limit_reached": event_limit_reached,
        "max_events": max_events,
        "events": events,
    }


def fetch_bybit_liquidations(source: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    try:
        import websocket
    except ImportError as exc:
        raise RuntimeError(
            "Bybit liquidation observer requires the existing websocket-client runtime dependency"
        ) from exc

    url = str(source.get("url") or "wss://stream.bybit.com/v5/public/linear")
    symbols_raw = source.get("symbols")
    symbols = (
        [str(item).strip().upper() for item in symbols_raw] if isinstance(symbols_raw, list) else []
    )
    if not symbols or len(symbols) != len(set(symbols)):
        raise ValueError(
            "Bybit liquidation source requires a non-empty, duplicate-free symbol list"
        )
    topics = [f"allLiquidation.{symbol}" for symbol in symbols]
    duration = min(60, max(1, int(source.get("observation_seconds", 60))))
    max_events = min(500, max(1, int(source.get("max_events", 500))))
    max_event_age = min(300, max(1, int(source.get("max_event_age_seconds", 60))))
    ack_timeout = min(10, max(1, int(source.get("ack_timeout_seconds", 10))))
    socket_client = websocket.create_connection(url, timeout=1.0, enable_multithread=True)
    try:
        socket_client.settimeout(1.0)
        _await_bybit_subscription_ack(
            socket_client, websocket.WebSocketTimeoutException, topics, ack_timeout
        )
        return _receive_bybit_observation(
            socket_client,
            websocket.WebSocketTimeoutException,
            symbols=symbols,
            duration=duration,
            max_events=max_events,
            max_event_age=max_event_age,
        )
    finally:
        socket_client.close()


def _liquidation_window_metadata(
    source: dict[str, Any], payload: Any
) -> tuple[float, datetime, datetime, int, int, bool, list[Any], list[str]]:
    if not isinstance(payload, dict) or payload.get("acknowledged") is not True:
        raise ValueError(
            "Bybit liquidation observation has no successful subscription acknowledgement"
        )
    try:
        duration = float(payload.get("observation_seconds"))
        ping_count = int(payload.get("ping_count"))
        pong_count = int(payload.get("pong_count"))
    except (TypeError, ValueError, OverflowError):
        raise ValueError("Bybit liquidation observation window metadata is malformed")
    if (
        not math.isfinite(duration)
        or duration <= 0
        or duration > 60
        or ping_count < 1
        or pong_count < 1
        or pong_count > ping_count
    ):
        raise ValueError(
            "Bybit liquidation observation lacks a bounded, heartbeat-verified live window"
        )
    started = _iso_clock(payload.get("observation_started_at"))
    ended = _iso_clock(payload.get("observation_ended_at"))
    events = payload.get("events")
    symbols_raw = source.get("symbols")
    symbols = (
        [str(item).strip().upper() for item in symbols_raw] if isinstance(symbols_raw, list) else []
    )
    if (
        started is None
        or ended is None
        or ended < started
        or not isinstance(events, list)
        or not symbols
        or len(symbols) != len(set(symbols))
    ):
        raise ValueError("Bybit liquidation observation coverage metadata is incomplete")
    event_limit = payload.get("event_limit_reached") is True
    if len(events) > min(500, max(1, int(source.get("max_events", 500)))):
        raise ValueError("Bybit liquidation observation exceeded its configured event bound")
    return duration, started, ended, ping_count, pong_count, event_limit, events, symbols


def _group_liquidation_events(
    events: list[Any], symbols: list[str]
) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {
        (symbol, side): [] for symbol in symbols for side in ("Buy", "Sell")
    }
    for event in events:
        if not isinstance(event, dict):
            raise ValueError("Malformed normalized Bybit liquidation event")
        symbol = str(event.get("symbol") or "").upper()
        side = str(event.get("side") or "")
        clock = _iso_clock(event.get("source_ts"))
        size = _finite_number(event.get("size"))
        bankruptcy_price = _finite_number(event.get("bankruptcy_price"))
        if (
            (symbol, side) not in grouped
            or clock is None
            or size is None
            or size <= 0
            or bankruptcy_price is None
            or bankruptcy_price <= 0
        ):
            raise ValueError("Malformed normalized Bybit liquidation event fields")
        grouped[(symbol, side)].append(
            {"clock": clock, "size": size, "bankruptcy_price": bankruptcy_price}
        )
    return grouped


def _liquidation_window_note(
    duration: float,
    started: datetime,
    ended: datetime,
    pong_count: int,
    ping_count: int,
    event_limit: bool,
    events: list[Any],
) -> str:
    note = (
        f"Verified Bybit public linear WebSocket observation only: {duration:.2f}s from "
        f"{started.isoformat()} to {ended.isoformat()}; subscription acknowledged and "
        f"{pong_count}/{ping_count} documented JSON heartbeat(s) answered. "
        "Buy denotes long liquidation and Sell short liquidation. "
        "p is bankruptcy price, not fill price; "
        "bankruptcy-notional values are size x bankruptcy-price proxies only. "
        "This is not a complete hour/day flow."
    )
    if event_limit:
        note += (
            f" Event bound reached at {len(events)} event(s); "
            "the observed window ended early and is explicitly partial."
        )
    return note


def _liquidation_metric_rows(
    source: dict[str, Any],
    symbols: list[str],
    grouped: dict[tuple[str, str], list[dict[str, Any]]],
    window_note: str,
    *,
    store_raw: bool,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for symbol in symbols:
        for side in ("Buy", "Sell"):
            rows = grouped[(symbol, side)]
            latest = max((row["clock"] for row in rows), default=None)
            safe_symbol = symbol.lower().removesuffix("usdt")
            side_label = "long" if side == "Buy" else "short"
            total_size = sum(row["size"] for row in rows)
            total_proxy = sum(row["size"] * row["bankruptcy_price"] for row in rows)
            values = (
                ("event_count", float(len(rows)), "events"),
                ("size_base", total_size, "base_units"),
                ("bankruptcy_notional_proxy", total_proxy, "bankruptcy_price_proxy_usd"),
            )
            for metric, value, unit in values:
                if not math.isfinite(value):
                    raise ValueError(
                        "Bybit liquidation aggregate overflowed; source metrics are unknown"
                    )
                output.append(
                    result(
                        source,
                        metric_key=f"bybit_liquidation_observed_{safe_symbol}_{side_label}_{metric}",
                        score=None,
                        signal="",
                        value=value,
                        unit=unit,
                        notes=window_note
                        + (
                            " No matching event was observed in the verified live window."
                            if not rows
                            else ""
                        ),
                        source_ts=latest.isoformat() if latest else None,
                        raw=rows if store_raw else None,
                    )
                )
    return output


def normalize_bybit_liquidations(
    source: dict[str, Any], payload: Any, *, store_raw: bool = True
) -> list[dict[str, Any]]:
    (
        duration,
        started,
        ended,
        ping_count,
        pong_count,
        event_limit,
        events,
        symbols,
    ) = _liquidation_window_metadata(source, payload)
    grouped = _group_liquidation_events(events, symbols)
    window_note = _liquidation_window_note(
        duration, started, ended, pong_count, ping_count, event_limit, events
    )
    return _liquidation_metric_rows(source, symbols, grouped, window_note, store_raw=store_raw)
