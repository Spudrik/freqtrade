"""Prospective, descriptive follow-up for root-owned news interpretations.

This ledger contains no account or bot context. It records a ticker baseline at
registration and measures only closed public futures candles after that baseline.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from math import isfinite
from pathlib import Path
from urllib.parse import urlencode, urlparse

from user_data.Custom_Launcher.collector_runtime import atomic_write_text
from user_data.strategies.integrated_paper_context import _utc


ROOT = Path(__file__).resolve().parents[3]
STATE_PATH = (ROOT / "user_data/research_news_data/context_features"
              / "integrated_paper_20260926/paper_news_followups.json")
SYMBOLS = ("BTCUSDT", "ETHUSDT", "SOLUSDT")
HORIZONS_MINUTES = (30, 60, 240)
MAX_EVENTS_PER_UPDATE = 3
TICKER_URL = "https://fapi.binance.com/fapi/v1/ticker/price"
KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"


def _clock(value: str, label: str, *, allow_unknown: bool = False) -> datetime | None:
    if allow_unknown and value.strip().casefold() == "unknown":
        return None
    try:
        return _utc(value)
    except (AttributeError, TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be an offset-aware UTC timestamp" +
                         (" or 'unknown'" if allow_unknown else "")) from exc


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _positive(value) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if isfinite(number) and number > 0 else None


def _state(path: Path = STATE_PATH) -> tuple[dict, bool]:
    if not path.is_file():
        return {"schema_version": 1, "events": {}}, False
    import json

    row = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(row, dict) or row.get("schema_version") != 1
            or not isinstance(row.get("events"), dict)):
        raise ValueError(f"Refusing malformed news follow-up state: {path}")
    for event_id, event in row["events"].items():
        if (not isinstance(event_id, str) or not isinstance(event, dict)
                or event.get("event_id") != event_id
                or not isinstance(event.get("baselines"), dict)
                or not isinstance(event.get("outcomes"), dict)):
            raise ValueError(f"Refusing malformed news follow-up event in: {path}")
        for horizon in HORIZONS_MINUTES:
            outcomes = event["outcomes"].get(str(horizon))
            if not isinstance(outcomes, dict) or any(
                    not isinstance(outcomes.get(symbol), dict) for symbol in SYMBOLS):
                raise ValueError(f"Refusing malformed {horizon}m follow-up in: {path}")
    return row, True


def _save(state: dict, path: Path) -> None:
    import json

    atomic_write_text(path, json.dumps(state, indent=2, ensure_ascii=False,
                                       allow_nan=False) + "\n")


def _fetch_json(url: str, timeout: int):
    # Lazy import keeps snapshot.py -> follow-up reporting free of an import cycle.
    from user_data.Custom_Launcher.research.paper_trial_snapshot import _public_json

    return _public_json(url, timeout=timeout)


def _ticker_prices(payload) -> dict[str, float]:
    from user_data.Custom_Launcher.research.paper_trial_snapshot import _ticker_prices as parse

    return parse(payload)


def _validate_choice(value: str, allowed: tuple[str, ...], label: str) -> str:
    if value not in allowed:
        raise ValueError(f"{label} must be one of: {', '.join(allowed)}")
    return value


def register_interpretation(*, event_id: str, rationale: str, action: str,
                            expected_direction: str, expected_reaction: str,
                            confidence: str, evidence_context: str,
                            reasoning_version: str, source_urls: list[str],
                            event_time_utc: str, publication_time_utc: str,
                            first_seen_clock_utc: str, forecast_minutes: int = 60,
                            now: datetime | None = None, fetch_json=None,
                            state_path: Path = STATE_PATH) -> dict:
    """Register prospectively; a duplicate ID only appends an evidence note."""
    event_id = event_id.strip()
    if not event_id or len(event_id) > 120:
        raise ValueError("event_id must be a stable non-empty underlying-event key (max 120 chars)")
    if not rationale.strip() or not reasoning_version.strip():
        raise ValueError("rationale and reasoning_version are required")
    _validate_choice(action, ("trade", "hold", "decline", "observe"), "action")
    _validate_choice(expected_direction, ("up", "down", "mixed", "unknown"), "expected_direction")
    _validate_choice(expected_reaction, ("elevated", "normal", "unknown"), "expected_reaction")
    _validate_choice(confidence, ("low", "medium", "high"), "confidence")
    _validate_choice(evidence_context, ("supporting", "contrary", "unknown"), "evidence_context")
    if type(forecast_minutes) is not int or forecast_minutes not in HORIZONS_MINUTES:
        raise ValueError(f"forecast_minutes must be one of {HORIZONS_MINUTES}")
    if not source_urls:
        raise ValueError("At least one direct source URL is required")
    for url in source_urls:
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"Invalid HTTP(S) source URL: {url}")

    event_time = _clock(event_time_utc, "event_time_utc", allow_unknown=True)
    publication_time = _clock(publication_time_utc, "publication_time_utc", allow_unknown=True)
    first_seen = _clock(first_seen_clock_utc, "first_seen_clock_utc", allow_unknown=True)
    fixed_clock = now is not None
    if fixed_clock and now.tzinfo is None:
        raise ValueError("Registration clock must be timezone-aware")
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    if event_time is not None and event_time > now:
        raise ValueError("event_time_utc cannot be in the future; use 'unknown' unless it is an actual event clock")
    if publication_time is not None and publication_time > now:
        raise ValueError("publication_time_utc cannot be in the future")
    if first_seen is not None and first_seen > now:
        raise ValueError("first_seen_clock_utc cannot be in the future")

    state, _ = _state(state_path)
    if event_id in state["events"]:
        event = state["events"][event_id]
        event.setdefault("duplicate_evidence_notes", []).append({
            "registered_at_utc": _iso(now), "rationale": rationale.strip(),
            "evidence_context": evidence_context, "source_urls": list(source_urls),
            "note": "Evidence update only; original assessment and baseline are unchanged.",
        })
        _save(state, state_path)
        return {"status": "duplicate_evidence_attached", "event_id": event_id,
                "independent_event_count_changed": False,
                "original_decision_at_utc": event["decision_at_utc"]}

    fetch_json = fetch_json or _fetch_json
    payload = fetch_json(TICKER_URL, 8)
    observed_at = now if fixed_clock else datetime.now(timezone.utc)
    # The futures ticker response has no source timestamp; this is the local
    # receipt clock for the actual fetched prices, not an exchange event clock.
    quote_clock = _iso(observed_at)
    prices = _ticker_prices(payload)
    baselines = {}
    for symbol in SYMBOLS:
        price = _positive(prices.get(symbol))
        baselines[symbol] = {
            "status": "ok" if price is not None else "unknown",
            "price_usdt": price,
            "observed_at_utc": quote_clock,
            "source": "Binance USD-M futures ticker; local response-receipt clock",
            "unknown_reason": None if price is not None else (
                "fetch_failed" if payload is None else "missing_or_invalid_quote"),
        }

    outcomes = {}
    for horizon in HORIZONS_MINUTES:
        outcomes[str(horizon)] = {}
        for symbol in SYMBOLS:
            base = baselines[symbol]
            outcomes[str(horizon)][symbol] = {
                "status": "pending" if base["status"] == "ok" else "unavailable",
                "reason": None if base["status"] == "ok" else "baseline_quote_unknown",
                "target_at_utc": _iso(_clock(quote_clock, "quote clock") + timedelta(minutes=horizon))
                    if base["status"] == "ok" else None,
            }

    event = {
        "event_id": event_id,
        "rationale": rationale.strip(),
        "action": action,
        "expected_direction": expected_direction,
        "expected_reaction": expected_reaction,
        "confidence": confidence,
        "evidence_context": evidence_context,
        "reasoning_version": reasoning_version.strip(),
        "source_urls": list(source_urls),
        "event_time_utc": None if event_time is None else _iso(event_time),
        "publication_time_utc": None if publication_time is None else _iso(publication_time),
        "first_seen_clock_utc": None if first_seen is None else _iso(first_seen),
        "decision_at_utc": _iso(now),
        "forecast_minutes": forecast_minutes,
        "coins": list(SYMBOLS),
        "baselines": baselines,
        "outcomes": outcomes,
        "duplicate_evidence_notes": [],
        "clock_note": "Decision and ticker receipt clocks are actual local UTC observations; event/publication/first-seen may be unknown.",
    }
    state["events"][event_id] = event
    _save(state, state_path)
    return {"status": "registered", "event_id": event_id,
            "decision_at_utc": event["decision_at_utc"],
            "baseline_status_by_coin": {symbol: baselines[symbol]["status"] for symbol in SYMBOLS},
            "independent_event_count_changed": True}


def _minute_floor(value: datetime) -> datetime:
    return value.replace(second=0, microsecond=0)


def _minute_ceil(value: datetime) -> datetime:
    floored = _minute_floor(value)
    return floored if floored == value else floored + timedelta(minutes=1)


def _candle_map(payload, now: datetime, requested_start: datetime,
                requested_end: datetime) -> tuple[dict[int, dict], str | None]:
    if not isinstance(payload, list):
        return {}, "fetch_failed" if payload is None else "invalid_candle_payload"
    candles = {}
    for row in payload:
        if not isinstance(row, list) or len(row) < 9:
            return {}, "invalid_candle_row"
        try:
            opened_ms = int(row[0])
            close_ms = int(row[6])
        except (TypeError, ValueError, OverflowError):
            return {}, "invalid_candle_clock"
        if opened_ms % 60_000 or close_ms < opened_ms + 59_999 or close_ms > opened_ms + 60_000:
            return {}, "invalid_candle_clock"
        if opened_ms in candles:
            return {}, "duplicate_candle"
        try:
            values = [float(row[index]) for index in (1, 2, 3, 4, 5)]
        except (TypeError, ValueError, OverflowError):
            return {}, "invalid_candle_values"
        if not all(isfinite(value) for value in values):
            return {}, "invalid_candle_values"
        opened, high, low, close, volume = values
        if min(opened, high, low, close) <= 0 or volume < 0 or high < max(opened, close, low) or low > min(opened, close, high):
            return {}, "invalid_candle_values"
        taker_buy = None
        if len(row) > 9:
            try:
                parsed_taker = float(row[9])
                if isfinite(parsed_taker) and 0 <= parsed_taker <= volume:
                    taker_buy = parsed_taker
            except (TypeError, ValueError, OverflowError):
                pass
        if opened_ms < int(requested_start.timestamp() * 1000) or opened_ms > int(requested_end.timestamp() * 1000):
            continue
        # Binance close time is inclusive; the bar is safely closed only when
        # its next-minute boundary is no later than both clocks.
        close_at = datetime.fromtimestamp((opened_ms + 60_000) / 1000, timezone.utc)
        if close_at <= requested_end and close_at <= now:
            candles[opened_ms] = {"open": opened, "high": high, "low": low,
                                 "close": close, "volume": volume,
                                 "taker_buy_volume": taker_buy,
                                 "close_at_utc": _iso(close_at)}
    return candles, None


def _measure_horizon(candles: dict[int, dict], baseline: dict, horizon: int,
                     now: datetime) -> tuple[dict | None, str | None]:
    baseline_at = _clock(baseline["observed_at_utc"], "baseline observed_at_utc")
    baseline_price = _positive(baseline.get("price_usdt"))
    if baseline_at is None or baseline_price is None:
        return None, "baseline_quote_unknown"
    target = baseline_at + timedelta(minutes=horizon)
    first_open = _minute_ceil(baseline_at)
    last_open = _minute_floor(target - timedelta(minutes=1))
    if last_open < first_open:
        return None, "no_complete_post_baseline_minutes"
    count = int((last_open - first_open).total_seconds() // 60) + 1
    post_opens = [int((first_open + timedelta(minutes=index)).timestamp() * 1000)
                  for index in range(count)]
    # End the pre-decision control at the floored baseline minute. This excludes
    # the partial candle containing the ticker observation.
    prior_end = _minute_floor(baseline_at)
    prior_first = prior_end - timedelta(minutes=count)
    prior_opens = [int((prior_first + timedelta(minutes=index)).timestamp() * 1000)
                   for index in range(count)]
    expected = prior_opens + post_opens
    if any(opened not in candles for opened in expected):
        return None, "coverage_gap"
    post = [candles[opened] for opened in post_opens]
    prior = [candles[opened] for opened in prior_opens]
    last = post[-1]
    high = max(row["high"] for row in post)
    low = min(row["low"] for row in post)
    final = last["close"]
    post_volume = sum(row["volume"] for row in post)
    prior_volume = sum(row["volume"] for row in prior)
    volume_ratio = post_volume / prior_volume if prior_volume > 0 else None
    taker_available = all(row["taker_buy_volume"] is not None for row in post)
    taker_share = (sum(row["taker_buy_volume"] for row in post) / post_volume
                   if taker_available and post_volume > 0 else None)
    taker_pressure = (2 * taker_share - 1) if taker_share is not None else None
    first_excluded = _minute_floor(baseline_at) if first_open > baseline_at else None
    return {
        "status": "matured",
        "target_at_utc": _iso(target),
        "measurement_started_at_utc": _iso(baseline_at),
        "measurement_ended_at_utc": last["close_at_utc"],
        "first_bar_open_at_utc": _iso(first_open),
        "last_bar_open_at_utc": _iso(last_open),
        "complete_post_baseline_minutes": count,
        "effective_measurement_minutes": round(((last_open + timedelta(minutes=1)) - baseline_at).total_seconds() / 60, 3),
        "baseline_minute_rounding_seconds": round((first_open - baseline_at).total_seconds(), 3),
        "excluded_partial_first_minute": first_excluded is not None,
        "excluded_first_minute_open_at_utc": None if first_excluded is None else _iso(first_excluded),
        "excluded_terminal_partial_minute": target != _minute_floor(target),
        "baseline_price_usdt": baseline_price,
        "final_price_usdt": final,
        "return_pct": (final / baseline_price - 1) * 100,
        "highest_price_usdt": high,
        "lowest_price_usdt": low,
        "high_move_pct": (high / baseline_price - 1) * 100,
        "low_move_pct": (low / baseline_price - 1) * 100,
        "range_pct": (high - low) / baseline_price * 100,
        "post_volume": post_volume,
        "prior_equal_duration_volume": prior_volume,
        "volume_ratio": volume_ratio,
        "taker_buy_volume_share_proxy": taker_share,
        "taker_pressure_proxy": taker_pressure,
        "taker_proxy_note": "Candle taker-buy volume proxy only; not order-book pressure.",
    }, None


def update_followups(*, now: datetime | None = None, fetch_json=None,
                    state_path: Path = STATE_PATH,
                    max_events: int = MAX_EVENTS_PER_UPDATE) -> dict:
    """Mature due windows with one bounded kline request per coin/event."""
    if now is not None and now.tzinfo is None:
        raise ValueError("Update clock must be timezone-aware")
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    fetch_json = fetch_json or _fetch_json
    state, exists = _state(state_path)
    if not exists:
        return {"status": "no_ledger", "events_considered": 0, "requests": 0}
    if type(max_events) is not int or max_events < 1:
        raise ValueError("max_events must be a positive integer")
    changed = False
    requests = 0
    considered = 0
    def fair_key(item):
        return (item.get("last_update_attempt_at_utc") or item["decision_at_utc"],
                item["decision_at_utc"])

    for event in sorted(state["events"].values(), key=fair_key):
        pending_due = False
        for horizon in HORIZONS_MINUTES:
            for symbol in SYMBOLS:
                outcome = event["outcomes"][str(horizon)][symbol]
                base = event["baselines"][symbol]
                if (outcome.get("status") == "pending" and base.get("status") == "ok"
                        and _clock(outcome["target_at_utc"], "target_at_utc") <= now):
                    pending_due = True
        if not pending_due:
            continue
        if considered >= max_events:
            break
        considered += 1
        event_had_request = False
        for symbol in SYMBOLS:
            baseline = event["baselines"][symbol]
            if baseline.get("status") != "ok":
                continue
            due = [h for h in HORIZONS_MINUTES
                   if event["outcomes"][str(h)][symbol].get("status") == "pending"
                   and _clock(event["outcomes"][str(h)][symbol]["target_at_utc"], "target_at_utc") <= now]
            if not due:
                continue
            baseline_at = _clock(baseline["observed_at_utc"], "baseline observed_at_utc")
            request_start = _minute_floor(baseline_at - timedelta(minutes=max(HORIZONS_MINUTES)))
            request_end = baseline_at + timedelta(minutes=max(HORIZONS_MINUTES))
            params = {"symbol": symbol, "interval": "1m", "startTime": int(request_start.timestamp() * 1000),
                      "endTime": int(request_end.timestamp() * 1000), "limit": 1000}
            url = KLINES_URL + "?" + urlencode(params)
            payload = fetch_json(url, 8)
            requests += 1
            event_had_request = True
            candles, fetch_error = _candle_map(payload, now, request_start, request_end)
            for horizon in due:
                outcome = event["outcomes"][str(horizon)][symbol]
                if fetch_error is not None:
                    outcome["last_attempt_at_utc"] = _iso(now)
                    outcome["last_attempt_status"] = fetch_error
                    outcome["reason"] = fetch_error
                    changed = True
                    continue
                measurement, error = _measure_horizon(candles, baseline, horizon, now)
                outcome["last_attempt_at_utc"] = _iso(now)
                if error is not None:
                    outcome["last_attempt_status"] = error
                    outcome["reason"] = error
                else:
                    outcome.clear()
                    outcome.update(measurement)
                    outcome["last_attempt_at_utc"] = _iso(now)
                    outcome["last_attempt_status"] = "ok"
                changed = True
        if event_had_request:
            event["last_update_attempt_at_utc"] = _iso(now)
            changed = True
    if changed:
        _save(state, state_path)
    return {"status": "updated" if changed else "nothing_due", "events_considered": considered,
            "requests": requests, "max_events_per_update": max_events}


def _return_direction(outcome: dict) -> str | None:
    value = outcome.get("return_pct")
    if outcome.get("status") != "matured" or not isinstance(value, (int, float)) or not isfinite(value):
        return None
    return "up" if value > 0 else "down" if value < 0 else None


def news_followup_report(*, state_path: Path = STATE_PATH,
                         now: datetime | None = None) -> dict:
    """Read-only compact summary; never fetches, initializes, or writes state."""
    if now is not None and now.tzinfo is None:
        raise ValueError("Report clock must be timezone-aware")
    now = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    state, exists = _state(state_path)
    events = list(state["events"].values())
    action_counts = {action: sum(event.get("action") == action for event in events)
                     for action in ("trade", "hold", "decline", "observe")}

    def report_status(outcome: dict) -> str:
        status = outcome.get("status")
        if status == "unavailable":
            return "no_baseline_unknown"
        if status != "pending":
            return status or "unknown"
        attempt_status = outcome.get("last_attempt_status")
        if attempt_status and attempt_status != "ok":
            return "retryable_unavailable"
        target = _clock(outcome.get("target_at_utc"), "target_at_utc")
        return "due_pending" if target <= now else "not_yet_due_pending"

    per_coin = {}
    for symbol in SYMBOLS:
        per_coin[symbol] = {}
        for horizon in HORIZONS_MINUTES:
            outcomes = [(event, event["outcomes"][str(horizon)][symbol]) for event in events]
            matured = [(event, row) for event, row in outcomes if row.get("status") == "matured"]
            statuses = [report_status(row) for _, row in outcomes]
            intended_matured = [(event, row) for event, row in matured
                                if event.get("forecast_minutes") == horizon]
            matched = calls = 0
            for event, row in intended_matured:
                expected = event.get("expected_direction")
                actual = _return_direction(row)
                if expected in {"up", "down"} and actual is not None:
                    calls += 1
                    matched += expected == actual
            per_coin[symbol][str(horizon)] = {
                "matured": len(matured),
                "intended_forecast_matured": len(intended_matured),
                "pending": statuses.count("not_yet_due_pending"),
                "due_pending": statuses.count("due_pending"),
                "retryable_unavailable": statuses.count("retryable_unavailable"),
                "unknown": statuses.count("no_baseline_unknown"),
                "direction_calls_with_nonzero_return": calls,
                "raw_direction_agreement_count": matched,
                "agreement_basis": "intended horizon only; descriptive raw sign, not profitability, usefulness, or causation",
            }

    cohorts = {}
    for event in events:
        horizon = event.get("forecast_minutes")
        version = event.get("reasoning_version")
        row = event.get("outcomes", {}).get(str(horizon), {}).get("BTCUSDT", {})
        if (horizon in HORIZONS_MINUTES and event.get("expected_direction") in {"up", "down"}
                and row.get("status") == "matured" and _return_direction(row) is not None):
            cohorts.setdefault((horizon, version), []).append((event, row))
    comparable = max(cohorts.items(), key=lambda item: len(item[1]), default=None)
    chronology = {"status": "insufficient_sample", "required_disjoint_events": 20,
                  "note": "No improvement claim; chronology is descriptive only."}
    if comparable is not None and len(comparable[1]) >= 20:
        (horizon, version), rows = comparable
        rows = sorted(rows, key=lambda pair: pair[0]["decision_at_utc"])
        first = rows[:10]
        recent = rows[-10:]
        def period_summary(period):
            matched = sum(event["expected_direction"] == _return_direction(outcome)
                          for event, outcome in period)
            return {"n": len(period), "raw_direction_agreement_count": matched,
                    "start_decision_at_utc": period[0][0]["decision_at_utc"],
                    "end_decision_at_utc": period[-1][0]["decision_at_utc"]}
        chronology = {"status": "descriptive_comparison", "coin": "BTCUSDT",
                      "forecast_minutes": horizon, "reasoning_version": version,
                      "first_10": period_summary(first), "latest_10": period_summary(recent),
                      "disjoint": len({event["event_id"] for event, _ in first}
                                      & {event["event_id"] for event, _ in recent}) == 0,
                      "note": "Same intended horizon and reasoning version; descriptive only, not proof of improvement, profit, or causation."}

    latest = sorted(events, key=lambda event: event.get("decision_at_utc", ""))[-3:]
    recent_events = []
    for event in reversed(latest):
        intended_horizon = str(event.get("forecast_minutes"))
        intended_outcomes = {}
        for symbol in SYMBOLS:
            outcome = event.get("outcomes", {}).get(intended_horizon, {}).get(symbol, {})
            intended_outcomes[symbol] = {**outcome, "report_status": report_status(outcome)}
        recent_events.append({
            "event_id": event["event_id"], "decision_at_utc": event.get("decision_at_utc"),
            "first_seen_clock_utc": event.get("first_seen_clock_utc"),
            "event_time_utc": event.get("event_time_utc"),
            "publication_time_utc": event.get("publication_time_utc"),
            "action": event.get("action"), "expected_direction": event.get("expected_direction"),
            "expected_reaction": event.get("expected_reaction"), "confidence": event.get("confidence"),
            "evidence_context": event.get("evidence_context"),
            "reasoning_version": event.get("reasoning_version"),
            "forecast_minutes": event.get("forecast_minutes"), "rationale": event.get("rationale"),
            "source_urls": event.get("source_urls", []),
            "intended_outcomes": intended_outcomes,
        })
    return {
        "status": "ok" if exists else "no_ledger",
        "independent_underlying_event_count": len(events),
        "action_counts": action_counts,
        "per_coin_horizon": per_coin,
        "btc_primary_chronology": chronology,
        "recent_events": recent_events,
        "limitations": [
            "Raw sign agreement is descriptive only, not profit, usefulness, or news causation; a tiny correct-sign move is not evidence of a useful or profitable decision.",
            "Raw direction agreement is counted only at each event's frozen intended horizon; other matured horizons are diagnostics.",
            "Each coin and horizon is reported separately; correlated coins/horizons are not independent confirmations.",
            "Returns start at each actual ticker baseline after the root decision; event-anchored or already-moved historical reaction is not scored.",
            "Ticker exchange timestamp is unavailable; local response-receipt clocks are recorded. Incomplete candle coverage remains pending/unknown.",
            "Taker-buy share/pressure is only a candle-volume proxy, not order-book pressure.",
        ],
    }


def format_news_followup_report(report: dict) -> str:
    if report.get("status") == "no_ledger":
        return "## News interpretation follow-ups\n\nNo follow-up ledger has been initialized."
    lines = ["## News interpretation follow-ups — descriptive, not a trading verdict", "",
             f"Independent underlying events: {report['independent_underlying_event_count']}. "
             + ", ".join(f"{key}: {value}" for key, value in report["action_counts"].items()), "",
             "BTCUSDT is primary; each coin and horizon stays separate. Sign agreement is scored only at each event's intended horizon; other windows are diagnostics.", "",
             "| Coin | Horizon | Matured | Intended | Not-yet-due | Due pending | Retryable unknown | No-baseline unknown | Intended sign matched |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for symbol in SYMBOLS:
        for horizon in HORIZONS_MINUTES:
            row = report["per_coin_horizon"][symbol][str(horizon)]
            lines.append(f"| {symbol} | {horizon}m | {row['matured']} | {row['intended_forecast_matured']} | {row['pending']} | {row['due_pending']} | {row['retryable_unavailable']} | {row['unknown']} | "
                         f"{row['raw_direction_agreement_count']}/{row['direction_calls_with_nonzero_return']} |")
    chronology = report["btc_primary_chronology"]
    if chronology["status"] == "descriptive_comparison":
        lines.extend(["", f"BTC first 10 vs latest 10: {chronology['first_10']['raw_direction_agreement_count']}/10 "
                      f"vs {chronology['latest_10']['raw_direction_agreement_count']}/10; "
                      f"same {chronology['forecast_minutes']}m horizon and reasoning version "
                      f"{chronology['reasoning_version']!r}. Descriptive only, not evidence of improvement."])
    else:
        lines.extend(["", "BTC chronology comparison: insufficient sample (requires 20 matured, comparable "
                      "directional events for disjoint first-10/latest-10 groups)."])
    if report["recent_events"]:
        lines.extend(["", "Recent root interpretations (latest first):"])
        for event in report["recent_events"]:
            lines.append(f"- `{event['event_id']}` — {event['action']}; "
                         f"{event['expected_direction']}/{event['expected_reaction']}, "
                         f"intended {event['forecast_minutes']}m, {event['confidence']} confidence; "
                         f"event {event['event_time_utc'] or 'unknown'}, publication "
                         f"{event['publication_time_utc'] or 'unknown'}, first seen "
                         f"{event['first_seen_clock_utc'] or 'unknown'}.")
            measurements = []
            for symbol in SYMBOLS:
                outcome = event["intended_outcomes"][symbol]
                if outcome.get("status") == "matured":
                    ratio = outcome.get("volume_ratio")
                    range_pct = outcome.get("range_pct")
                    taker_share = outcome.get("taker_buy_volume_share_proxy")
                    measurement = (f"{symbol} {outcome['return_pct']:+.3f}% / "
                                   f"vol {ratio:.2f}x" if ratio is not None else
                                   f"{symbol} {outcome['return_pct']:+.3f}% / vol unknown")
                    measurement += f" / range {range_pct:.3f}%" if range_pct is not None else " / range unknown"
                    if taker_share is not None:
                        measurement += (f" / taker-buy share {taker_share * 100:.1f}% "
                                        "candle-volume proxy (not true order-book)")
                    measurements.append(measurement)
                else:
                    status = outcome.get("report_status", outcome.get("status", "unknown"))
                    if status == "retryable_unavailable":
                        reason = outcome.get("last_attempt_status") or outcome.get("reason") or "unknown error"
                        measurements.append(f"{symbol} retryable unknown ({reason})")
                    elif status == "no_baseline_unknown":
                        measurements.append(f"{symbol} no-baseline unknown")
                    elif status == "not_yet_due_pending":
                        measurements.append(f"{symbol} not-yet-due pending")
                    elif status == "due_pending":
                        measurements.append(f"{symbol} due pending")
                    else:
                        measurements.append(f"{symbol} {status}")
            lines.append("  " + "; ".join(measurements) + f". Rationale: {event['rationale']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    register = sub.add_parser("register", help="Record a prospective root interpretation and actual ticker baseline")
    register.add_argument("--event-id", required=True, help="Stable root-chosen underlying event key")
    register.add_argument("--rationale", required=True)
    register.add_argument("--action", required=True, choices=("trade", "hold", "decline", "observe"))
    register.add_argument("--expected-direction", required=True, choices=("up", "down", "mixed", "unknown"))
    register.add_argument("--expected-reaction", required=True, choices=("elevated", "normal", "unknown"))
    register.add_argument("--confidence", required=True, choices=("low", "medium", "high"))
    register.add_argument("--evidence-context", required=True, choices=("supporting", "contrary", "unknown"))
    register.add_argument("--reasoning-version", required=True)
    register.add_argument("--source-url", action="append", required=True)
    register.add_argument("--event-time-utc", required=True, help="Offset-aware UTC or literal 'unknown'")
    register.add_argument("--publication-time-utc", required=True, help="Offset-aware UTC or literal 'unknown'")
    register.add_argument("--first-seen-clock-utc", required=True, help="Actual alert receipt clock or literal 'unknown'")
    register.add_argument("--forecast-minutes", type=int, choices=HORIZONS_MINUTES, default=60)
    sub.add_parser("update", help="Fetch one bounded candle window for each of up to three due events")
    sub.add_parser("report", help="Read-only report; no network or state writes")
    args = parser.parse_args(argv)
    if args.command == "register":
        result = register_interpretation(
            event_id=args.event_id, rationale=args.rationale, action=args.action,
            expected_direction=args.expected_direction, expected_reaction=args.expected_reaction,
            confidence=args.confidence, evidence_context=args.evidence_context,
            reasoning_version=args.reasoning_version, source_urls=args.source_url,
            event_time_utc=args.event_time_utc, publication_time_utc=args.publication_time_utc,
            first_seen_clock_utc=args.first_seen_clock_utc,
            forecast_minutes=args.forecast_minutes)
    elif args.command == "update":
        result = update_followups()
    else:
        result = news_followup_report()
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
