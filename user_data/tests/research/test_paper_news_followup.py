from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from user_data.Custom_Launcher.research import paper_news_followup as followups


_UNSET = object()


def _register(path: Path, monkeypatch, *, event_id="event-1", now=None, ticker=_UNSET,
              rationale="Sourced interpretation", action="decline",
              direction="up", version="v1", event_time="unknown",
              publication_time="unknown", first_seen="unknown"):
    now = now or datetime(2026, 10, 1, tzinfo=timezone.utc)
    monkeypatch.setattr(followups, "_ticker_prices",
                        lambda payload: payload if isinstance(payload, dict) else {})
    return followups.register_interpretation(
        event_id=event_id, rationale=rationale, action=action,
        expected_direction=direction, expected_reaction="elevated", confidence="medium",
        evidence_context="supporting", reasoning_version=version,
        source_urls=["https://example.com/story"], event_time_utc=event_time,
        publication_time_utc=publication_time, first_seen_clock_utc=first_seen,
        forecast_minutes=60, now=now,
        fetch_json=lambda _url, _timeout: ({"BTCUSDT": 100.0, "ETHUSDT": 20.0, "SOLUSDT": 5.0}
            if ticker is _UNSET else ticker), state_path=path)


def _candle(open_at: datetime, close: float = 100.0) -> list:
    opened_ms = int(open_at.timestamp() * 1000)
    return [opened_ms, "100", str(max(101.0, close)), "99", str(close), "10",
            opened_ms + 59_999, "1000", 100, "5", "500", "0"]


def test_duplicate_only_attaches_evidence_and_report_is_read_only(tmp_path, monkeypatch):
    state_path = tmp_path / "followups.json"
    now = datetime(2026, 10, 1, tzinfo=timezone.utc)
    _register(state_path, monkeypatch, now=now)
    before, _ = followups._state(state_path)
    baseline = before["events"]["event-1"]["baselines"]["BTCUSDT"].copy()

    result = followups.register_interpretation(
        event_id="event-1", rationale="New corroborating source", action="hold",
        expected_direction="down", expected_reaction="normal", confidence="low",
        evidence_context="contrary", reasoning_version="v2",
        source_urls=["https://example.com/update"], event_time_utc="unknown",
        publication_time_utc="unknown", first_seen_clock_utc="unknown",
        forecast_minutes=240, now=now + timedelta(minutes=1),
        fetch_json=lambda *_: pytest.fail("duplicate must not refresh a baseline"),
        state_path=state_path)
    after, _ = followups._state(state_path)
    event = after["events"]["event-1"]
    assert result["status"] == "duplicate_evidence_attached"
    assert result["independent_event_count_changed"] is False
    assert event["action"] == "decline"
    assert event["expected_direction"] == "up"
    assert event["forecast_minutes"] == 60
    assert event["baselines"]["BTCUSDT"] == baseline
    assert event["event_time_utc"] is None
    assert event["publication_time_utc"] is None
    assert event["first_seen_clock_utc"] is None
    assert len(event["duplicate_evidence_notes"]) == 1

    raw_before = state_path.read_bytes()
    monkeypatch.setattr(followups, "_fetch_json", lambda *_: pytest.fail("report must not fetch"))
    report = followups.news_followup_report(state_path=state_path)
    assert report["independent_underlying_event_count"] == 1
    assert state_path.read_bytes() == raw_before


def test_closed_horizon_rounding_and_prior_volume_exclude_partial_baseline_bar():
    baseline_at = datetime(2026, 10, 1, 12, 0, 30, tzinfo=timezone.utc)
    target = baseline_at + timedelta(minutes=30)
    first_open = baseline_at.replace(second=0, microsecond=0) + timedelta(minutes=1)
    last_open = target.replace(second=0, microsecond=0) - timedelta(minutes=1)
    count = int((last_open - first_open).total_seconds() // 60) + 1
    prior_first = baseline_at.replace(second=0, microsecond=0) - timedelta(minutes=count)
    all_opens = []
    cursor = prior_first
    while cursor <= last_open:
        all_opens.append(cursor)
        cursor += timedelta(minutes=1)
    rows = [_candle(open_at, 101.0 if open_at >= first_open else 100.0)
            for open_at in all_opens]
    now = target + timedelta(minutes=1)
    candle_map, error = followups._candle_map(
        rows, now, baseline_at - timedelta(hours=4), baseline_at + timedelta(hours=4))
    assert error is None
    measured, error = followups._measure_horizon(
        candle_map, {"observed_at_utc": baseline_at.isoformat(), "price_usdt": 100.0}, 30, now)

    assert error is None
    assert measured["status"] == "matured"
    assert measured["complete_post_baseline_minutes"] == 29
    assert measured["baseline_minute_rounding_seconds"] == 30
    assert measured["excluded_partial_first_minute"] is True
    assert measured["excluded_terminal_partial_minute"] is True
    assert measured["measurement_ended_at_utc"] == (last_open + timedelta(minutes=1)).isoformat()
    assert measured["return_pct"] == pytest.approx(1.0)
    assert measured["volume_ratio"] == pytest.approx(1.0)
    assert measured["taker_buy_volume_share_proxy"] == pytest.approx(0.5)


def test_failed_quotes_stay_unknown_and_failed_or_gapped_windows_retry_fairly(tmp_path, monkeypatch):
    state_path = tmp_path / "followups.json"
    t0 = datetime(2026, 10, 1, tzinfo=timezone.utc)
    _register(state_path, monkeypatch, event_id="unknown-quote", now=t0, ticker=None)
    # A failed quote must stay unknown; it is never replaced with candle open.
    monkeypatch.setattr(followups, "_ticker_prices", lambda _payload: {})
    followups.register_interpretation(
        event_id="failed-quote", rationale="No usable ticker", action="observe",
        expected_direction="unknown", expected_reaction="unknown", confidence="low",
        evidence_context="unknown", reasoning_version="v1", source_urls=["https://example.com/a"],
        event_time_utc="unknown", publication_time_utc="unknown", first_seen_clock_utc="unknown",
        now=t0, fetch_json=lambda *_: None, state_path=state_path)
    state, _ = followups._state(state_path)
    failed = state["events"]["failed-quote"]
    assert failed["baselines"]["BTCUSDT"]["status"] == "unknown"
    assert failed["baselines"]["BTCUSDT"]["price_usdt"] is None
    assert failed["outcomes"]["30"]["BTCUSDT"]["status"] == "unavailable"

    for index in range(3):
        _register(state_path, monkeypatch, event_id=f"due-{index}", now=t0)
    due_time = t0 + timedelta(minutes=31)
    failed_update = followups.update_followups(
        now=due_time, fetch_json=lambda *_: None, state_path=state_path, max_events=3)
    assert failed_update["events_considered"] == 3
    assert failed_update["requests"] == 9
    state, _ = followups._state(state_path)
    assert state["events"]["due-0"]["outcomes"]["30"]["BTCUSDT"]["status"] == "pending"
    assert state["events"]["due-0"]["outcomes"]["30"]["BTCUSDT"]["last_attempt_status"] == "fetch_failed"

    # With an empty but valid response, the next run records a retryable gap.
    next_update = followups.update_followups(
        now=due_time + timedelta(minutes=1), fetch_json=lambda *_: [],
        state_path=state_path, max_events=3)
    assert next_update["events_considered"] == 3
    state, _ = followups._state(state_path)
    assert state["events"]["due-0"]["outcomes"]["30"]["BTCUSDT"]["status"] == "pending"
    assert state["events"]["due-0"]["outcomes"]["30"]["BTCUSDT"]["last_attempt_status"] == "coverage_gap"
    assert "last_update_attempt_at_utc" in state["events"]["due-2"]


def test_retry_queue_clock_keeps_older_retry_ahead_of_new_due_events(tmp_path, monkeypatch):
    state_path = tmp_path / "followups.json"
    start = datetime(2026, 10, 1, tzinfo=timezone.utc)
    _register(state_path, monkeypatch, event_id="retry-old", now=start)

    first_attempt = start + timedelta(minutes=31)
    followups.update_followups(now=first_attempt, fetch_json=lambda *_: None,
                               state_path=state_path, max_events=1)
    _register(state_path, monkeypatch, event_id="new-due-1",
              now=start + timedelta(minutes=32))
    _register(state_path, monkeypatch, event_id="new-due-2",
              now=start + timedelta(minutes=33))

    retry_clock = start + timedelta(minutes=64)
    result = followups.update_followups(now=retry_clock, fetch_json=lambda *_: None,
                                        state_path=state_path, max_events=1)
    state, _ = followups._state(state_path)

    assert result["events_considered"] == 1
    assert result["requests"] == 3
    assert state["events"]["retry-old"]["last_update_attempt_at_utc"] == retry_clock.isoformat()
    assert "last_update_attempt_at_utc" not in state["events"]["new-due-1"]
    assert "last_update_attempt_at_utc" not in state["events"]["new-due-2"]


def test_report_labels_retryable_unknown_and_formats_range_and_taker_proxy(tmp_path, monkeypatch):
    state_path = tmp_path / "followups.json"
    start = datetime(2026, 10, 1, tzinfo=timezone.utc)
    _register(state_path, monkeypatch, now=start)
    state, _ = followups._state(state_path)
    outcomes = state["events"]["event-1"]["outcomes"]["60"]
    outcomes["BTCUSDT"] = {
        "status": "matured", "return_pct": 1.0, "volume_ratio": 1.2,
        "range_pct": 2.5, "taker_buy_volume_share_proxy": 0.55,
    }
    outcomes["ETHUSDT"].update({
        "last_attempt_status": "coverage_gap", "reason": "coverage_gap",
        "last_attempt_at_utc": (start + timedelta(minutes=61)).isoformat(),
    })
    outcomes["SOLUSDT"] = {
        "status": "unavailable", "reason": "baseline_quote_unknown",
        "target_at_utc": None,
    }
    followups._save(state, state_path)

    report = followups.news_followup_report(
        state_path=state_path, now=start + timedelta(hours=2))
    btc_60 = report["per_coin_horizon"]["BTCUSDT"]["60"]
    eth_outcome = report["recent_events"][0]["intended_outcomes"]["ETHUSDT"]
    text = followups.format_news_followup_report(report)

    assert btc_60["retryable_unavailable"] == 0
    assert report["per_coin_horizon"]["ETHUSDT"]["60"]["retryable_unavailable"] == 1
    assert report["per_coin_horizon"]["SOLUSDT"]["60"]["unknown"] == 1
    assert eth_outcome["status"] == "pending"
    assert eth_outcome["report_status"] == "retryable_unavailable"
    assert eth_outcome["last_attempt_status"] == "coverage_gap"
    assert "ETHUSDT retryable unknown (coverage_gap)" in text
    assert "SOLUSDT no-baseline unknown" in text
    assert "range 2.500%" in text
    assert "taker-buy share 55.0% candle-volume proxy (not true order-book)" in text


def test_chronology_uses_disjoint_comparable_btc_intended_calls(tmp_path, monkeypatch):
    state_path = tmp_path / "followups.json"
    start = datetime(2026, 10, 1, tzinfo=timezone.utc)
    for index in range(20):
        _register(state_path, monkeypatch, event_id=f"event-{index:02d}",
                  now=start + timedelta(minutes=index), direction="up", version="v1")
    state, _ = followups._state(state_path)
    for event in state["events"].values():
        event["outcomes"]["60"]["BTCUSDT"] = {
            "status": "matured", "return_pct": 0.1, "volume_ratio": 1.2,
        }
    followups._save(state, state_path)

    report = followups.news_followup_report(state_path=state_path)
    chronology = report["btc_primary_chronology"]
    assert chronology["status"] == "descriptive_comparison"
    assert chronology["disjoint"] is True
    assert chronology["first_10"]["n"] == chronology["latest_10"]["n"] == 10
    assert chronology["first_10"]["raw_direction_agreement_count"] == 10
    assert report["per_coin_horizon"]["BTCUSDT"]["60"]["raw_direction_agreement_count"] == 20
    assert report["per_coin_horizon"]["BTCUSDT"]["30"]["raw_direction_agreement_count"] == 0


def test_account_table_option_appends_compact_read_only_section(monkeypatch, capsys):
    from user_data.Custom_Launcher.research import paper_trial_snapshot as snapshot

    monkeypatch.setattr(snapshot, "account_snapshot", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(followups, "news_followup_report", lambda: {
        "status": "no_ledger", "independent_underlying_event_count": 0,
        "action_counts": {"trade": 0, "hold": 0, "decline": 0, "observe": 0},
        "per_coin_horizon": {}, "btc_primary_chronology": {"status": "insufficient_sample"},
        "recent_events": [],
    })
    monkeypatch.setattr("sys.argv", ["paper_trial_snapshot", "--account-table"])
    snapshot.main()
    output = capsys.readouterr().out
    assert "| TOTAL |" in output
    assert "## News interpretation follow-ups" in output


def test_gap_is_not_matured_and_duplicate_candles_fail_strict_coverage():
    now = datetime(2026, 10, 1, 1, tzinfo=timezone.utc)
    opened = now.replace(minute=0, second=0, microsecond=0) - timedelta(minutes=1)
    row = _candle(opened)
    candles, error = followups._candle_map(
        [row, row], now, opened - timedelta(minutes=1), opened + timedelta(minutes=1))
    assert candles == {}
    assert error == "duplicate_candle"
    measurement, error = followups._measure_horizon(
        {}, {"observed_at_utc": opened.isoformat(), "price_usdt": 100}, 30, now)
    assert measurement is None
    assert error == "coverage_gap"


def test_naive_and_future_clocks_are_rejected_before_recording(tmp_path, monkeypatch):
    state_path = tmp_path / "followups.json"
    with pytest.raises(ValueError, match="timezone-aware"):
        _register(state_path, monkeypatch, now=datetime(2026, 10, 1))
    with pytest.raises(ValueError, match="cannot be in the future"):
        _register(state_path, monkeypatch,
                  now=datetime(2026, 10, 1, tzinfo=timezone.utc),
                  event_time="2026-10-02T00:00:00+00:00")
    assert not state_path.exists()
