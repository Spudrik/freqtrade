"""Small read-only paper/source snapshot; --record-daily explicitly writes one daily history file.

Reads bounded local rows and existing account helpers, not historical research
exports. Optional public price lookup is only for estimated current open P/L.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timedelta, timezone
import hashlib
import json
from math import isclose, isfinite
from pathlib import Path
import re
import sqlite3
from urllib.parse import urlencode, urlparse
from urllib.request import urlopen
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import Session, selectinload
from freqtrade.enums import TradingMode
from freqtrade.persistence import Order, Trade
from user_data.Custom_Launcher.research.paper_trial_runtime import ROOT, REPORT, RECORD, BASE, ACCOUNTS, account_record, account_specs_for_record, lifecycle_for, process_inventory, matching_processes, worker_heartbeat_state
from user_data.Custom_Launcher.research.paper_trial_control import _configs, _api, _journal, _verify_running_account
from freqtrade.configuration.load_config import load_from_files
from user_data.Custom_Launcher.research.crypto_derivatives_snapshot import collect_crypto_derivatives_snapshot
from user_data.strategies.integrated_paper_context import _utc, load_luna_context
from user_data.strategies.paper_fast_context import CONTROL_FILE, load_fast_control
from user_data.strategies.paper_fast_context import CONTROL_JOURNAL, parse_fast_control
from user_data.strategies.paper_news_manual import decision_rows
from user_data.strategies.paper_trial_level_orderbook import _recent_pressure
from user_data.strategies.paper_aggressive_context import (
    AGGRESSIVE_ACCOUNT_KEYS, AGGRESSIVE_ACCOUNTS, CONTROL_FILES,
    load_aggressive_control, validate_aggressive_filled_plan,
)


GLOBAL_CONTEXT_SOURCE_LIMIT = 40
GLOBAL_CONTEXT_METRIC_LIMITS = {
    "deribit_options": 18,
    "hyperliquid_clearinghouse": 7,
    "bybit_eth_native_wallet": 5,
    "bybit_eth_erc20_wallet": 5,
    "bybit_btc_wallet": 5,
}
GLOBAL_CONTEXT_DEFAULT_METRIC_LIMIT = 20
PAPER_DAILY_HISTORY = REPORT / "paper_account_daily.json"
BTC_PERPETUAL_SOURCE = "Binance USD-M BTCUSDT perpetual price proxy; not spot buy-and-hold"
PAPER_DAILY_OBSERVATION_POLICY = "first actual report observation per UTC date; timestamp is not exact midnight or a full 24-hour observation"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _web_headline_window():
    path = ROOT/"user_data"/"Custom_Launcher"/"collectors"/"context"/"config"/"web_research_sources.json"
    config = read_json(path)
    if not isinstance(config, dict):
        raise ValueError(f"Web research config must be an object: {path}")
    seconds = config.get("poll_interval_seconds")
    if type(seconds) is not int or seconds <= 0:
        raise ValueError(f"Web research poll_interval_seconds must be a positive integer: {path}")
    try:
        return timedelta(seconds=seconds), seconds
    except OverflowError as exc:
        raise ValueError(f"Web research poll_interval_seconds is out of range: {path}") from exc


def _timestamp_age(value, now):
    if not isinstance(value, str) or not value.strip():
        return None, "missing", None
    try:
        timestamp = _utc(value)
    except (AttributeError, TypeError, ValueError, OverflowError):
        return None, "invalid", None
    age = now - timestamp
    if age < timedelta(0):
        return None, "future", timestamp
    return round(age.total_seconds()/3600, 2), "known", timestamp


def source_snapshot(now):
    output, headlines, seen, titles = {}, [], set(), {}
    if now.tzinfo is None:
        raise ValueError("Source snapshot clock must be timezone-aware")
    now = now.astimezone(timezone.utc)
    web_window, web_window_seconds = _web_headline_window()
    for key, folder in (("news","collector_data/news"),("web","collector_data/web"),
                        ("global","collector_data/global_context"),("orderbook","collector_data/orderbook")):
        path = ROOT/"user_data"/folder/"collector_status.json"
        if not path.is_file():
            output[key] = {"status":"missing"}; continue
        status = read_json(path)
        output[key] = {name:status.get(name) for name in ("status","heartbeat_at","last_fetch_at","last_message_at","last_error","sources_enabled","sources_total","capacity_level") if name in status}
        if key == "web":
            output[key]["headline_window_seconds"] = web_window_seconds
        db = Path(status["db_path"]).resolve()
        with closing(sqlite3.connect(db.as_uri()+"?mode=ro",uri=True,timeout=2)) as con:
            con.row_factory = sqlite3.Row
            if key in {"news","web"}:
                window = timedelta(hours=4) if key == "news" else web_window
                cutoff = (now-window).isoformat()
                rows = con.execute("SELECT title,canonical_url,source_url,published_at,collected_at,source_group FROM articles WHERE collected_at>=? ORDER BY collected_at DESC LIMIT 16",(cutoff,)).fetchall()
                for row in rows:
                    item=dict(row); url=item["canonical_url"] or item["source_url"]
                    collected_age, collected_status, collected_at = _timestamp_age(item["collected_at"], now)
                    if collected_status != "known" or now-collected_at > window:
                        continue
                    if not url or url in seen:
                        continue
                    seen.add(url)
                    title = " ".join(str(item["title"] or "").casefold().split())
                    if not title:
                        continue
                    published_age, published_status, _ = _timestamp_age(item["published_at"], now)
                    clocks = {"published_age_hours":published_age,"published_age_status":published_status,
                              "collected_age_hours":collected_age,"collected_age_status":collected_status}
                    duplicate = titles.get(title)
                    if duplicate is not None:
                        duplicate["additional_sources"].append({"url":url,"published_at":item["published_at"],
                            "collected_at":item["collected_at"],"source_group":item["source_group"],**clocks})
                        continue
                    entry = {"collector":key,"title":item["title"],"url":url,
                        "published_at":item["published_at"],"collected_at":item["collected_at"],
                        "source_group":item["source_group"],"additional_sources":[],**clocks}
                    titles[title] = entry
                    headlines.append(entry)
            elif key=="global":
                source_count=int(con.execute("SELECT COUNT(*) FROM context_sources").fetchone()[0])
                sources=con.execute(
                    "SELECT source_id,source_group,source_type,enabled,url,last_success_at,last_failure_at,last_error,last_notes "
                    "FROM context_sources ORDER BY source_id LIMIT ?",
                    (GLOBAL_CONTEXT_SOURCE_LIMIT,),
                ).fetchall()
                output["global"]["sources"]=[]
                output["global"]["source_count"]=source_count
                output["global"]["source_limit"]=GLOBAL_CONTEXT_SOURCE_LIMIT
                output["global"]["sources_truncated"]=source_count>len(sources)
                for source in sources:
                    tick=con.execute("SELECT ts,source_ts,metric_key,value,unit,notes FROM global_context_ticks WHERE source_id=? ORDER BY ts DESC LIMIT 1",(source["source_id"],)).fetchone()
                    metric_limit=GLOBAL_CONTEXT_METRIC_LIMITS.get(
                        str(source["source_type"] or ""), GLOBAL_CONTEXT_DEFAULT_METRIC_LIMIT
                    )
                    metric_count=0
                    metrics=[]
                    if source["enabled"]:
                        metric_count=int(con.execute(
                            "SELECT COUNT(DISTINCT metric_key) FROM global_context_ticks WHERE source_id=?",
                            (source["source_id"],),
                        ).fetchone()[0])
                        metrics=con.execute(
                            "WITH latest AS ("
                            "SELECT metric_key,MAX(id) AS id FROM global_context_ticks "
                            "WHERE source_id=? GROUP BY metric_key) "
                            "SELECT t.ts,t.source_ts,t.metric_key,t.value,t.unit,t.notes "
                            "FROM global_context_ticks t JOIN latest ON latest.id=t.id "
                            "WHERE t.source_id=? "
                            "ORDER BY COALESCE(NULLIF(t.source_ts,''),t.ts) DESC,t.ts DESC,t.metric_key "
                            "LIMIT ?",
                            (source["source_id"],source["source_id"],metric_limit),
                        ).fetchall()
                    latest_metrics=[]
                    for metric in metrics:
                        item=dict(metric)
                        notes=item["notes"]
                        item["notes_truncated"]=notes is not None and len(notes)>500
                        if item["notes_truncated"]:
                            item["notes"]=notes[:500]
                        latest_metrics.append(item)
                    output["global"]["sources"].append({**dict(source),
                        "last_observation_at":None if tick is None else tick["source_ts"],
                        "last_collected_at":None if tick is None else tick["ts"],
                        "latest_metrics":latest_metrics,
                        "metric_count":metric_count,
                        "metric_limit":metric_limit,
                        "metrics_truncated":metric_count>len(metrics)})
    output["recent_unique_headlines"] = headlines[:20]
    output["book_pressure"] = {pair:{"imbalance":value,"coverage":coverage} for pair in
        ("BTC/USDT:USDT","ETH/USDT:USDT","SOL/USDT:USDT","BNB/USDT:USDT","DOGE/USDT:USDT","1000PEPE/USDT:USDT")
        for value,coverage in [_recent_pressure(pair,now)]}
    return output


def market_snapshot():
    record = read_json(RECORD)
    readers = {spec.key: spec for spec in account_specs_for_record(record)
               if spec.key in {"news_manual", "news_lab", "news_fast", "manual"}
               and lifecycle_for(record, spec) == "ACTIVE"}
    account = next((key for key in ("news_manual", "news_lab", "news_fast", "manual")
                    if key in readers), None)
    if account is None:
        raise RuntimeError("No ACTIVE approved paper candle reader is registered")
    _,overlay=_configs(account)
    _verify_running_account(overlay)
    output={}
    for pair in sorted(readers[account].pairs):
        data=_api(overlay,"pair_candles?"+urlencode({"pair":pair,"timeframe":"1h","limit":4}))
        selected={name:index for index,name in enumerate(data["columns"]) if name in
            {"date","open","high","low","close","volume","paper_atr","vp_poc_4h","vp_hvn_above_4h","vp_hvn_below_4h","vp_lvn_above_4h","vp_lvn_below_4h","rolling_20_high_4h","rolling_20_low_4h","prior_24h_high","prior_24h_low"}}
        output[pair]=[{name:row[index] for name,index in selected.items()} for row in data["data"][-4:]]
    return {"source":f"Existing verified local {account} read-only analyzed-candle API",
            "account": account,
            "timeframe":"1h; embedded completed 4h levels where available",
            "daily_candle_coverage":"not supplied by this local API; do not invent a daily trend", "pairs":output}


def _watch_times(watch):
    raw = watch.get("proposed_checks_utc")
    if not isinstance(raw, list):
        raw = [watch.get("check_at_utc")]
    times, errors = set(), []
    for value in raw:
        if not isinstance(value, str):
            errors.append("Missing/non-text approved check time")
            continue
        for part in re.split(r"\s+and\s+", value.strip(), flags=re.IGNORECASE):
            try:
                times.add(_utc(part))
            except (TypeError, ValueError, OverflowError):
                errors.append(f"Invalid approved check time: {part}")
    return sorted(times), errors


def _latest_manual_news_decision(account, bot_name):
    """Read one journal's latest exact-account transition; malformed data fails closed."""
    journal = _journal(account)
    journal_state = "observed" if journal.is_file() else "missing"
    try:
        rows = decision_rows(journal)
    except (OSError, ValueError, TypeError):
        return "malformed", 0, None
    try:
        indexed = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError("Decision journal row must be an object")
            if row.get("account") == bot_name:
                indexed.append((_utc(row["at_utc"]), index, row))
    except (KeyError, TypeError, ValueError, OverflowError):
        return "malformed", len(rows), None
    if not indexed:
        return journal_state, len(rows), None
    _, _, row = max(indexed, key=lambda item: (item[0], item[1]))
    latest = {key: row.get(key) for key in
              ("decision_id", "action", "status", "at_utc", "luna_observed_at_utc")}
    return journal_state, len(rows), latest


def _manual_news_review_snapshot(record, now, account):
    """Bounded handoff for one exact manual-news account; no network scan."""
    spec = next((item for item in account_specs_for_record(record) if item.key == account), None)
    if spec is None:
        return None
    journal = _journal(account)
    journal_state, journal_row_count, latest = _latest_manual_news_decision(account, spec.bot_name)
    info = account_record(record, spec)
    database = REPORT / info["db"]
    if not database.is_file():
        database_state = "not_initialized" if not info.get("database_initialized_at_utc") else "missing"
        positions = None
    else:
        database_state = "initialized"
        positions = []
        uri = database.resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as connection:
            rows = connection.execute(
                "SELECT t.id,t.pair,t.is_short,t.leverage,t.stop_loss,c.cd_value "
                "FROM trades t LEFT JOIN trade_custom_data c "
                "ON c.ft_trade_id=t.id AND c.cd_key='paper_news_manual_plan' WHERE t.is_open=1"
            ).fetchall()
        from user_data.strategies.paper_news_manual import validate_manual_plan
        for trade_id, pair, is_short, leverage, stop_price, encoded in rows:
            plan = json.loads(encoded) if encoded is not None else None
            state = "stored_valid"
            try:
                validate_manual_plan(plan, float(plan["reference_rate"]), spec.bot_name,
                                     actual_leverage=float(leverage))
                if plan["pair"] != pair or (plan["side"] == "short") != bool(is_short):
                    raise ValueError("Persisted plan does not match its position")
            except (ValueError, TypeError, KeyError):
                state = "missing_or_invalid"
                plan = None
            positions.append({"trade_id": trade_id, "pair": pair, "side": "short" if is_short else "long",
                "leverage": leverage, "stop_price": stop_price, "protection_state": state, "stored_plan": plan})
    lifecycle = lifecycle_for(record, spec)
    return {"account": spec.key, "lifecycle": lifecycle,
        "required": lifecycle == "ACTIVE",
        "review_policy": ("best_guess_each_four_hour_wake" if account == "news_fast"
                          else "consider_each_fresh_luna_handoff"),
        "database_state": database_state, "journal": str(journal), "journal_state": journal_state,
        "journal_row_count": journal_row_count, "latest_decision": latest, "open_positions": positions,
        "observed_at_utc": now.isoformat()}


def _news_fast_review_snapshot(record, now):
    return _manual_news_review_snapshot(record, now, "news_fast")


def _news_input_reviews(record, now, aggressive_controls, fast_context, manual_reviews):
    """Explicit per-handoff decision coverage, independent of narrative change."""
    issues = []
    try:
        luna = load_luna_context(now)
    except (OSError, KeyError, TypeError, ValueError, OverflowError):
        luna_status, clock = "malformed", None
        issues.append("malformed_luna_context")
    else:
        luna_status = luna.status
        clock = luna.observed_at.isoformat() if luna.status == "observed" else None
    reviews = []
    for review in manual_reviews:
        if review and review["lifecycle"] == "ACTIVE":
            latest = review.get("latest_decision")
            latest = latest if isinstance(latest, dict) else {}
            reviewed = (clock is not None and latest.get("luna_observed_at_utc") == clock
                        and latest.get("status") in {"submitted", "recorded_no_action", "recorded_protection"})
            reviews.append({"account": review["account"], "kind": "manual_news",
                            "considered_for_current_luna": reviewed, "required": not reviewed})
    for review in aggressive_controls:
        if review["lifecycle"] == "ACTIVE":
            reviewed = (clock is not None
                        and review.get("luna_observed_at_utc") == clock
                        and review["status"] == "observed")
            reviews.append({"account": review["account"], "kind": "context_control",
                            "considered_for_current_luna": reviewed,
                            "required": not reviewed or review.get("renewal_required", True)})
    if fast_context["lifecycle"] == "ACTIVE":
        reviewed = (clock is not None and fast_context.get("luna_observed_at_utc") == clock
                    and fast_context["status"] == "observed")
        expires = fast_context.get("valid_until_utc")
        try:
            nearing_expiry = expires is None or _utc(expires) <= now + timedelta(minutes=30)
        except (TypeError, ValueError, OverflowError):
            nearing_expiry = True
        reviews.append({"account": "fast_context", "kind": "context_control",
                        "considered_for_current_luna": reviewed, "required": not reviewed or nearing_expiry})
    # The original auto-with-overrides account keeps its narrower authority.
    original = next(spec for spec in account_specs_for_record(record) if spec.key == "manual")
    if lifecycle_for(record, original) == "ACTIVE":
        _, _, latest = _latest_manual_news_decision("manual", original.bot_name)
        latest = latest or {}
        reviewed = (clock is not None and latest.get("luna_observed_at_utc") == clock
                    and latest.get("status") in {"submitted", "recorded_no_action", "recorded_protection"})
        reviews.append({"account": "manual", "kind": "narrow_news_override",
                        "considered_for_current_luna": reviewed, "required": not reviewed})
    return {"luna_status": luna_status, "luna_observed_at_utc": clock,
            "required": any(row["required"] for row in reviews), "accounts": reviews,
            "issues": issues}


def review_context_snapshot(now, record=None):
    """Current identities/watches/control only; no health scan or record writes."""
    record = read_json(RECORD) if record is None else record
    approved, due, issues = [], [], []
    for watch in record.get("announcement_watches", []):
        if not isinstance(watch, dict):
            issues.append("Malformed announcement watch")
            continue
        status = str(watch.get("status", "")).casefold()
        if "approved" not in status or any(word in status for word in ("declined","observed","closed","retired")):
            continue
        times, errors = _watch_times(watch)
        issues.extend({"event_id":watch.get("event_id"),"error":error} for error in errors)
        current = [stamp for stamp in times if now-timedelta(hours=4) <= stamp <= now+timedelta(hours=8)]
        if current:
            approved.append({"event_id":watch.get("event_id"),"name":watch.get("name"),
                "status":status,"source":watch.get("source"),
                "checks_utc":[stamp.isoformat() for stamp in current]})
        for stamp in current:
            if now-timedelta(hours=4) <= stamp <= now:
                due.append({"event_id":watch.get("event_id"),"check_at_utc":stamp.isoformat(),"source":watch.get("source")})
    fast_context_spec = next(spec for spec in account_specs_for_record(record) if spec.key == "fast_context")
    fast_context_lifecycle = lifecycle_for(record,fast_context_spec)
    control = load_fast_control(now) if fast_context_lifecycle == "ACTIVE" else None
    control_times = {}
    if fast_context_lifecycle == "ACTIVE" and CONTROL_FILE.is_file():
        try:
            encoded = read_json(CONTROL_FILE)
            if isinstance(encoded, dict):
                control_times = {key:encoded.get(key) for key in
                                 ("observed_at_utc","valid_until_utc","luna_observed_at_utc")}
            else:
                issues.append("Fast control is not an object; preserve malformed-control entry block")
        except (ValueError, TypeError):
            issues.append("Fast control clock unavailable; preserve malformed-control entry block")
    fast_context = ({"status":control.status,"lifecycle":fast_context_lifecycle,
            "decision_id":control.decision_id,"bias":control.bias,"exposure":control.exposure,
            "entry_permission":control.entry_permission,"long_leverage":control.leverage_for("long",now),
            "short_leverage":control.leverage_for("short",now),
            "blackouts":[{"start_utc":start.isoformat(),"end_utc":end.isoformat()} for start,end in control.blackouts],
            **control_times} if control is not None else {
            "status":"not_needed_for_lifecycle","lifecycle":fast_context_lifecycle,
            "renewal_required":False,"stored_control_not_read":True})
    aggressive_controls = []
    for key in AGGRESSIVE_ACCOUNT_KEYS:
        spec = next((item for item in account_specs_for_record(record) if item.key == key), None)
        if spec is None:
            continue
        lifecycle = lifecycle_for(record, spec)
        if lifecycle != "ACTIVE":
            aggressive_controls.append({"account":key,"lifecycle":lifecycle,
                "status":"not_needed_for_lifecycle","renewal_required":False,"stored_control_not_read":True})
            continue
        control = load_aggressive_control(key, now)
        expires = None
        luna_clock = None
        path = CONTROL_FILES[key]
        if path.is_file():
            try:
                encoded = read_json(path)
                if isinstance(encoded, dict) and encoded.get("account") == AGGRESSIVE_ACCOUNTS[key]["bot_name"]:
                    expires = encoded.get("valid_until_utc")
                    luna_clock = encoded.get("luna_observed_at_utc")
            except (ValueError, TypeError, OSError):
                issues.append(f"{key}: control clock unavailable; technical-only 3x mode remains active")
        renewal = control.status != "observed"
        if expires:
            try:
                renewal = renewal or _utc(expires) <= now + timedelta(minutes=30)
            except (TypeError, ValueError, OverflowError):
                renewal = True
        aggressive_controls.append({"account":key,"lifecycle":lifecycle,"status":control.status,
            "technical_only":control.technical_only,"renewal_required":renewal,
            "decision_id":control.decision_id,"bias":control.bias,
            "side_permission":control.side_permission,"long_leverage_cap":control.long_leverage_cap,
            "short_leverage_cap":control.short_leverage_cap,"exposure":control.exposure,
            "valid_until_utc":expires, "luna_observed_at_utc": luna_clock})
    manual_reviews = [_manual_news_review_snapshot(record, now, key)
                      for key in ("news_manual", "news_lab")]
    news_fast = _news_fast_review_snapshot(record, now)
    if news_fast is not None:
        manual_reviews.append(news_fast)
    news_input_reviews = _news_input_reviews(record, now, aggressive_controls, fast_context, manual_reviews)
    issues.extend({"news_input_review": issue} for issue in news_input_reviews["issues"])
    return {"trial":record.get("trial"),"run_record_status":record.get("status"),
        "accounts":[{"account":spec.key,"lifecycle":lifecycle_for(record,spec),"strategy":spec.strategy,"config":spec.config,
            "db":account_record(record,spec)["db"],"log":account_record(record,spec)["log"],
            "api_port":spec.port} for spec in account_specs_for_record(record)],
        "approved_watches":approved,"due_watches":due,"issues":issues,
        "fast_context":fast_context,"aggressive_controls":aggressive_controls,
        "manual_news_reviews": manual_reviews,
        "news_input_reviews": news_input_reviews,
        "news_fast_review":news_fast}


def _candle_facts(candles, now, hours):
    """Facts only: closed, fresh and consecutive candles; never a trend verdict."""
    if not candles:
        return {"state":"unavailable","completed_candle_count":0}
    last_close = datetime.fromtimestamp(int(candles[-1][6])/1000,timezone.utc)
    age_minutes = (now-last_close).total_seconds()/60
    consecutive = all(int(b[0])-int(a[0]) == hours*3_600_000 for a,b in zip(candles,candles[1:]))
    recent = candles[-3:]
    high, low = max(float(row[2]) for row in recent), min(float(row[3]) for row in recent)
    start, close = float(recent[0][1]), float(recent[-1][4])
    state = "observed" if consecutive and age_minutes <= hours*60+5 else "stale_or_gapped"
    return {"state":state,"completed_candle_count":len(recent),"last_close_utc":last_close.isoformat(),
        "age_minutes":age_minutes,"last_candle_change_pct":100*(close/float(recent[-1][1])-1),
        "previous_candle_change_pct":None if len(recent)<2 else 100*(float(recent[-2][4])/float(recent[-2][1])-1),
        "three_candle_change_pct":100*(close/start-1) if len(recent)==3 else None,
        "latest_close_position_in_recent_range":None if high<=low else (close-low)/(high-low)}


def higher_snapshot(now):
    """Four bounded public requests, only when the reviewer requests this gap fill."""
    output=[]
    for symbol in ("BTCUSDT", "ETHUSDT"):
        for interval,hours in (("4h",4),("1d",24)):
            url="https://fapi.binance.com/fapi/v1/klines?"+urlencode({"symbol":symbol,"interval":interval,"limit":9})
            with urlopen(url,timeout=12) as response:
                candles=json.load(response)
            completed=[row for row in candles if int(row[6]) < now.timestamp()*1000]
            if len(completed)<2:
                raise ValueError(f"Insufficient completed {symbol} {interval} candles")
            first,last=completed[0],completed[-1]
            output.append({"symbol":symbol,"timeframe":interval,"source":url,
                "last_close_utc":datetime.fromtimestamp(int(last[6])/1000,timezone.utc).isoformat(),
                "close":float(last[4]),"last_candle_change_pct":100*(float(last[4])/float(last[1])-1),
                "recent_change_pct":100*(float(last[4])/float(first[1])-1),
                "recent_completed_hours":hours*len(completed),
                "recent_high":max(float(row[2]) for row in completed),
                "recent_low":min(float(row[3]) for row in completed),
                "completed_candle_facts":_candle_facts(completed,now,hours)})
    return output


def _protection_state(trade, session, account=None):
    """Reuse the strategy's persisted-plan validators, including inverse/manual bots."""
    if not isfinite(float(trade.stop_loss or 0)) or float(trade.stop_loss or 0) <= 0:
        return "missing_or_invalid"
    try:
        if trade.strategy in {"PaperFastAuto","PaperFastContext","PaperFastLevelInverse"}:
            from user_data.strategies.paper_fast_reaction import validate_fast_plan
            encoded = session.execute(text("SELECT cd_value FROM trade_custom_data WHERE ft_trade_id=:trade_id AND cd_key=:key"),
                {"trade_id":trade.id,"key":"paper_fast_plan"}).scalar_one_or_none()
            validate_fast_plan(json.loads(encoded) if encoded is not None else None)
        elif trade.strategy in {"PaperSwingLevelBounce", "PaperSwingLevelBreakHold"}:
            from user_data.strategies.paper_swing_levels import validate_swing_plan
            encoded = session.execute(text("SELECT cd_value FROM trade_custom_data WHERE ft_trade_id=:trade_id AND cd_key=:key"),
                {"trade_id":trade.id,"key":"paper_fast_plan"}).scalar_one_or_none()
            validate_swing_plan(json.loads(encoded) if encoded is not None else None,
                                actual_leverage=float(trade.leverage))
        elif trade.strategy == "PaperNewsManual":
            from user_data.strategies.paper_news_manual import validate_manual_plan
            encoded = session.execute(text("SELECT cd_value FROM trade_custom_data WHERE ft_trade_id=:trade_id AND cd_key=:key"),
                {"trade_id":trade.id,"key":"paper_news_manual_plan"}).scalar_one_or_none()
            plan = json.loads(encoded) if encoded is not None else None
            validate_manual_plan(plan,float(plan["reference_rate"]),account or "paper_news_manual",
                actual_leverage=float(trade.leverage))
        elif trade.strategy in {spec["strategy"] for spec in AGGRESSIVE_ACCOUNTS.values()}:
            encoded = session.execute(text("SELECT cd_value FROM trade_custom_data WHERE ft_trade_id=:trade_id AND cd_key=:key"),
                {"trade_id":trade.id,"key":"paper_aggressive_plan"}).scalar_one_or_none()
            validate_aggressive_filled_plan(json.loads(encoded) if encoded is not None else None,
                actual_pair=trade.pair,actual_side="short" if trade.is_short else "long",
                actual_open_rate=float(trade.open_rate),actual_quantity=float(trade.amount),
                actual_stake=float(trade.stake_amount),actual_leverage=float(trade.leverage))
            plan = json.loads(encoded)
            if not isclose(float(plan["contract_size"]),float(trade.contract_size),rel_tol=1e-9,abs_tol=1e-9):
                return "missing_or_invalid"
    except (ValueError, TypeError, KeyError):
        return "missing_or_invalid"
    return "stored_valid"


def _worker_log_health(spec, tree, now, lifecycle="ACTIVE"):
    """Bounded tail facts, not a claim that a present process is trading correctly."""
    path = REPORT / f"{spec.key}.log"
    if lifecycle == "PARKED":
        return {"state":"unexpected_process_for_parked" if tree is not None else "expected_parked",
            "path":str(path)}
    if tree is None or not path.is_file():
        return {"state":"missing_process_or_log","path":str(path)}
    parent, worker = tree
    pid = (worker or parent)["pid"]
    with path.open("rb") as stream:
        stream.seek(max(0,path.stat().st_size-65536))
        lines = stream.read(65536).decode("utf-8",errors="replace").splitlines()
    heartbeat, errors, ambiguous = worker_heartbeat_state(spec,tree,now), [], False
    for line in lines:
        match = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) - ",line)
        if match is None:
            continue
        local = datetime.strptime(match[1],"%Y-%m-%d %H:%M:%S,%f").replace(tzinfo=ZoneInfo("Europe/London"))
        if local.replace(fold=0).utcoffset() != local.replace(fold=1).utcoffset():
            ambiguous = True
            continue
        clock = local.astimezone(timezone.utc)
        if " - ERROR - " in line or " - CRITICAL - " in line:
            if now-timedelta(hours=4) <= clock <= now+timedelta(minutes=2):
                errors.append(line[-180:])
    fresh = heartbeat["fresh"]
    expected = "PAUSED" if lifecycle == "DRAINING" else "RUNNING"
    healthy = fresh and heartbeat["state"] == expected and not errors
    return {"state":("draining_paused_observed" if expected == "PAUSED" else "running_observed") if healthy else "issue_or_unknown",
        "heartbeat_at_utc":heartbeat["at_utc"],
        "heartbeat_running":None if heartbeat["state"] is None else heartbeat["state"] == "RUNNING",
        "heartbeat_state":heartbeat["state"],
        "expected_state":expected,
        "recent_error_count_in_tail":len(errors),"recent_errors":errors[-2:],
        "ambiguous_local_clock_in_tail":ambiguous,"tail_limit_bytes":65536,"path":str(path)}


def _utc_datetime(value):
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _runtime_span(record, spec, now):
    info = account_record(record, spec)
    recorded_start = info.get("started_at_utc")
    start = _utc_datetime(recorded_start)
    start_source = "account_started_at_utc" if start else None
    events = record.get("process_recovery", {}).get("events", [])
    if "started_at_utc" not in info:
        first = [(_utc_datetime(event.get("at_utc")), event) for event in events
                 if isinstance(event, dict) and event.get("account") == spec.key
                 and event.get("status") == "running"]
        first = [(clock, event) for clock, event in first if clock is not None]
        if first:
            start = min(clock for clock, _ in first)
            start_source = "first_recorded_running_event_not_guaranteed_original_start"
    if start is None or start > now:
        return {"status": "unknown", "elapsed_seconds": None,
                "start_source": start_source or "no_valid_account_start",
                "end_source": None, "started_at_utc": None if start is None else start.isoformat(),
                "ended_at_utc": None}

    lifecycle = lifecycle_for(record, spec)
    end, end_source = now, "snapshot_time_active"
    if lifecycle == "DRAINING":
        boundary = _utc_datetime(info.get("lifecycle_changed_at_utc"))
        if boundary is None or boundary > now:
            return {"status": "unknown", "elapsed_seconds": None, "start_source": start_source,
                    "end_source": "invalid_or_future_draining_lifecycle_boundary",
                    "started_at_utc": start.isoformat(), "ended_at_utc": None}
        pauses = [(_utc_datetime(event.get("at_utc")), event) for event in events
                  if isinstance(event, dict) and event.get("account") == spec.key
                  and event.get("status") == "draining_paused"]
        pauses = [(clock, event) for clock, event in pauses if clock is not None
                  and boundary <= clock <= now]
        if not pauses:
            return {"status": "unknown", "elapsed_seconds": None, "start_source": start_source,
                    "end_source": "no_pause_confirmation_after_lifecycle_boundary",
                    "started_at_utc": start.isoformat(), "ended_at_utc": None}
        end = min(clock for clock, _ in pauses)
        end_source = "first_recorded_draining_paused_confirmation_observed_not_exact_onset"
    elif lifecycle == "PARKED":
        current = _utc_datetime(info.get("lifecycle_changed_at_utc"))
        if current is None or current > now:
            return {"status": "unknown", "elapsed_seconds": None, "start_source": start_source,
                    "end_source": "invalid_or_future_parked_lifecycle_boundary",
                    "started_at_utc": start.isoformat(), "ended_at_utc": None}
        parked = [(_utc_datetime(event.get("at_utc")), event) for event in events
                  if isinstance(event, dict) and event.get("account") == spec.key
                  and event.get("status") == "parked_flat"]
        parked = [(clock, event) for clock, event in parked if clock is not None
                  and current <= clock <= now]
        if parked:
            end = min(clock for clock, _ in parked)
            end_source = "recorded_parked_flat_stop"
        elif (info.get("last_recovery", {}).get("status") == "parked_flat"
              and _utc_datetime(info.get("last_recovery", {}).get("at_utc")) == current):
            end = current
            end_source = "current_verified_parked_flat_lifecycle_change"
        else:
            return {"status": "unknown", "elapsed_seconds": None, "start_source": start_source,
                    "end_source": "no_verified_parked_flat_stop", "started_at_utc": start.isoformat(),
                    "ended_at_utc": None}
    if end < start:
        return {"status": "unknown", "elapsed_seconds": None, "start_source": start_source,
                "end_source": "endpoint_precedes_start", "started_at_utc": start.isoformat(),
                "ended_at_utc": end.isoformat()}
    return {"status": "known", "elapsed_seconds": (end-start).total_seconds(),
            "start_source": start_source, "end_source": end_source,
            "started_at_utc": start.isoformat(), "ended_at_utc": end.isoformat(),
            "meaning": "elapsed trial span including downtime; not continuous uptime"}


def _profit_factor(positive, negative, closed_count, invalid=None):
    if invalid:
        return {"value": None, "status": invalid}
    if closed_count == 0:
        return {"value": None, "status": "no_closed_trades"}
    if positive is None or negative is None:
        return {"value": None, "status": "missing_closed_profit"}
    if not isfinite(positive) or not isfinite(negative):
        return {"value": None, "status": "nonfinite_closed_profit"}
    if negative == 0:
        return {"value": None, "status": "no_loss_trades"}
    try:
        factor = positive / abs(negative)
    except (OverflowError, ZeroDivisionError):
        factor = float("inf")
    if not isfinite(factor):
        return {"value": None, "status": "nonfinite_profit_factor"}
    return {"value": factor, "status": "ok"}


def _finite_sum(values):
    total = sum(values)
    return total if isfinite(total) else None


def _positive_price(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if isfinite(number) and number > 0 else None


def _strict_pnl_sum(trades, field):
    values = []
    for trade in trades:
        value = getattr(trade, field, None)
        if value is None:
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if not isfinite(number):
            return None
        values.append(number)
    return _finite_sum(values)


def _public_json(url, timeout):
    try:
        with urlopen(url, timeout=timeout) as response:
            return json.load(response)
    except (OSError, ValueError):
        return None


def _ticker_prices(payload):
    if not isinstance(payload, list):
        return {}
    prices = {}
    for row in payload:
        if not isinstance(row, dict) or not isinstance(row.get("symbol"), str):
            continue
        price = _positive_price(row.get("price"))
        if price is not None:
            prices[row["symbol"]] = price
    return prices


def _settled_partial_exit_funding(trade):
    """Funding already included in realized partial exits, using the Trade order stream."""
    if getattr(trade, "trading_mode", None) != TradingMode.FUTURES:
        return 0.0, "ok"
    filled_orders = [order for order in trade.orders if not order.ft_is_open and order.filled]
    last_exit = next((index for index in range(len(filled_orders) - 1, -1, -1)
                      if filled_orders[index].ft_order_side != trade.entry_side), None)
    if last_exit is None:
        return 0.0, "ok"

    settled_funding = 0.0
    current_funding = 0.0
    for order in filled_orders[:last_exit + 1]:
        funding = order.funding_fee
        if funding is None or isinstance(funding, bool):
            return None, "settled_funding_unknown"
        try:
            funding = float(funding)
        except (TypeError, ValueError, OverflowError):
            return None, "settled_funding_unknown"
        if not isfinite(funding):
            return None, "settled_funding_unknown"
        current_funding += funding
        if not isfinite(current_funding):
            return None, "settled_funding_unknown"
        if order.ft_order_side != trade.entry_side:
            settled_funding += current_funding
            if not isfinite(settled_funding):
                return None, "settled_funding_unknown"
            current_funding = 0.0
    return settled_funding, "ok"


def _open_pnl(opened, quotes, prices_requested):
    if not opened:
        return 0.0, "no_open_positions"
    if not prices_requested:
        return None, "prices_not_requested"
    values = []
    for trade in opened:
        symbol = trade.pair.split(":")[0].replace("/", "")
        price = _positive_price(quotes.get(symbol))
        if price is None:
            return None, "missing_or_invalid_quote"
        settled_funding, funding_status = _settled_partial_exit_funding(trade)
        if funding_status != "ok":
            return None, funding_status
        try:
            value = float(trade.calculate_profit(price).profit_abs) - settled_funding
        except (KeyError, TypeError, ValueError, OverflowError):
            return None, "invalid_trade_pnl"
        if not isfinite(value):
            return None, "invalid_trade_pnl"
        values.append(value)
    total = _finite_sum(values)
    return (total, "ok") if total is not None else (None, "invalid_trade_pnl")


def account_snapshot(prices=False, now=None, price_context=None):
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("Account snapshot requires a timezone-aware clock")
    record=read_json(RECORD); processes=process_inventory(); quotes={}; price_status="not_requested"
    if prices:
        payload=_public_json("https://fapi.binance.com/fapi/v1/ticker/price",timeout=8)
        quotes=_ticker_prices(payload)
        price_status="ok" if quotes else ("fetch_failed" if payload is None else "invalid_ticker_payload")
    if price_context is not None:
        price_context.update({"quotes":quotes,"status":price_status})
    output=[]
    for spec in account_specs_for_record(record):
        row=account_record(record,spec); path=(REPORT/row["db"]).resolve()
        tree=matching_processes(spec,processes)
        lifecycle=lifecycle_for(record,spec)
        if not path.is_file() and spec.new_identity and not row.get("database_initialized_at_utc"):
            output.append({"account":spec.key,"lifecycle":lifecycle,"database_state":"not_initialized",
                "running_exact_tree":tree is not None,"worker_log":_worker_log_health(spec,tree,now,lifecycle),
                "open_longs":None,"open_shorts":None,"closed_longs":None,"closed_shorts":None,
                "wins":None,"losses":None,"banked_pnl_usdt":None,"estimated_open_pnl_usdt":None,
                "banked_pnl_status":"database_not_initialized","estimated_open_pnl_status":"database_not_initialized",
                "open_order_count":None,"protection_issue_count":None,"open_positions":None,
                "database_trade_count":None,"trades_per_day":None,"trades_per_day_status":"database_not_initialized",
                "profit_factor":{"value":None,"status":"database_not_initialized"},
                "runtime_span":_runtime_span(record,spec,now)})
            continue
        if spec.new_identity and path.is_file() and not row.get("database_initialized_at_utc"):
            raise ValueError(f"{spec.key}: DB exists without a registered successful initialization")
        engine=create_engine("sqlite:///file:"+path.as_posix()+"?mode=ro&uri=true")
        with Session(engine) as session:
            trades=list(session.scalars(select(Trade))); opened=[t for t in trades if t.is_open]; closed=[t for t in trades if not t.is_open]
            closed_banked=_strict_pnl_sum(closed,"close_profit_abs")
            open_banked=_strict_pnl_sum(opened,"realized_profit")
            banked=(closed_banked+open_banked
                    if closed_banked is not None and open_banked is not None
                    and isfinite(closed_banked+open_banked) else None)
            closed_profit_values=[]; invalid_closed_profit=None
            for trade in closed:
                value=trade.close_profit_abs
                if value is None:
                    invalid_closed_profit="missing_closed_profit"; continue
                try: value=float(value)
                except (TypeError,ValueError):
                    invalid_closed_profit="missing_closed_profit"; continue
                if not isfinite(value):
                    invalid_closed_profit="nonfinite_closed_profit"; continue
                closed_profit_values.append(value)
            positive=_finite_sum(value for value in closed_profit_values if value>0)
            negative=_finite_sum(value for value in closed_profit_values if value<0)
            if positive is None or negative is None:
                invalid_closed_profit = "nonfinite_closed_profit"
                positive = negative = None
            pf=_profit_factor(positive,negative,len(closed),invalid_closed_profit)
            span=_runtime_span(record,spec,now)
            trades_per_day=(len(trades)/(span["elapsed_seconds"]/86400)
                            if span["status"]=="known" and span["elapsed_seconds"] and span["elapsed_seconds"]>0 else None)
            unrealized,open_pnl_status=_open_pnl(opened,quotes,prices)
            positions=[{"pair":t.pair,"short":t.is_short,"leverage":t.leverage,"stop_price":t.stop_loss,
                "margin":t.stake_amount,"protection_state":_protection_state(t,session,spec.bot_name)} for t in opened]
            open_order_count=session.scalar(select(func.count(Order.id)).where(Order.ft_is_open.is_(True)))
            output.append({"account":spec.key,"lifecycle":lifecycle,"database_state":"initialized","running_exact_tree":tree is not None,
                "worker_log":_worker_log_health(spec,tree,now,lifecycle),
                "open_order_count":open_order_count,
                "open_longs":sum(not t.is_short for t in opened),"open_shorts":sum(t.is_short for t in opened),
                "closed_longs":sum(not t.is_short for t in closed),"closed_shorts":sum(t.is_short for t in closed),
                "wins":sum(float(t.close_profit_abs or 0)>0 for t in closed),"losses":sum(float(t.close_profit_abs or 0)<0 for t in closed),
                "banked_pnl_usdt":banked,"banked_pnl_status":"ok" if banked is not None else "invalid_or_missing_pnl",
                "estimated_open_pnl_usdt":unrealized,"estimated_open_pnl_status":open_pnl_status,
                "database_trade_count":len(trades),"trades_per_day":trades_per_day,
                "trades_per_day_status":"ok" if trades_per_day is not None else "unknown_runtime_or_zero_duration",
                "closed_profit_positive_usdt":positive if not invalid_closed_profit else None,
                "closed_profit_negative_usdt":negative if not invalid_closed_profit else None,
                "profit_factor":pf,"runtime_span":span,
                "protection_issue_count":sum(p["protection_state"]!="stored_valid" for p in positions),
                "parked_activity_issue":lifecycle=="PARKED" and (bool(opened) or bool(open_order_count)),
                "parked_process_issue":lifecycle=="PARKED" and tree is not None,
                "open_positions":positions})
        engine.dispose()
    return output


def _table_number(value, digits=2):
    return "—" if value is None or not isinstance(value, (int, float)) or not isfinite(value) else f"{value:.{digits}f}"


def _runtime_label(span):
    if span.get("status") != "known":
        return "unknown"
    seconds = int(span["elapsed_seconds"])
    days, remainder = divmod(seconds, 86400)
    hours = remainder // 3600
    return f"{days}d {hours}h"


def _pooled_factor(accounts):
    if any(row.get("profit_factor", {}).get("status") == "database_not_initialized" for row in accounts):
        return {"value": None, "status": "database_not_initialized"}
    if any(row.get("profit_factor", {}).get("status") in {"missing_closed_profit", "nonfinite_closed_profit"} for row in accounts):
        return {"value": None, "status": "incomplete_closed_profit"}
    if any(row.get("closed_longs") is None or row.get("closed_shorts") is None for row in accounts):
        return {"value": None, "status": "incomplete_account_coverage"}
    closed = sum(row.get("closed_longs", 0) + row.get("closed_shorts", 0) for row in accounts)
    if not closed:
        return {"value": None, "status": "no_closed_trades"}
    positive_values = [row.get("closed_profit_positive_usdt") for row in accounts]
    negative_values = [row.get("closed_profit_negative_usdt") for row in accounts]
    if any(value is None or not isinstance(value, (int, float)) or not isfinite(value)
           for value in positive_values + negative_values):
        return {"value": None, "status": "incomplete_closed_profit"}
    positive = _finite_sum(positive_values)
    negative = _finite_sum(negative_values)
    if positive is None or negative is None:
        return {"value": None, "status": "nonfinite_closed_profit"}
    return _profit_factor(positive, negative, closed)


def _configured_starting_capital(spec):
    try:
        config = load_from_files([str(ROOT / BASE), str(ROOT / spec.config)])
    except (OSError, ValueError, TypeError, KeyError):
        return None, "invalid_registered_config"
    value = _positive_price(config.get("dry_run_wallet"))
    return (value, "ok") if value is not None else (None, "missing_or_invalid_dry_run_wallet")


def _account_attempt(record, spec, now, starting_capital):
    info = account_record(record, spec)
    runtime = _runtime_span(record, spec, now)
    start = _utc_datetime(info.get("started_at_utc"))
    source = "account_started_at_utc" if start else None
    if start is None:
        start = _utc_datetime(info.get("database_initialized_at_utc"))
        if start is not None:
            source = "database_initialized_at_utc_approximation"
    if start is None:
        start = _utc_datetime(runtime.get("started_at_utc"))
        source = runtime.get("start_source") if start is not None else None
    boundary_known = source in {"account_started_at_utc", "database_initialized_at_utc_approximation"}
    identity = {
        "account": spec.key,
        "bot_name": spec.bot_name,
        "database": info.get("db"),
        "started_at_utc": None if start is None else start.isoformat(),
        "recorded_started_at_utc": info.get("started_at_utc"),
        "database_initialized_at_utc": info.get("database_initialized_at_utc"),
        "starting_capital_usdt": starting_capital,
    }
    attempt_key = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()[:24]
    return {"key": attempt_key, "identity": identity,
            "start_at": start, "start_source": source,
            "start_status": ("known" if source == "account_started_at_utc"
                             else "approximate_database_initialization" if boundary_known
                             else "unverified_start" if start else "missing_start"),
            "benchmark_start_known": boundary_known}


def _load_daily_history():
    if not PAPER_DAILY_HISTORY.exists():
        return {"schema_version": 1, "benchmark_source": BTC_PERPETUAL_SOURCE,
                "observation_policy": PAPER_DAILY_OBSERVATION_POLICY, "attempts": {}}
    history = read_json(PAPER_DAILY_HISTORY)
    if (not isinstance(history, dict) or history.get("schema_version") != 1
            or not isinstance(history.get("attempts"), dict)):
        raise ValueError(f"Refusing to use malformed daily paper history: {PAPER_DAILY_HISTORY}")
    for entry in history["attempts"].values():
        if not isinstance(entry, dict) or not isinstance(entry.get("daily", {}), dict):
            raise ValueError(f"Refusing to replace malformed daily paper history: {PAPER_DAILY_HISTORY}")
        baseline = entry.get("btc_baseline")
        if baseline is not None and not isinstance(baseline, dict):
            raise ValueError(f"Refusing to replace malformed daily paper history: {PAPER_DAILY_HISTORY}")
        if (isinstance(baseline, dict) and baseline.get("status") == "ok"
                and (_positive_price(baseline.get("price_usdt")) is None
                     or _utc_datetime(baseline.get("candle_open_at_utc")) is None)):
            raise ValueError(f"Refusing to replace malformed daily paper history: {PAPER_DAILY_HISTORY}")
    return history


def _fetch_btc_baseline(start, now):
    if start is None:
        return {"status": "missing_attempt_start"}
    if start > now:
        return {"status": "future_attempt_start"}
    start_ms = int(start.timestamp() * 1000)
    url = "https://fapi.binance.com/fapi/v1/klines?" + urlencode({
        "symbol": "BTCUSDT", "interval": "1m", "startTime": start_ms, "limit": 2,
    })
    payload = _public_json(url, timeout=3)
    if not isinstance(payload, list):
        return {"status": "fetch_failed" if payload is None else "invalid_candle_payload"}
    candidates = []
    for candle in payload:
        if not isinstance(candle, list) or len(candle) < 2:
            continue
        try:
            opened_ms = int(candle[0])
        except (TypeError, ValueError, OverflowError):
            continue
        price = _positive_price(candle[1])
        if start_ms <= opened_ms <= start_ms + 60_000 and price is not None:
            candidates.append((opened_ms, price))
    if not candidates:
        return {"status": "no_valid_candle_within_60s"}
    opened_ms, price = min(candidates)
    return {"status": "ok", "price_usdt": price,
            "candle_open_at_utc": datetime.fromtimestamp(opened_ms / 1000, timezone.utc).isoformat(),
            "start_at_utc": start.isoformat(), "interval": "1m", "price_field": "open"}


def _return_fields(row, starting_capital, capital_status, attempt, baseline, current_btc):
    banked = None
    if row.get("banked_pnl_usdt") is not None:
        try:
            banked = float(row["banked_pnl_usdt"])
        except (TypeError, ValueError, OverflowError):
            banked = None
        if banked is not None and not isfinite(banked):
            banked = None
    open_pnl = row.get("estimated_open_pnl_usdt")
    try:
        open_pnl = float(open_pnl) if open_pnl is not None else None
    except (TypeError, ValueError, OverflowError):
        open_pnl = None
    if open_pnl is not None and not isfinite(open_pnl):
        open_pnl = None
    equity = (starting_capital + banked + open_pnl
              if starting_capital is not None and banked is not None and open_pnl is not None else None)
    if equity is not None and not isfinite(equity):
        equity = None
    if capital_status != "ok":
        bot_status = capital_status
    elif banked is None or open_pnl is None or equity is None:
        bot_status = "pnl_unknown"
    else:
        bot_status = "ok"
    bot_return = ((equity / starting_capital) - 1) * 100 if bot_status == "ok" else None
    if bot_return is not None and not isfinite(bot_return):
        bot_return, bot_status = None, "invalid_return"

    baseline_price = _positive_price(baseline.get("price_usdt")) if baseline.get("status") == "ok" else None
    current_price = _positive_price(current_btc)
    if not attempt["benchmark_start_known"]:
        btc_status = "attempt_start_unknown"
    elif baseline_price is None:
        btc_status = baseline.get("status", "baseline_unknown")
    elif current_price is None:
        btc_status = "current_quote_unknown"
    else:
        btc_status = "ok"
        if attempt["start_status"] != "known":
            btc_status = "ok_approximate_start"
    btc_return = ((current_price / baseline_price) - 1) * 100 \
        if btc_status in {"ok", "ok_approximate_start"} else None
    if btc_return is not None and not isfinite(btc_return):
        btc_return, btc_status = None, "invalid_return"
    delta = bot_return - btc_return if bot_return is not None and btc_return is not None else None
    if baseline.get("status") == "ok":
        baseline = {**baseline, "start_source": attempt["start_source"],
                    "start_status": attempt["start_status"]}
    row.update({
        "starting_capital_usdt": starting_capital,
        "starting_capital_status": capital_status,
        "equity_usdt": equity,
        "equity_status": "ok" if equity is not None else "pnl_or_capital_unknown",
        "attempt_key": attempt["key"],
        "attempt_start_at_utc": None if attempt["start_at"] is None else attempt["start_at"].isoformat(),
        "attempt_start_source": attempt["start_source"],
        "attempt_start_status": attempt["start_status"],
        "bot_return_pct": bot_return,
        "bot_return_status": bot_status,
        "btc_baseline": baseline,
        "btc_return_pct": btc_return,
        "btc_return_status": btc_status,
        "bot_btc_delta_pp": delta,
    })


def enrich_account_returns(accounts, now, price_context, *, record_daily=False):
    now = now.astimezone(timezone.utc)
    record = read_json(RECORD)
    history = _load_daily_history() if record_daily or price_context is not None else None
    current_btc = (price_context or {}).get("quotes", {}).get("BTCUSDT")
    specs = {spec.key: spec for spec in account_specs_for_record(record)}
    history_changed = False
    if record_daily and "observation_policy" not in history:
        history["observation_policy"] = PAPER_DAILY_OBSERVATION_POLICY
        history_changed = True
    for row in accounts:
        if row.get("lifecycle") != "ACTIVE":
            continue
        spec = specs.get(row.get("account"))
        if spec is None:
            continue
        starting_capital, capital_status = _configured_starting_capital(spec)
        attempt = _account_attempt(record, spec, now, starting_capital)
        cached_attempt = (history or {}).get("attempts", {}).get(attempt["key"], {})
        baseline = cached_attempt.get("btc_baseline")
        if not (isinstance(baseline, dict) and baseline.get("status") == "ok"):
            if attempt["benchmark_start_known"]:
                baseline = _fetch_btc_baseline(attempt["start_at"], now)
            else:
                baseline = {"status": attempt["start_status"]}
        if baseline.get("status") == "ok":
            baseline = {**baseline, "start_source": attempt["start_source"],
                        "start_status": attempt["start_status"]}
        _return_fields(row, starting_capital, capital_status, attempt, baseline, current_btc)

        if record_daily:
            entry = history["attempts"].setdefault(attempt["key"], {
                "account": spec.key, "bot_name": spec.bot_name,
                "identity": attempt["identity"], "daily": {},
            })
            old_baseline = entry.get("btc_baseline")
            if not (isinstance(old_baseline, dict) and old_baseline.get("status") == "ok"):
                entry["btc_baseline"] = baseline
                history_changed = True
            daily = entry.setdefault("daily", {})
            date_key = now.date().isoformat()
            if date_key not in daily:
                closed_longs, closed_shorts = row.get("closed_longs"), row.get("closed_shorts")
                closed_count = (closed_longs + closed_shorts
                                if isinstance(closed_longs, int) and isinstance(closed_shorts, int) else None)
                daily[date_key] = {
                    "observed_at_utc": now.isoformat(),
                    "equity_usdt": row.get("equity_usdt"),
                    "btc_quote_usdt": _positive_price(current_btc),
                    "btc_quote_status": "ok" if _positive_price(current_btc) is not None else "current_quote_unknown",
                    "cumulative_entries": row.get("database_trade_count"),
                    "cumulative_closed": closed_count,
                    "open_longs": row.get("open_longs"),
                    "open_shorts": row.get("open_shorts"),
                    "equity_status": row.get("equity_status"),
                }
                row["daily_record_status"] = "recorded_first_observation_for_utc_date"
                history_changed = True
            else:
                row["daily_record_status"] = "already_recorded_for_utc_date"

    if record_daily and history_changed:
        from user_data.Custom_Launcher.research.context_features.global_context_source_preflight import write_text_atomic
        write_text_atomic(PAPER_DAILY_HISTORY,
                          json.dumps(history, separators=(",", ":"), allow_nan=False) + "\n")
    return accounts


ACCOUNT_DISPLAY_NAMES = {
    "news_manual": "News manual slow",
    "news_fast": "News manual fast",
    "news_lab": "News manual slow adventurous",
    "swing_level_bounce": "Level bounce — auto",
    "swing_level_break_hold": "Level breakout — auto",
    "aggressive_reclaim": "Sweep recovery — news-assisted auto",
    "aggressive_vacuum": "Thin-volume breakout — news-assisted auto",
    "aggressive_auction": "Value-zone reaction — news-assisted auto",
    "aggressive_rotation": "Relative strength — news-assisted auto",
}


def format_account_table(accounts, observed_at_utc):
    """Pure formatter: no disk, network, process, or database reads."""
    count_fields = ("open_longs", "open_shorts", "closed_longs", "closed_shorts", "wins", "losses")
    pnl_fields = ("banked_pnl_usdt", "estimated_open_pnl_usdt")
    header = ("| Account | Runtime† | Trades/day | PF (closed) | Open L/S | Closed L/S | W/L | "
              "Banked USDT | Open P/L USDT | Lifecycle | Bot % | BTC % (perp proxy) | Delta pp |")
    separator = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|"

    def render_row(row):
        pf = row.get("profit_factor", {})
        factor = _table_number(pf.get("value"), 3) if pf.get("status") == "ok" else "—"
        if pf.get("status") == "no_loss_trades": factor = "no losses"
        span = row.get("runtime_span", {})
        runtime = _runtime_label(span)
        if span.get("status") == "known" and span.get("start_source") == "first_recorded_running_event_not_guaranteed_original_start":
            runtime += "†"
        account_key = row.get("account", "?")
        account = ACCOUNT_DISPLAY_NAMES.get(account_key, account_key)
        if row.get("database_state") == "not_initialized":
            account += " (DB not initialized)"
        return "| {account} | {runtime} | {rate} | {factor} | {ol}/{os} | {cl}/{cs} | {wins}/{losses} | {banked} | {open_pnl} | {lifecycle} | {bot_return} | {btc_return} | {delta} |".format(
            account=account, runtime=runtime,
            rate=_table_number(row.get("trades_per_day")), factor=factor,
            ol=row.get("open_longs") if row.get("open_longs") is not None else "—",
            os=row.get("open_shorts") if row.get("open_shorts") is not None else "—",
            cl=row.get("closed_longs") if row.get("closed_longs") is not None else "—",
            cs=row.get("closed_shorts") if row.get("closed_shorts") is not None else "—",
            wins=row.get("wins") if row.get("wins") is not None else "—",
            losses=row.get("losses") if row.get("losses") is not None else "—",
            banked=_table_number(row.get("banked_pnl_usdt")),
            open_pnl=_table_number(row.get("estimated_open_pnl_usdt")), lifecycle=row.get("lifecycle", "unknown"),
            bot_return=_table_number(row.get("bot_return_pct")),
            btc_return=_table_number(row.get("btc_return_pct")),
            delta=_table_number(row.get("bot_btc_delta_pp")))

    def runtime_sort_key(row):
        span = row.get("runtime_span", {})
        seconds = span.get("elapsed_seconds") if span.get("status") == "known" else None
        known = isinstance(seconds, (int, float)) and isfinite(seconds)
        return (not known, -seconds if known else 0, str(row.get("account", "?")).casefold())

    active = [row for row in accounts if row.get("lifecycle") == "ACTIVE"]
    lines = [f"ACTIVE paper account snapshot — {observed_at_utc}", ""]
    if not active:
        lines.extend(["No ACTIVE paper accounts; PF undefined.", ""])
    lines.extend([header, separator])
    for row in sorted(active, key=runtime_sort_key):
        lines.append(render_row(row))

    total = {}
    for field in count_fields + pnl_fields:
        values = [row.get(field) for row in active]
        if not active:
            total[field] = 0.0 if field in pnl_fields else 0
        else:
            total[field] = sum(values) if all(value is not None for value in values) else None
    total.update({"account": "TOTAL", "lifecycle": "—", "runtime_span": {"status": "unknown"},
                  "trades_per_day": None, "profit_factor": _pooled_factor(active)})
    capital_values = [row.get("starting_capital_usdt") for row in active]
    equity_values = [row.get("equity_usdt") for row in active]
    total_bot_return = None
    if (active and all(isinstance(value, (int, float)) and isfinite(value) and value > 0
                       for value in capital_values)
            and all(isinstance(value, (int, float)) and isfinite(value) for value in equity_values)):
        total_capital = _finite_sum(capital_values)
        total_equity = _finite_sum(equity_values)
        if total_capital is not None and total_capital > 0 and total_equity is not None:
            total_bot_return = ((total_equity / total_capital) - 1) * 100
            if not isfinite(total_bot_return):
                total_bot_return = None
    pf = total["profit_factor"]
    factor = _table_number(pf.get("value"), 3) if pf.get("status") == "ok" else "—"
    if pf.get("status") == "no_loss_trades": factor = "no losses"
    lines.append("| TOTAL | — | — | {factor} | {ol}/{os} | {cl}/{cs} | {wins}/{losses} | {banked} | {open_pnl} | — | {bot_return} | — | — |".format(
        factor=factor, ol=total["open_longs"] if total["open_longs"] is not None else "—",
        os=total["open_shorts"] if total["open_shorts"] is not None else "—",
        cl=total["closed_longs"] if total["closed_longs"] is not None else "—",
        cs=total["closed_shorts"] if total["closed_shorts"] is not None else "—",
        wins=total["wins"] if total["wins"] is not None else "—",
        losses=total["losses"] if total["losses"] is not None else "—",
        banked=_table_number(total["banked_pnl_usdt"]), open_pnl=_table_number(total["estimated_open_pnl_usdt"]),
        bot_return=_table_number(total_bot_return)))
    return "\n".join(lines)


def _finite_number(value, label):
    if value is None:
        raise ValueError(f"Missing numeric input: {label}")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid numeric input: {label}") from error
    if not isfinite(number):
        raise ValueError(f"Non-finite numeric input: {label}")
    return number


def _filled_order_audit(trade):
    """Reconstruct realized gross movement and rate-based fees from recorded fills."""
    fills = []
    for order in trade.orders:
        if order.filled is None:
            if not order.ft_is_open:
                return {"complete":False,"filled_order_count":len(fills),
                    "gross_price_pnl_usdt":None,"estimated_fees_usdt":None,
                    "issues":[f"terminal order {order.id} has no recorded filled quantity"]}
            continue
        quantity = _finite_number(order.filled, f"trade {trade.id} order {order.id} filled")
        if quantity < 0:
            raise ValueError(f"trade {trade.id} order {order.id} has negative filled quantity")
        if quantity == 0:
            continue
        filled_at = order.order_filled_utc
        if filled_at is None:
            return {"complete":False,"filled_order_count":len(fills),
                "gross_price_pnl_usdt":None,"estimated_fees_usdt":None,
                "issues":[f"filled order {order.id} has no recorded fill time"]}
        if order.average is not None:
            price = _finite_number(order.average, f"trade {trade.id} order {order.id} average")
            if price <= 0:
                return {"complete":False,"filled_order_count":len(fills),
                    "gross_price_pnl_usdt":None,"estimated_fees_usdt":None,
                    "issues":[f"order {order.id} has non-positive average fill price"]}
        elif order.cost is not None:
            cost = _finite_number(order.cost, f"trade {trade.id} order {order.id} cost")
            if cost <= 0:
                return {"complete":False,"filled_order_count":len(fills),
                    "gross_price_pnl_usdt":None,"estimated_fees_usdt":None,
                    "issues":[f"order {order.id} has non-positive filled cost"]}
            price = cost / quantity
        else:
            return {"complete":False,"filled_order_count":len(fills),
                "gross_price_pnl_usdt":None,"estimated_fees_usdt":None,
                "issues":[f"filled order {order.id} has neither average fill price nor filled cost"]}
        if order.cost is not None:
            notional = _finite_number(order.cost, f"trade {trade.id} order {order.id} cost")
            if notional <= 0:
                return {"complete":False,"filled_order_count":len(fills),
                    "gross_price_pnl_usdt":None,"estimated_fees_usdt":None,
                    "issues":[f"order {order.id} has non-positive filled notional"]}
        else:
            notional = quantity * price
        fills.append({"order":order,"quantity":quantity,"price":price,"notional":notional,
            "filled_at":filled_at})
    if not fills:
        return {"complete":False,"filled_order_count":0,"gross_price_pnl_usdt":None,
            "estimated_fees_usdt":None,"issues":["no recorded filled orders"]}
    fills.sort(key=lambda item:item["filled_at"])
    entry_side = trade.entry_side
    entry_fill_count = sum(fill["order"].ft_order_side == entry_side for fill in fills)
    exit_fill_count = sum(fill["order"].ft_order_side != entry_side and
        fill["order"].ft_order_side in {"buy","sell","stoploss"} for fill in fills)
    if entry_fill_count == 0 or exit_fill_count == 0:
        missing = "entry" if entry_fill_count == 0 else "exit"
        return {"complete":False,"filled_order_count":len(fills),
            "entry_fill_count":entry_fill_count,"exit_fill_count":exit_fill_count,
            "gross_price_pnl_usdt":None,"estimated_fees_usdt":None,
            "issues":[f"no recorded filled {missing} orders"]}
    signed_direction = -1. if trade.is_short else 1.
    open_quantity = open_cost = gross = estimated_fees = entry_notional = exit_notional = 0.
    for fill in fills:
        order = fill["order"]
        if order.ft_order_side == entry_side:
            role = "entry"
            fee_rate = _finite_number(trade.fee_open, f"trade {trade.id} fee_open")
        elif order.ft_order_side in {"buy","sell","stoploss"}:
            role = "exit"
            fee_rate = _finite_number(trade.fee_close, f"trade {trade.id} fee_close")
        else:
            raise ValueError(f"trade {trade.id} order {order.id} has unrecognized filled side {order.ft_order_side!r}")
        estimated_fees += fill["notional"] * fee_rate
        if role == "entry":
            entry_notional += fill["notional"]
            open_quantity += fill["quantity"]
            open_cost += fill["quantity"] * fill["price"]
            continue
        exit_notional += fill["notional"]
        tolerance = max(1e-12, open_quantity * 1e-8)
        if open_quantity <= 0 or fill["quantity"] > open_quantity + tolerance:
            raise ValueError(f"trade {trade.id} order {order.id} exit exceeds filled entry quantity")
        average_entry = open_cost / open_quantity
        gross += signed_direction * (fill["price"] - average_entry) * fill["quantity"]
        remaining = open_quantity - fill["quantity"]
        if abs(remaining) <= tolerance:
            open_quantity = open_cost = 0.
        else:
            open_cost = average_entry * remaining
            open_quantity = remaining
    if not trade.is_open and open_quantity > max(1e-12, open_quantity * 1e-8):
        raise ValueError(f"closed trade {trade.id} retains unclosed filled quantity {open_quantity}")
    return {"complete":True,"filled_order_count":len(fills),"entry_fill_count":entry_fill_count,
        "exit_fill_count":exit_fill_count,"gross_price_pnl_usdt":gross,"estimated_fees_usdt":estimated_fees,
        "entry_notional_usdt":entry_notional,"exit_notional_usdt":exit_notional}


def _closed_financials(trades):
    closed = [trade for trade in trades if not trade.is_open]
    audits = [_filled_order_audit(trade) for trade in closed]
    closed_nets, realized, funding = [], [], []
    close_vs_realized_mismatches = 0
    for trade in closed:
        close_net = _finite_number(trade.close_profit_abs, f"closed trade {trade.id} close_profit_abs")
        realized_net = _finite_number(trade.realized_profit, f"closed trade {trade.id} realized_profit")
        closed_nets.append(close_net)
        realized.append(realized_net)
        if not isclose(close_net,realized_net,rel_tol=1e-7,abs_tol=1e-7):
            close_vs_realized_mismatches += 1
        funding.append(None if trade.funding_fees is None else _finite_number(
            trade.funding_fees,f"closed trade {trade.id} funding_fees"))
    funding_complete = all(value is not None for value in funding)
    fill_complete = all(audit["complete"] for audit in audits)
    errors = [{"trade_id":trade.id,"issues":audit["issues"]}
        for trade,audit in zip(closed,audits) if not audit["complete"]]
    gross = sum(audit["gross_price_pnl_usdt"] for audit in audits) if fill_complete else None
    fees = sum(audit["estimated_fees_usdt"] for audit in audits) if fill_complete else None
    funding_known = sum(value for value in funding if value is not None)
    funding_total = funding_known if funding_complete else None
    closed_net = sum(closed_nets)
    realized_total = sum(realized)
    reconstructed = gross-fees+funding_total if fill_complete and funding_complete else None
    open_realized = sum(_finite_number(trade.realized_profit,f"open trade {trade.id} realized_profit")
        for trade in trades if trade.is_open)
    groups = defaultdict(list)
    for trade,audit in zip(closed,audits):
        groups[(trade.enter_tag,trade.exit_reason)].append((trade,audit))
    by_entry_exit = []
    for (tag,reason),group in sorted(groups.items(),key=lambda item:(str(item[0][0]),str(item[0][1]))):
        group_net = sum(_finite_number(trade.close_profit_abs,f"closed trade {trade.id} close_profit_abs")
            for trade,_ in group)
        group_complete = all(audit["complete"] for _,audit in group)
        group_gross = sum(audit["gross_price_pnl_usdt"] for _,audit in group) if group_complete else None
        group_fees = sum(audit["estimated_fees_usdt"] for _,audit in group) if group_complete else None
        group_funding = [trade.funding_fees for trade,_ in group]
        group_funding_total = (sum(_finite_number(value,"group funding_fees") for value in group_funding)
            if all(value is not None for value in group_funding) else None)
        by_entry_exit.append({"entry_tag":tag,"exit_reason":reason,"closed_trade_count":len(group),
            "closed_net_usdt":group_net,"gross_filled_order_price_pnl_usdt":group_gross,
            "estimated_fees_usdt":group_fees,"recorded_funding_usdt":group_funding_total,
            "reconciliation_residual_usdt":None if not group_complete or group_funding_total is None else
                group_net-(group_gross-group_fees+group_funding_total)})
    return {"closed_trade_count":len(closed),"closed_net_usdt":closed_net,
        "open_partial_realized_profit_usdt":open_realized,
        "closed_realized_profit_usdt":realized_total,
        "close_profit_abs_vs_realized_profit_mismatch_count":close_vs_realized_mismatches,
        "gross_filled_order_price_pnl_usdt":gross,"estimated_fees_usdt":fees,
        "recorded_funding_usdt":funding_total,"recorded_funding_known_sum_usdt":funding_known,
        "funding_missing_trade_count":sum(value is None for value in funding),
        "reconstructed_net_usdt":reconstructed,
        "reconciliation_residual_usdt":None if reconstructed is None else closed_net-reconstructed,
        "filled_order_count":sum(audit["filled_order_count"] for audit in audits),
        "fill_metrics_complete":fill_complete,"fill_metric_issues":errors,
        "by_entry_tag_exit_reason":by_entry_exit}


def _journal_summary(account):
    path = _journal(account)
    rows = decision_rows(path)
    latest = {}
    for row in rows:
        if not isinstance(row,dict) or not isinstance(row.get("decision_id"),str) or not row["decision_id"].strip():
            raise ValueError(f"Malformed decision row in {path}")
        stamp = _utc(row["at_utc"])
        previous = latest.get(row["decision_id"])
        if previous is None or stamp >= previous[0]:
            latest[row["decision_id"]] = (stamp,row)
    decisions = sorted(latest.values(),key=lambda item:item[0])
    actions = Counter(str(row.get("action")) for _,row in decisions)
    statuses = Counter(str(row.get("status")) for _,row in decisions)
    holds = [{"decision_id":row["decision_id"],"at_utc":stamp.isoformat(),
        "reason":None if not isinstance(row.get("reason"),str) else row["reason"][:240],
        "status":row.get("status")} for stamp,row in decisions
        if row.get("action") in {"hold","no-action","no_action"}][-3:]
    return {"journal":str(path),"journal_state":"observed" if path.is_file() else "missing",
        "journal_rows":len(rows),"unique_decision_ids":len(decisions),
        "superseded_transition_rows":len(rows)-len(decisions),"latest_action_counts":dict(actions),
        "latest_status_counts":dict(statuses),"recent_hold_decisions":holds}


def _control_history_snapshot(record, now):
    approval_start = None
    approved_at = record.get("fast_reaction_pair",{}).get("approved_at_utc")
    if approved_at:
        approval_start = _utc(approved_at)
    raw_rows = decision_rows(CONTROL_JOURNAL)
    parsed = []
    seen = set()
    for row in raw_rows:
        observed = _utc(row["observed_at_utc"])
        expires = _utc(row["valid_until_utc"])
        validation_clock = observed + min(timedelta(seconds=1),(expires-observed)/2)
        parse_fast_control(row,validation_clock)
        decision_id = row["decision_id"]
        if decision_id in seen:
            raise ValueError(f"Duplicate fast-context decision ID: {decision_id}")
        seen.add(decision_id)
        parsed.append({"row":row,"start":observed,"expiry":expires})
    parsed.sort(key=lambda item:item["start"])
    gaps = []
    if approval_start is not None and parsed and parsed[0]["start"] > approval_start:
        gaps.append((approval_start,parsed[0]["start"]))
    elif approval_start is not None and not parsed and now > approval_start:
        gaps.append((approval_start,now))
    for index,item in enumerate(parsed):
        next_start = parsed[index+1]["start"] if index+1<len(parsed) else None
        effective_end = min(item["expiry"],next_start) if next_start is not None else item["expiry"]
        gap_end = next_start if next_start is not None else now
        if effective_end < gap_end:
            gaps.append((effective_end,gap_end))
    choices = Counter((item["row"]["entry_permission"],item["row"]["bias"],
        item["row"]["exposure"],len(item["row"].get("blackouts",[]))) for item in parsed)
    current = load_fast_control(now)
    current_row = None
    if CONTROL_FILE.is_file() and current.status in {"observed","stale"}:
        current_row = read_json(CONTROL_FILE)
    return {"journal":str(CONTROL_JOURNAL),"journal_state":"observed" if CONTROL_JOURNAL.is_file() else "missing",
        "journal_approved_window_count":len(parsed),"coverage_start_at_utc":None if approval_start is None else approval_start.isoformat(),
        "approval_expiry_gaps":{"count":len(gaps),"total_minutes":round(sum((end-start).total_seconds()/60 for start,end in gaps),1),
            "latest":None if not gaps else {"start_utc":gaps[-1][0].isoformat(),"end_utc":gaps[-1][1].isoformat()}},
        "approved_window_counts_by_choice":[{"entry_permission":permission,"bias":bias,
            "exposure":exposure,"blackout_count":blackouts,"window_count":count}
            for (permission,bias,exposure,blackouts),count in sorted(choices.items())],
        "latest_approval":None if not parsed else {"decision_id":parsed[-1]["row"]["decision_id"],
            "start_utc":parsed[-1]["start"].isoformat(),"valid_until_utc":parsed[-1]["expiry"].isoformat(),
            "bias":parsed[-1]["row"]["bias"],"entry_permission":parsed[-1]["row"]["entry_permission"],
            "exposure":parsed[-1]["row"]["exposure"],"blackout_count":len(parsed[-1]["row"].get("blackouts",[]))},
        "current_control":{"status":current.status,
            "decision_id":current.decision_id,"bias":current.bias,"entry_permission":current.entry_permission,
            "exposure":current.exposure,"long_leverage":current.leverage_for("long",now),
            "short_leverage":current.leverage_for("short",now),
            "blackouts":[{"start_utc":start.isoformat(),"end_utc":end.isoformat()} for start,end in current.blackouts],
            "valid_until_utc":None if current_row is None else current_row.get("valid_until_utc"),
            "current_file_matches_latest_journal":None if current_row is None or not parsed else
                current_row.get("decision_id")==parsed[-1]["row"].get("decision_id")}}


def _operational_caveats(record):
    update = record.get("operational_update",{})
    information = update.get("information_path_update_20261004",{})
    repair = information.get("entry_data_repair",{})
    recovery = record.get("process_recovery",{})
    failures = [{"account":row.get("account"),"at_utc":row.get("at_utc"),
        "acknowledged_at_utc":row.get("acknowledged_at_utc")} for row in recovery.get("events",[])
        if row.get("status")=="failed"]
    return {"uptime_claim":"No historical uptime is inferred from process, log, or recovery snapshots; elapsed time since last DB entry is not uptime-adjusted inactivity.",
        "financial_method":{"closed_net":"sum of Trade.close_profit_abs; compared with Trade.realized_profit",
            "gross_price_pnl":"chronological filled quantity and actual average fill price; average is derived from filled cost only when average is absent",
            "estimated_fees":"filled notional times Trade.fee_open for entry fills and Trade.fee_close for exit fills",
            "reconciliation":"closed_net - (gross_price_pnl - estimated_fees + recorded_funding); funding follows stored Freqtrade sign"},
        "recorded_reboot_observation_gap":{"reboot_at_utc":update.get("reboot_at_utc"),
            "six_existing_accounts_restart_at_utc":update.get("six_existing_accounts_restart_at_utc"),
            "description":update.get("observation_gap")},
        "recorded_entry_data_repair":{"file":repair.get("file"),"affected_accounts":repair.get("affected_accounts"),
            "activation":repair.get("activation")},
        "recorded_process_recovery":{"last_check_at_utc":recovery.get("last_check_at_utc"),
            "initial_startup_failures":failures}}


def learning_review_snapshot(now=None):
    """Read-only closed-fill reconciliation, account dates, manual journal and control coverage."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("Learning review requires a timezone-aware clock")
    record = read_json(RECORD)
    reports, proxy_groups = [], defaultdict(set)
    specs = account_specs_for_record(record)
    for spec in specs:
        info = account_record(record,spec)
        lifecycle = lifecycle_for(record,spec)
        expected_db = f"{spec.key}_trades.sqlite"
        if info.get("db") != expected_db or info.get("strategy") != spec.strategy:
            raise ValueError(f"{spec.key}: recorded database/strategy identity does not match allowlist")
        path = (REPORT/info["db"]).resolve()
        if path.parent != REPORT.resolve():
            raise FileNotFoundError(f"{spec.key}: expected database is outside report folder: {path}")
        if not path.is_file() and spec.new_identity and not info.get("database_initialized_at_utc"):
            reports.append({"account":spec.key,"strategy":spec.strategy,"lifecycle":lifecycle,
                "database_state":"not_initialized","recorded_dates":{},
                "first_db_entry_at_utc":None,"last_db_entry_at_utc":None,
                "elapsed_minutes_since_last_recorded_entry":None,
                "db_trade_counts":None,"closed_financials":None})
            continue
        if spec.new_identity and path.is_file() and not info.get("database_initialized_at_utc"):
            raise ValueError(f"{spec.key}: DB exists without a registered successful initialization")
        if not path.is_file():
            raise FileNotFoundError(f"{spec.key}: expected existing database missing or outside report folder: {path}")
        engine = create_engine("sqlite:///file:"+path.as_posix()+"?mode=ro&uri=true")
        with Session(engine) as session:
            trades = list(session.scalars(select(Trade).options(selectinload(Trade.orders))))
            opened = [trade for trade in trades if trade.is_open]
            closed = [trade for trade in trades if not trade.is_open]
            financials = _closed_financials(trades)
            entries = [trade.open_date_utc for trade in trades]
            first_entry = min(entries) if entries else None
            last_entry = max(entries) if entries else None
            if last_entry is not None and last_entry > now+timedelta(minutes=5):
                raise ValueError(f"{spec.key}: database has a future trade open time")
            dates = {key:info.get(key) for key in ("started_at_utc","restarted_at_utc","parent_created_at_utc") if info.get(key)}
            if info.get("last_recovery",{}).get("at_utc"):
                dates["last_recovery_at_utc"] = info["last_recovery"]["at_utc"]
            reports.append({"account":spec.key,"strategy":spec.strategy,"lifecycle":lifecycle,"database_state":"initialized",
                "entry_mode":info.get("entry_mode"),"recorded_dates":dates,
                "first_db_entry_at_utc":None if first_entry is None else first_entry.isoformat(),
                "last_db_entry_at_utc":None if last_entry is None else last_entry.isoformat(),
                "elapsed_minutes_since_last_recorded_entry":None if last_entry is None else round((now-last_entry).total_seconds()/60,1),
                "db_trade_counts":{"all":len(trades),"open":len(opened),"closed":len(closed)},
                "closed_financials":financials})
            for trade in trades:
                minute = trade.open_date_utc.replace(second=0,microsecond=0)
                side = "short" if trade.is_short else "long"
                proxy_groups[(trade.pair,side,minute)].add(spec.key)
        engine.dispose()
    shared = Counter((pair,side,tuple(sorted(accounts))) for (pair,side,_),accounts in proxy_groups.items() if len(accounts)>1)
    shared_summary = [{"pair":pair,"side":side,"accounts":accounts,"shared_open_minute_groups":count}
        for (pair,side,accounts),count in sorted(shared.items())]
    manual_keys = ["manual","news_manual","news_lab"]
    if any(spec.key == "news_fast" for spec in specs):
        manual_keys.append("news_fast")
    manual_keys.extend(key for key in AGGRESSIVE_ACCOUNT_KEYS if any(spec.key == key for spec in specs))
    manual = {key:_journal_summary(key) for key in manual_keys}
    control_history = _control_history_snapshot(record,now)
    return {"observed_at_utc":now.isoformat(),"source":"allowlisted existing local account databases and journals; read-only",
        "account_count":len(reports),"active_entry_enabled_count":sum(row.get("lifecycle")=="ACTIVE" for row in reports),
        "draining_or_parked_count":sum(row.get("lifecycle") in {"DRAINING","PARKED"} for row in reports),
        "accounts":reports,"manual_decisions":manual,
        "entry_proxies":{"basis":"same pair, side and Trade.open_date minute across paper accounts",
            "shared_group_counts_by_pair_side":shared_summary,
            "caveat":"Proxy grouping is not proof of identical source candidates; approval windows are not uptime."},
        "fast_context_control_history":control_history,
        "operational_caveats":_operational_caveats(record)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources",action="store_true")
    parser.add_argument("--market",action="store_true")
    parser.add_argument("--crypto",action="store_true",help="Compact public futures/breadth context; no trading authority")
    parser.add_argument("--review-context",action="store_true",help="Current recorded identities, due watches and unchanged schema-1 control")
    parser.add_argument("--higher",action="store_true",help="Only if local BTC/ETH 4h/daily candles are unavailable: four public requests")
    parser.add_argument("--accounts",action="store_true")
    parser.add_argument("--prices",action="store_true")
    parser.add_argument("--record-daily",action="store_true",
        help="Persist each account's first observed equity/BTC quote for this UTC date; implies priced accounts")
    parser.add_argument("--account-table",action="store_true",help="Print the account table as Markdown; optionally include --prices")
    parser.add_argument("--learning-review",action="store_true",help="User-requested read-only manual learning/accounting review; local databases and journals only")
    args=parser.parse_args();now=datetime.now(timezone.utc)
    result={"observed_at_utc":now.isoformat()}
    if args.sources:result["sources"]=source_snapshot(now)
    if args.market:result["market"]=market_snapshot()
    if args.crypto:result["crypto_derivatives"]=collect_crypto_derivatives_snapshot(now)
    if args.review_context:result["review_context"]=review_context_snapshot(now)
    if args.higher:result["higher_timeframes"]=higher_snapshot(now)
    if args.accounts or args.prices or args.account_table or args.record_daily:
        priced=args.prices or args.record_daily
        if priced:
            price_context={}
            accounts=account_snapshot(True,now,price_context)
            accounts=enrich_account_returns(accounts,now,price_context,record_daily=args.record_daily)
            result["paper_return_basis"]={
                "equity":"configured dry_run_wallet + recorded net banked P/L + estimated remaining open P/L",
                "btc_benchmark":BTC_PERPETUAL_SOURCE,
                "baseline":"open price of earliest BTCUSDT 1m futures candle at/after the recorded attempt start, within 60s",
                "daily_observation":"first actual --record-daily observation per attempt/UTC date; not exact midnight or a full 24-hour observation",
            }
            if args.record_daily:
                result["daily_record"]={"path":str(PAPER_DAILY_HISTORY),"status":"saved_or_already_recorded",
                    "observation_date_utc":now.date().isoformat()}
        else:
            accounts=account_snapshot(False,now)
        result["accounts"]=accounts
        result["account_table"]=format_account_table(accounts,now.isoformat())
    if args.learning_review:result["learning_review"]=learning_review_snapshot(now)
    if args.account_table and not (args.sources or args.market or args.crypto or args.review_context or args.higher or args.accounts or args.learning_review or args.record_daily):
        print(result["account_table"])
        return
    print(json.dumps(result,separators=(",",":"),allow_nan=False))


if __name__=="__main__":main()
