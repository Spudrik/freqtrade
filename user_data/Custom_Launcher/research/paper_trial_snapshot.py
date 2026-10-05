"""Small read-only paper/source snapshot for scheduled reviews; writes no files.

Reads bounded local rows and existing account helpers, not historical research
exports. Optional public price lookup is only for estimated current open P/L.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime, timedelta, timezone
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
from freqtrade.persistence import Order, Trade
from user_data.Custom_Launcher.research.paper_trial_runtime import ROOT, REPORT, RECORD, ACCOUNTS, account_record, account_specs_for_record, lifecycle_for, process_inventory, matching_processes, worker_heartbeat_state
from user_data.Custom_Launcher.research.paper_trial_control import _configs, _api, _journal, _verify_running_account
from user_data.Custom_Launcher.research.crypto_derivatives_snapshot import collect_crypto_derivatives_snapshot
from user_data.strategies.integrated_paper_context import _utc
from user_data.strategies.paper_fast_context import CONTROL_FILE, load_fast_control
from user_data.strategies.paper_fast_context import CONTROL_JOURNAL, parse_fast_control
from user_data.strategies.paper_news_manual import decision_rows
from user_data.strategies.paper_trial_level_orderbook import _recent_pressure


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def source_snapshot(now):
    output, headlines, seen, titles = {}, [], set(), {}
    cutoff = (now-timedelta(hours=4)).isoformat()
    for key, folder in (("news","research_news_data/news"),("web","research_news_data/web"),
                        ("global","research_news_data/global_context"),("orderbook","orderbook_data/live")):
        path = ROOT/"user_data"/folder/"collector_status.json"
        if not path.is_file():
            output[key] = {"status":"missing"}; continue
        status = read_json(path)
        output[key] = {name:status.get(name) for name in ("status","heartbeat_at","last_fetch_at","last_message_at","last_error","sources_enabled","sources_total","capacity_level") if name in status}
        db = Path(status["db_path"]).resolve()
        with closing(sqlite3.connect(db.as_uri()+"?mode=ro",uri=True,timeout=2)) as con:
            con.row_factory = sqlite3.Row
            if key in {"news","web"}:
                rows = con.execute("SELECT title,canonical_url,source_url,published_at,collected_at,source_group FROM articles WHERE collected_at>=? ORDER BY collected_at DESC LIMIT 16",(cutoff,)).fetchall()
                for row in rows:
                    item=dict(row); url=item["canonical_url"] or item["source_url"]
                    if not url or url in seen:
                        continue
                    seen.add(url)
                    title = " ".join(str(item["title"] or "").casefold().split())
                    if not title:
                        continue
                    duplicate = titles.get(title)
                    if duplicate is not None:
                        duplicate["additional_sources"].append({"url":url,"published_at":item["published_at"],
                            "collected_at":item["collected_at"],"source_group":item["source_group"]})
                        continue
                    entry = {"collector":key,"title":item["title"],"url":url,
                        "published_at":item["published_at"],"collected_at":item["collected_at"],
                        "source_group":item["source_group"],"additional_sources":[]}
                    titles[title] = entry
                    headlines.append(entry)
            elif key=="global":
                sources=con.execute("SELECT source_id,source_group,enabled,url,last_success_at,last_failure_at,last_error FROM context_sources ORDER BY source_id LIMIT 25").fetchall()
                output["global"]["sources"]=[]
                for source in sources:
                    tick=con.execute("SELECT ts,source_ts,metric_key,value,unit,notes FROM global_context_ticks WHERE source_id=? ORDER BY ts DESC LIMIT 1",(source["source_id"],)).fetchone()
                    metrics=con.execute("SELECT t.ts,t.source_ts,t.metric_key,t.value,t.unit,t.notes FROM global_context_ticks t JOIN (SELECT metric_key,MAX(ts) AS ts FROM global_context_ticks WHERE source_id=? GROUP BY metric_key) latest ON t.metric_key=latest.metric_key AND t.ts=latest.ts WHERE t.source_id=? ORDER BY t.metric_key LIMIT 20",(source["source_id"],source["source_id"])).fetchall() if source["enabled"] else []
                    output["global"]["sources"].append({**dict(source),
                        "last_observation_at":None if tick is None else tick["source_ts"],
                        "last_collected_at":None if tick is None else tick["ts"],
                        "latest_metrics":[dict(metric) for metric in metrics],"metric_limit":20})
    output["recent_unique_headlines"] = headlines[:20]
    output["book_pressure"] = {pair:{"imbalance":value,"coverage":coverage} for pair in
        ("BTC/USDT:USDT","ETH/USDT:USDT","SOL/USDT:USDT","BNB/USDT:USDT","DOGE/USDT:USDT","1000PEPE/USDT:USDT")
        for value,coverage in [_recent_pressure(pair,now)]}
    return output


def market_snapshot():
    _,overlay=_configs("manual")
    _verify_running_account(overlay)
    output={}
    for pair in ("BTC/USDT:USDT","ETH/USDT:USDT","SOL/USDT:USDT","BNB/USDT:USDT","DOGE/USDT:USDT","1000PEPE/USDT:USDT"):
        data=_api(overlay,"pair_candles?"+urlencode({"pair":pair,"timeframe":"1h","limit":4}))
        selected={name:index for index,name in enumerate(data["columns"]) if name in
            {"date","open","high","low","close","volume","paper_atr","vp_poc_4h","vp_hvn_above_4h","vp_hvn_below_4h","vp_lvn_above_4h","vp_lvn_below_4h","rolling_20_high_4h","rolling_20_low_4h","prior_24h_high","prior_24h_low"}}
        output[pair]=[{name:row[index] for name,index in selected.items()} for row in data["data"][-4:]]
    return {"source":"Existing verified local IntegratedPaper manual read-only analyzed-candle API",
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
                control_times = {key:encoded.get(key) for key in ("observed_at_utc","valid_until_utc")}
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
    return {"trial":record.get("trial"),"run_record_status":record.get("status"),
        "accounts":[{"account":spec.key,"lifecycle":lifecycle_for(record,spec),"strategy":spec.strategy,"config":spec.config,
            "db":account_record(record,spec)["db"],"log":account_record(record,spec)["log"],
            "api_port":spec.port} for spec in account_specs_for_record(record)],
        "approved_watches":approved,"due_watches":due,"issues":issues,
        "fast_context":fast_context}


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


def _protection_state(trade, session):
    """Reuse the strategy's persisted-plan validators, including inverse/manual bots."""
    if not isfinite(float(trade.stop_loss or 0)) or float(trade.stop_loss or 0) <= 0:
        return "missing_or_invalid"
    try:
        if trade.strategy in {"PaperFastAuto","PaperFastContext","PaperFastLevelInverse"}:
            from user_data.strategies.paper_fast_reaction import validate_fast_plan
            encoded = session.execute(text("SELECT cd_value FROM trade_custom_data WHERE ft_trade_id=:trade_id AND cd_key=:key"),
                {"trade_id":trade.id,"key":"paper_fast_plan"}).scalar_one_or_none()
            validate_fast_plan(json.loads(encoded) if encoded is not None else None)
        elif trade.strategy == "PaperNewsManual":
            from user_data.strategies.paper_news_manual import validate_manual_plan
            encoded = session.execute(text("SELECT cd_value FROM trade_custom_data WHERE ft_trade_id=:trade_id AND cd_key=:key"),
                {"trade_id":trade.id,"key":"paper_news_manual_plan"}).scalar_one_or_none()
            plan = json.loads(encoded) if encoded is not None else None
            validate_manual_plan(plan,float(plan["reference_rate"]))
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


def account_snapshot(prices=False):
    record=read_json(RECORD); processes=process_inventory(); quotes={}
    if prices:
        with urlopen("https://fapi.binance.com/fapi/v1/ticker/price",timeout=12) as response:
            quotes={row["symbol"]:float(row["price"]) for row in json.load(response)}
    output=[]
    for spec in account_specs_for_record(record):
        row=account_record(record,spec); path=(REPORT/row["db"]).resolve()
        tree=matching_processes(spec,processes)
        lifecycle=lifecycle_for(record,spec)
        if not path.is_file() and spec.new_identity and not row.get("database_initialized_at_utc"):
            output.append({"account":spec.key,"lifecycle":lifecycle,"database_state":"not_initialized",
                "running_exact_tree":tree is not None,"worker_log":_worker_log_health(spec,tree,datetime.now(timezone.utc),lifecycle),
                "open_longs":None,"open_shorts":None,"closed_longs":None,"closed_shorts":None,
                "wins":None,"losses":None,"banked_pnl_usdt":None,"estimated_open_pnl_usdt":None,
                "open_order_count":None,"protection_issue_count":None,"open_positions":None})
            continue
        if spec.new_identity and path.is_file() and not row.get("database_initialized_at_utc"):
            raise ValueError(f"{spec.key}: DB exists without a registered successful initialization")
        engine=create_engine("sqlite:///file:"+path.as_posix()+"?mode=ro&uri=true")
        with Session(engine) as session:
            trades=list(session.scalars(select(Trade))); opened=[t for t in trades if t.is_open]; closed=[t for t in trades if not t.is_open]
            banked=sum(float(t.close_profit_abs or 0) for t in closed)+sum(float(t.realized_profit or 0) for t in opened)
            unrealized=sum(t.calculate_profit(quotes[t.pair.split(':')[0].replace('/','')]).profit_abs for t in opened) if prices else (0. if not opened else None)
            positions=[{"pair":t.pair,"short":t.is_short,"leverage":t.leverage,"stop_price":t.stop_loss,
                "margin":t.stake_amount,"protection_state":_protection_state(t,session)} for t in opened]
            open_order_count=session.scalar(select(func.count(Order.id)).where(Order.ft_is_open.is_(True)))
            output.append({"account":spec.key,"lifecycle":lifecycle,"database_state":"initialized","running_exact_tree":tree is not None,
                "worker_log":_worker_log_health(spec,tree,datetime.now(timezone.utc),lifecycle),
                "open_order_count":open_order_count,
                "open_longs":sum(not t.is_short for t in opened),"open_shorts":sum(t.is_short for t in opened),
                "closed_longs":sum(not t.is_short for t in closed),"closed_shorts":sum(t.is_short for t in closed),
                "wins":sum(float(t.close_profit_abs or 0)>0 for t in closed),"losses":sum(float(t.close_profit_abs or 0)<0 for t in closed),
                "banked_pnl_usdt":banked,"estimated_open_pnl_usdt":unrealized,
                "protection_issue_count":sum(p["protection_state"]!="stored_valid" for p in positions),
                "parked_activity_issue":lifecycle=="PARKED" and (bool(opened) or bool(open_order_count)),
                "parked_process_issue":lifecycle=="PARKED" and tree is not None,
                "open_positions":positions})
        engine.dispose()
    return output


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
    manual = {key:_journal_summary(key) for key in ("manual","news_manual","news_lab")}
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
    parser.add_argument("--learning-review",action="store_true",help="User-requested read-only manual learning/accounting review; local databases and journals only")
    args=parser.parse_args();now=datetime.now(timezone.utc)
    result={"observed_at_utc":now.isoformat()}
    if args.sources:result["sources"]=source_snapshot(now)
    if args.market:result["market"]=market_snapshot()
    if args.crypto:result["crypto_derivatives"]=collect_crypto_derivatives_snapshot(now)
    if args.review_context:result["review_context"]=review_context_snapshot(now)
    if args.higher:result["higher_timeframes"]=higher_snapshot(now)
    if args.accounts or args.prices:result["accounts"]=account_snapshot(args.prices)
    if args.learning_review:result["learning_review"]=learning_review_snapshot(now)
    print(json.dumps(result,separators=(",",":"),allow_nan=False))


if __name__=="__main__":main()
