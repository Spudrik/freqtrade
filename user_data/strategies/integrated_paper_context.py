"""Strict, paper-only contract for a Luna-produced market observation.

The observation is optional, but a present malformed or expired observation is
never interpreted as a calm market.  Malformed input blocks new entries;
expired input is reported as stale and supplies no directional vote.
Only source-grounded, time-bounded assertions can affect paper decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from urllib.parse import urlparse


CONTEXT_FILE = (
    Path(__file__).resolve().parents[1]
    / "research_news_data/context_features/integrated_paper_20260926/luna_context.json"
)


@dataclass(frozen=True)
class LunaContext:
    status: str
    observed_at: datetime | None = None
    risk_bias: str = "unknown"
    event_scale: str = "none"
    attention: str = "unknown"
    event_id: str = ""
    sources: tuple[str, ...] = ()
    brief: dict | None = None
    watch_proposals: tuple[dict, ...] = ()


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("Luna context timestamps must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def load_luna_context(now: datetime, path: Path = CONTEXT_FILE) -> LunaContext:
    """Read one atomic snapshot; stale/missing is explicit and never a neutral vote."""
    if now.tzinfo is None:
        raise ValueError("Decision time must be timezone-aware")
    if not path.is_file():
        return LunaContext(status="missing")
    with path.open(encoding="utf-8") as handle:
        row = json.load(handle)
    return parse_luna_context(row, now)


def parse_luna_context(row: dict, now: datetime) -> LunaContext:
    """Validate both legacy observations and optional discretionary-account briefs."""
    if now.tzinfo is None:
        raise ValueError("Decision time must be timezone-aware")
    if not isinstance(row, dict) or row.get("schema_version") != 1:
        raise ValueError("Luna context schema_version must be 1")
    observed = _utc(row["observed_at_utc"])
    expires = _utc(row["valid_until_utc"])
    clock = now.astimezone(timezone.utc)
    if observed > clock + timedelta(minutes=2) or expires <= observed:
        raise ValueError("Luna context has an impossible observation window")
    if expires - observed > timedelta(hours=6):
        raise ValueError("Luna context validity cannot exceed six hours")
    if clock >= expires:
        return LunaContext(status="stale", observed_at=observed)

    bias = row["risk_bias"]
    scale = row["event_scale"]
    attention = row["attention"]
    if bias not in {"risk_on", "risk_off", "mixed", "unknown"}:
        raise ValueError("Invalid Luna risk_bias")
    if scale not in {"none", "minor", "major"}:
        raise ValueError("Invalid Luna event_scale")
    if attention not in {"normal", "elevated", "unknown"}:
        raise ValueError("Invalid Luna attention")
    sources = row["sources"]
    if not isinstance(sources, list) or not all(isinstance(url, str) for url in sources):
        raise ValueError("Luna sources must be a URL list")
    if any(urlparse(url).scheme not in {"http", "https"} or not urlparse(url).netloc
           for url in sources):
        raise ValueError("Luna sources must be absolute HTTP(S) URLs")
    if bias in {"risk_on", "risk_off"} and not sources:
        raise ValueError("A directional Luna risk assessment requires source URLs")
    event_id = row.get("event_id", "")
    if not isinstance(event_id, str) or len(event_id) > 120:
        raise ValueError("Invalid Luna event_id")
    brief = row.get("brief")
    if brief is not None and not isinstance(brief, dict):
        raise ValueError("Luna brief must be an object")
    watches = row.get("watch_proposals", [])
    if not isinstance(watches, list) or any(not isinstance(w, dict) for w in watches):
        raise ValueError("Luna watch_proposals must be a list of objects")
    for watch in watches:
        if not all(isinstance(watch.get(key), str) and watch[key].strip()
                   for key in ("event_id", "source", "rationale")):
            raise ValueError("Watch proposals need event_id, source and rationale")
        if watch["source"] not in sources:
            raise ValueError("Watch source must be included in Luna source URLs")
        _utc(watch["event_time_utc"])
        if not isinstance(watch.get("proposed_checks_utc"), list):
            raise ValueError("Watch proposals need explicit proposed check times")
        for check in watch["proposed_checks_utc"]:
            _utc(check)
    return LunaContext(
        status="observed", observed_at=observed, risk_bias=bias,
        event_scale=scale, attention=attention, event_id=event_id,
        sources=tuple(sources),
        brief=brief, watch_proposals=tuple(watches),
    )


def publish_luna_context(row: dict, path: Path = CONTEXT_FILE,
                         now: datetime | None = None) -> None:
    """Validate before atomically replacing the shared file; never expose partial JSON."""
    from user_data.Custom_Launcher.research.context_features.global_context_source_preflight import write_text_atomic

    clock = now or datetime.now(timezone.utc)
    if parse_luna_context(row, clock).status != "observed":
        raise ValueError("Cannot publish an already expired Luna observation")
    encoded = json.dumps(row, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, encoded)
