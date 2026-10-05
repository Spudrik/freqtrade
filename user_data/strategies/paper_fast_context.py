"""Main-agent-only, expiring controls for the matched fast PAPER account."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

from user_data.strategies.integrated_paper_context import CONTEXT_FILE, _utc, load_luna_context

REPORT = CONTEXT_FILE.parent
CONTROL_FILE = REPORT / "fast_context.json"
CONTROL_JOURNAL = REPORT / "fast_context_decisions.jsonl"
LEVERAGES = {
    -2: {"long": 0., "short": 5.},
    -1: {"long": 2., "short": 4.},
    0: {"long": 3., "short": 3.},
    1: {"long": 4., "short": 2.},
    2: {"long": 5., "short": 0.},
}


@dataclass(frozen=True)
class FastControl:
    status: str
    decision_id: str = "unavailable"
    bias: int = 0
    exposure: str = "reduced"
    entry_permission: str = "paused"
    blackouts: tuple[tuple[datetime, datetime], ...] = ()

    def leverage_for(self, side: str, now: datetime) -> float:
        if (self.status != "observed" or self.entry_permission == "paused"
                or any(start <= now < end for start, end in self.blackouts)):
            return 0.
        return LEVERAGES[self.bias][side]

    @property
    def size_factor(self) -> float:
        return 1. if self.exposure == "normal" else .5


def parse_fast_control(row: dict, now: datetime) -> FastControl:
    if now.tzinfo is None or not isinstance(row, dict):
        raise ValueError("Control requires a dictionary and timezone-aware decision time")
    if row.get("schema_version") != 1 or row.get("author") != "main_agent":
        raise ValueError("Only schema-1 main-agent controls are accepted")
    if not isinstance(row.get("decision_id"), str) or not row["decision_id"].strip():
        raise ValueError("Control requires a decision ID")
    observed, expires = _utc(row["observed_at_utc"]), _utc(row["valid_until_utc"])
    if observed > now + timedelta(minutes=2) or not timedelta(0) < expires-observed <= timedelta(hours=4):
        raise ValueError("Invalid control observation/expiry window")
    bias = row["bias"]
    if type(bias) is not int or bias not in LEVERAGES:
        raise ValueError("Bias must be an integer from -2 to +2")
    if row["exposure"] not in {"normal", "reduced"}:
        raise ValueError("Exposure must be normal or reduced")
    if row["entry_permission"] not in {"normal", "strong_only", "paused"}:
        raise ValueError("Invalid entry permission")
    if not isinstance(row.get("reason"), str) or not row["reason"].strip():
        raise ValueError("A control needs a reason")
    if not isinstance(row.get("sources"), list) or not row["sources"]:
        raise ValueError("A control needs fresh Luna source references")
    blackouts = []
    for window in row.get("blackouts", []):
        start, end = _utc(window["start_utc"]), _utc(window["end_utc"])
        if start >= end or end-start > timedelta(hours=2):
            raise ValueError("Blackout must be a bounded interval of at most two hours")
        blackouts.append((start, end))
    if now >= expires:
        return FastControl("stale", row["decision_id"])
    return FastControl("observed", row["decision_id"], bias, row["exposure"],
                       row["entry_permission"], tuple(blackouts))


def load_fast_control(now: datetime, path: Path = CONTROL_FILE) -> FastControl:
    if not path.is_file():
        return FastControl("missing")
    # Expected input faults block entries explicitly; never reinterpret them as calm news.
    try:
        return parse_fast_control(json.loads(path.read_text(encoding="utf-8")), now)
    except (ValueError, KeyError, TypeError):
        return FastControl("malformed")


def publish_fast_control(row: dict, *, now: datetime | None = None,
                         path: Path = CONTROL_FILE, journal: Path = CONTROL_JOURNAL) -> None:
    from user_data.Custom_Launcher.research.context_features.global_context_source_preflight import write_text_atomic
    clock = now or datetime.now(timezone.utc)
    if parse_fast_control(row, clock).status != "observed":
        raise ValueError("Cannot publish an expired control")
    luna = load_luna_context(clock)
    if luna.status != "observed" or not set(row["sources"]) <= set(luna.sources):
        raise ValueError("Control sources must belong to the fresh Luna observation")
    if _utc(row["valid_until_utc"]) > _utc(json.loads(CONTEXT_FILE.read_text(encoding="utf-8"))["valid_until_utc"]):
        raise ValueError("Control cannot outlive its supporting Luna brief")
    if journal.is_file():
        with journal.open(encoding="utf-8") as stream:
            if any(json.loads(line)["decision_id"] == row["decision_id"] for line in stream if line.strip()):
                raise ValueError("Decision already journalled; inspect current control rather than duplicate it")
    encoded = json.dumps(row, allow_nan=False, sort_keys=True)
    # Journal approval first. If replacement fails, inspect the journal/current snapshot.
    with journal.open("a", encoding="utf-8") as stream:
        stream.write(encoded + "\n")
    write_text_atomic(path, json.dumps(row, indent=2, allow_nan=False) + "\n")
