"""ExplorerAcceptanceScoreV1.

Lower score is better. Trade count, profit total, winrate, and losing-window count are
reported for human review but are not acceptance gates.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any

MAX_ALLOWED_DRAWDOWN_PCT = 30.0
SOFT_DRAWDOWN_PCT = 10.0
DRAWDOWN_PENALTY_SCALE = 8.0
DRAWDOWN_PENALTY_WEIGHT = 100.0


@dataclass(frozen=True)
class ScoredWindow:
    window: str
    regime: str
    objective: float
    max_drawdown_pct: float
    drawdown_penalty: float
    final_score: float
    profit_total: float
    trade_count: int
    winrate: float
    hard_guard: str


@dataclass(frozen=True)
class AggregateScore:
    objective_total: float
    max_drawdown_pct: float
    drawdown_penalty_total: float
    final_score: float
    profit_total: float
    trade_count: int
    losing_window_count: int
    winrate: float
    hard_guard: str
    windows: list[dict[str, Any]]


def numeric(metrics: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return float(default)


def normalize_drawdown_pct(value: float | int | None) -> float:
    if value is None:
        return 0.0
    raw = abs(float(value))
    # Freqtrade commonly stores relative drawdown as a ratio. Some parsed fields
    # may already be percentages; values above 1.5 are treated as percentages.
    return raw * 100.0 if raw <= 1.5 else raw


def max_drawdown_pct(metrics: dict[str, Any]) -> float:
    for key in ("max_relative_drawdown", "max_drawdown_account", "max_drawdown_abs", "drawdown"):
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return normalize_drawdown_pct(value)
    return 0.0


def drawdown_penalty(drawdown_pct: float) -> float:
    excess = max(0.0, float(drawdown_pct) - SOFT_DRAWDOWN_PCT)
    return (math.exp(excess / DRAWDOWN_PENALTY_SCALE) - 1.0) * DRAWDOWN_PENALTY_WEIGHT


def hard_guard(drawdown_pct: float, max_allowed_drawdown_pct: float = MAX_ALLOWED_DRAWDOWN_PCT) -> str:
    return "FAIL" if float(drawdown_pct) > float(max_allowed_drawdown_pct) else "PASS"


def profit_total(metrics: dict[str, Any]) -> float:
    return numeric(metrics, "profit_total_abs", "profit_total", "profit_total_pct", default=0.0)


def trade_count(metrics: dict[str, Any]) -> int:
    return int(round(numeric(metrics, "total_trades", default=0.0)))


def winrate(metrics: dict[str, Any]) -> float:
    value = numeric(metrics, "winrate", default=0.0)
    return value * 100.0 if 0.0 <= value <= 1.0 else value


def score_window(window: dict[str, Any], metrics: dict[str, Any], objective: float) -> ScoredWindow:
    dd = max_drawdown_pct(metrics)
    penalty = drawdown_penalty(dd)
    return ScoredWindow(
        window=str(window.get("name") or window.get("timerange") or ""),
        regime=str(window.get("regime") or ""),
        objective=float(objective),
        max_drawdown_pct=float(dd),
        drawdown_penalty=float(penalty),
        final_score=float(objective) + float(penalty),
        profit_total=float(profit_total(metrics)),
        trade_count=int(trade_count(metrics)),
        winrate=float(winrate(metrics)),
        hard_guard=hard_guard(dd),
    )


def aggregate(scored_windows: list[ScoredWindow]) -> AggregateScore:
    if not scored_windows:
        return AggregateScore(0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0.0, "FAIL", [])
    trade_total = sum(item.trade_count for item in scored_windows)
    weighted_winrate_total = sum(item.winrate * max(1, item.trade_count) for item in scored_windows)
    weighted_trade_total = sum(max(1, item.trade_count) for item in scored_windows)
    max_dd = max(item.max_drawdown_pct for item in scored_windows)
    return AggregateScore(
        objective_total=sum(item.objective for item in scored_windows),
        max_drawdown_pct=float(max_dd),
        drawdown_penalty_total=sum(item.drawdown_penalty for item in scored_windows),
        final_score=sum(item.final_score for item in scored_windows),
        profit_total=sum(item.profit_total for item in scored_windows),
        trade_count=int(trade_total),
        losing_window_count=sum(1 for item in scored_windows if item.profit_total < 0.0),
        winrate=(weighted_winrate_total / weighted_trade_total) if weighted_trade_total else 0.0,
        hard_guard=hard_guard(max_dd),
        windows=[asdict(item) for item in scored_windows],
    )


def delta(champion: AggregateScore, challenger: AggregateScore) -> dict[str, float | int]:
    return {
        "objective_total": challenger.objective_total - champion.objective_total,
        "max_drawdown_pct": challenger.max_drawdown_pct - champion.max_drawdown_pct,
        "drawdown_penalty_total": challenger.drawdown_penalty_total - champion.drawdown_penalty_total,
        "final_score": challenger.final_score - champion.final_score,
        "profit_total": challenger.profit_total - champion.profit_total,
        "trade_count": challenger.trade_count - champion.trade_count,
        "losing_window_count": challenger.losing_window_count - champion.losing_window_count,
        "winrate": challenger.winrate - champion.winrate,
    }


def compare(champion: AggregateScore, challenger: AggregateScore) -> dict[str, Any]:
    if challenger.hard_guard == "FAIL":
        accepted = False
        decision_code = "REJECTED_DRAWDOWN_GUARD"
    elif challenger.final_score < champion.final_score:
        accepted = True
        decision_code = "ACCEPTED_SCORE_IMPROVED"
    else:
        accepted = False
        decision_code = "REJECTED_SCORE_NOT_IMPROVED"
    return {
        "accepted": accepted,
        "decision_code": decision_code,
        "guard": challenger.hard_guard,
        "champion": asdict(champion),
        "challenger": asdict(challenger),
        "delta": delta(champion, challenger),
    }


def score_table_rows(comparison: dict[str, Any]) -> list[dict[str, Any]]:
    champion = comparison.get("champion") or {}
    challenger = comparison.get("challenger") or {}
    delta_values = comparison.get("delta") or {}
    rows = []
    specs = [
        ("Objective total", "objective_total", "Score"),
        ("Max drawdown %", "max_drawdown_pct", "Hard guard"),
        ("Drawdown penalty", "drawdown_penalty_total", "Score"),
        ("Final score", "final_score", "Decision"),
        ("Profit total", "profit_total", "Info"),
        ("Trade count", "trade_count", "Info"),
        ("Losing windows", "losing_window_count", "Info"),
        ("Winrate", "winrate", "Info"),
    ]
    for label, key, role in specs:
        rows.append(
            {
                "metric": label,
                "champion": champion.get(key),
                "challenger": challenger.get(key),
                "delta": delta_values.get(key),
                "role": role,
            }
        )
    return rows
