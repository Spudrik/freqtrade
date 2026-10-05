"""Paper F: level-rejection baseline with a live order-book risk modifier.

This is a forward-only paper test.  Historical Bybit FreqAI findings motivated
testing a conditional risk role, but this deliberately simple Binance futures
pressure measure is a new proxy, not a reproduction of that historical feature.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import sqlite3

from freqtrade.enums import RunMode

from user_data.strategies.paper_trial_level import PaperTrialLevel


LOG = logging.getLogger(__name__)
BOOK_DB = Path(__file__).resolve().parents[1] / "orderbook_data/live/orderbook_events.sqlite"
MARKET_KEY = "binance_usdm_futures"
OBSERVATIONS = 3
MAX_AGE = timedelta(minutes=2)
MAX_GAP = timedelta(seconds=90)
CONTRARY_IMBALANCE = 0.15


def _recent_pressure(pair: str, decision_time: datetime) -> tuple[float | None, str]:
    """Read three completed, consecutive, adequately covered one-minute bars."""
    if not BOOK_DB.is_file():
        return None, "source_missing"
    if decision_time.tzinfo is None:
        raise ValueError("Order-book decision time must be timezone-aware")
    cutoff = decision_time.astimezone(timezone.utc)
    connection = sqlite3.connect(f"file:{BOOK_DB.as_posix()}?mode=ro", uri=True, timeout=2.0)
    try:
        rows = connection.execute(
            "SELECT ts_end, valid_samples, expected_samples, imbalance_top20_mean, market_key "
            "FROM orderbook_metric_bars WHERE pair = ? AND timeframe_seconds = 60 "
            "ORDER BY ts_start DESC LIMIT 80",
            (pair,),
        ).fetchall()
    finally:
        connection.close()
    eligible = []
    for end_text, valid, expected, imbalance, source in rows:
        if source != MARKET_KEY:
            continue
        end = datetime.fromisoformat(end_text).astimezone(timezone.utc)
        if end > cutoff:
            continue
        eligible.append((end, valid, expected, imbalance))
        if len(eligible) == OBSERVATIONS:
            break
    if len(eligible) != OBSERVATIONS:
        return None, "insufficient_bars"
    if cutoff - eligible[0][0] > MAX_AGE:
        return None, "stale"
    if any(older[0] >= newer[0] or newer[0] - older[0] > MAX_GAP
           for newer, older in zip(eligible, eligible[1:])):
        return None, "gap"
    if any(valid is None or expected is None or expected <= 0 or valid / expected < 0.8
           for _, valid, expected, _ in eligible):
        return None, "low_coverage"
    if any(imbalance is None for _, _, _, imbalance in eligible):
        return None, "missing_pressure"
    return sum(float(value) for _, _, _, value in eligible) / OBSERVATIONS, "observed"


class PaperTrialLevelOrderbook(PaperTrialLevel):
    def bot_start(self, **kwargs) -> None:
        if self.config.get("dry_run") is not True or self.config.get("runmode") != RunMode.DRY_RUN:
            raise RuntimeError("PaperTrialLevelOrderbook is dry-run only")
        if not BOOK_DB.is_file():
            raise FileNotFoundError(f"Live order-book database is missing: {BOOK_DB}")

    def custom_stake_amount(
        self, pair, current_time, current_rate, proposed_stake,
        min_stake, max_stake, leverage, entry_tag, side, **kwargs,
    ) -> float:
        normal_stake = super().custom_stake_amount(
            pair, current_time, current_rate, proposed_stake,
            min_stake, max_stake, leverage, entry_tag, side, **kwargs,
        )
        if normal_stake <= 0:
            return normal_stake
        pressure, coverage = _recent_pressure(pair, current_time)
        contrary = coverage == "observed" and (
            pressure <= -CONTRARY_IMBALANCE if side == "long"
            else pressure >= CONTRARY_IMBALANCE
        )
        chosen_stake = normal_stake * 0.5 if contrary else normal_stake
        LOG.info("paper_orderbook_decision %s", json.dumps({
            "at_utc": current_time.astimezone(timezone.utc).isoformat(),
            "pair": pair, "side": side, "entry_tag": entry_tag,
            "source": MARKET_KEY, "coverage": coverage,
            "imbalance_3m": pressure, "contrary": contrary,
            "normal_stake": normal_stake, "chosen_stake": chosen_stake,
        }, sort_keys=True))
        return chosen_stake
