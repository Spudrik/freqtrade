"""Four distinct, two-sided, fast PAPER experiments for liquid USDT perpetual alts.

Signals are ranked only at completed 15m boundaries.  The four families ask
different questions; exploratory best guesses stay active at technical-only 3x
when fresh main-agent context is absent.  Nothing here fetches news or submits
orders outside the ordinary Freqtrade/manual-controller paths.
"""
from __future__ import annotations

from datetime import datetime, timedelta
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
import talib.abstract as ta

from freqtrade.enums import RunMode
from freqtrade.exceptions import FreqtradeException
from freqtrade.persistence import Trade
from freqtrade.strategy import merge_informative_pair, stoploss_from_absolute
from user_data.Indicators.complex_volume_profile import add_volume_profile
from user_data.strategies.paper_trial_common import PaperTrialBase
from user_data.strategies.paper_aggressive_context import (
    AGGRESSIVE_ACCOUNTS, AGGRESSIVE_LIMITS, AGGRESSIVE_PAIRS,
    MANUAL_TAG_PREFIX, PLAN_KEY, account_key, aggressive_entry_plan,
    aggressive_fill_protection, apply_aggressive_protection_update,
    build_aggressive_filled_plan, load_aggressive_control, validate_aggressive_config,
    validate_aggressive_filled_plan, validate_aggressive_manual_plan,
)

LOG = logging.getLogger(__name__)
_TIMEFRAMES = ("15m", "1h", "4h", "1d")
_AUTO_ENTRY_PRICE_DEVIATION_ATR = 0.25
_MANUAL_ENTRY_PRICE_DEVIATION_ATR = 0.15


def _finite(value, label: str, *, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {label}") from error
    if not math.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"Invalid {label}")
    return result


def _margin_safe_entry_budget(equity: float, reference_rate: float, entry_atr: float,
                              *, manual: bool) -> tuple[float, float]:
    """Return the tolerated upper fill rate and callback stake that stays under the margin cap."""
    equity = _finite(equity, "paper equity", positive=True)
    reference_rate = _finite(reference_rate, "entry reference rate", positive=True)
    entry_atr = _finite(entry_atr, "entry ATR", positive=True)
    deviation_atr = (_MANUAL_ENTRY_PRICE_DEVIATION_ATR if manual
                     else _AUTO_ENTRY_PRICE_DEVIATION_ATR)
    upper_fill_rate = reference_rate + deviation_atr * entry_atr
    margin_cap = equity * AGGRESSIVE_LIMITS["max_margin_pct"]
    safe_stake = margin_cap * reference_rate / upper_fill_rate
    return upper_fill_rate, safe_stake


def rank_aggressive_candidates(candidates: list[dict], slots: int) -> list[str]:
    """Stable strongest-first selection; whitelist order never determines slots."""
    if type(slots) is not int or slots < 0:
        raise ValueError("Candidate rank needs a nonnegative available-slot count")
    normalized = []
    seen = set()
    for row in candidates:
        pair = row.get("pair")
        score = _finite(row.get("score"), "candidate score")
        if pair not in AGGRESSIVE_PAIRS or pair in seen or not 0 <= score <= 1:
            raise ValueError("Invalid or duplicated aggressive candidate")
        normalized.append((pair, score))
        seen.add(pair)
    return [pair for pair, _ in sorted(normalized, key=lambda item: (-item[1], item[0]))[:slots]]


class _PaperAggressiveAlt(PaperTrialBase):
    timeframe = "5m"
    startup_candle_count = 400
    stoploss = -0.25
    use_custom_stoploss = True
    use_exit_signal = True
    paper_hold_hours = 24
    account_key = "aggressive_vacuum"
    family = "vacuum"

    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._higher_cache: dict[tuple[str, str], tuple[tuple, DataFrame]] = {}
        self._cohort_cache: tuple[tuple, DataFrame] | None = None
        self._pending: dict[tuple[str, str, str], dict] = {}
        self._last_control_state: tuple | None = None

    def _account_name(self) -> str:
        return AGGRESSIVE_ACCOUNTS[account_key(self.account_key)]["bot_name"]

    def bot_start(self, **kwargs) -> None:
        if self.config.get("runmode") != RunMode.DRY_RUN:
            raise RuntimeError("Aggressive alternate accounts are PAPER-only")
        spec = AGGRESSIVE_ACCOUNTS[account_key(self.account_key)]
        root = Path(__file__).resolve().parents[2]
        base = json.loads((root / "user_data/configs/integrated_paper_20260926_base.json").read_text(encoding="utf-8"))
        overlay = json.loads((root / spec["config"]).read_text(encoding="utf-8"))
        validate_aggressive_config(base, overlay, self.account_key)
        api = self.config.get("api_server", {})
        exchange = self.config.get("exchange", {})
        if (self.config.get("dry_run") is not True or self.config.get("trading_mode") != "futures"
                or self.config.get("margin_mode") != "isolated" or exchange.get("name") != "binance"
                or any(exchange.get(k) for k in ("key", "secret", "password", "privateKey", "private_key"))
                or self.config.get("bot_name") != spec["bot_name"]
                or self.config.get("strategy") != self.__class__.__name__
                or self.config.get("db_url") != f"sqlite:///user_data/research_news_data/context_features/integrated_paper_20260926/{self.account_key}_trades.sqlite"
                or self.config.get("max_open_trades") != 3
                or set(exchange.get("pair_whitelist", ())) != AGGRESSIVE_PAIRS
                or self.config.get("force_entry_enable") is not True
                or api.get("enabled") is not True or api.get("listen_ip_address") != "127.0.0.1"
                or api.get("listen_port") != spec["port"]):
            raise RuntimeError("Aggressive alternate strategy requires its exact isolated local PAPER identity")
        super().bot_start(**kwargs)

    def informative_pairs(self):
        return sorted({(pair, tf) for pair in AGGRESSIVE_PAIRS for tf in _TIMEFRAMES}
                      | {(pair, "5m") for pair in ("BTC/USDT:USDT", "ETH/USDT:USDT")})

    @staticmethod
    def _signature(source: DataFrame) -> tuple:
        last = source.iloc[-1]
        return (len(source), *(last[name] for name in ("date", "open", "high", "low", "close", "volume")))

    def _higher(self, pair: str, timeframe: str) -> DataFrame:
        source = self.dp.get_pair_dataframe(pair=pair, timeframe=timeframe).copy()
        if source.empty:
            raise ValueError(f"Missing {timeframe} candles for {pair}")
        signature = self._signature(source)
        cached = self._higher_cache.get((pair, timeframe))
        if cached is not None and cached[0] == signature:
            return cached[1]
        if timeframe == "15m":
            atr = ta.ATR(source, timeperiod=14)
            prior_high = source["high"].shift(1).rolling(16, min_periods=16).max()
            prior_low = source["low"].shift(1).rolling(16, min_periods=16).min()
            prior_volume = source["volume"].shift(1).rolling(24, min_periods=12).median()
            candle_range = (source["high"] - source["low"]).replace(0, np.nan)
            pressure = ((source["close"] - source["open"]) / candle_range).clip(-1, 1)
            weighted_pressure = (pressure * source["volume"]).rolling(3, min_periods=2).sum()
            volume_sum = source["volume"].rolling(3, min_periods=2).sum().replace(0, np.nan)
            volume = source["volume"].copy()
            source = source[["date", "open", "high", "low", "close"]].copy()
            source["ag_atr"] = atr
            source["ag_prior_high"] = prior_high
            source["ag_prior_low"] = prior_low
            source["ag_relative_volume"] = volume.to_numpy() / prior_volume.to_numpy()
            source["ag_pressure"] = weighted_pressure.to_numpy() / volume_sum.to_numpy()
            source["ag_pressure_change"] = source["ag_pressure"] - source["ag_pressure"].shift(1)
            source["ag_range_expansion"] = (source["high"] - source["low"]) / source["ag_atr"]
            source["ag_return45"] = source["close"].pct_change(3)
        elif timeframe in {"1h", "4h"}:
            window = 36 if timeframe == "1h" else 24
            lookback = 24 if timeframe == "1h" else 12
            profile = add_volume_profile(source, window=window, bins=24, chunk_size=1024,
                                         prefix="ag_vp")
            source = source[["date", "high", "low", "close"]].copy()
            source["ag_range_high"] = profile["high"].shift(1).rolling(lookback, min_periods=lookback).max()
            source["ag_range_low"] = profile["low"].shift(1).rolling(lookback, min_periods=lookback).min()
            for output, name in (("ag_poc", "poc"), ("ag_vah", "vah"), ("ag_val", "val"),
                                 ("ag_lvn_above", "lvn_above"), ("ag_lvn_below", "lvn_below"),
                                 ("ag_thin_above", "lvn_above_thinness"), ("ag_thin_below", "lvn_below_thinness")):
                source[output] = profile[f"ag_vp_{name}"].shift(1)
            source["ag_value_width"] = source["ag_vah"] - source["ag_val"]
            source["ag_poc_migration"] = source["ag_poc"].pct_change(3)
        elif timeframe == "1d":
            source = source[["date", "high", "low", "close"]].copy()
            source["ag_day_high"] = source["high"].shift(1)
            source["ag_day_low"] = source["low"].shift(1)
            source["ag_day_range_high"] = source["high"].shift(1).rolling(20, min_periods=10).max()
            source["ag_day_range_low"] = source["low"].shift(1).rolling(20, min_periods=10).min()
        else:
            raise ValueError(f"Unsupported aggressive informative timeframe: {timeframe}")
        keep = ["date", *[name for name in source.columns if name.startswith("ag_")]]
        if timeframe == "15m":
            keep.extend(("open", "high", "low", "close"))
        output = source[keep].copy()
        self._higher_cache[(pair, timeframe)] = (signature, output)
        return output

    def _cohort_features(self) -> DataFrame:
        source_rows, signatures = [], []
        pairs = tuple(sorted(AGGRESSIVE_PAIRS))
        for pair in pairs:
            source = self.dp.get_pair_dataframe(pair=pair, timeframe="5m")
            if source.empty:
                return DataFrame(columns=["date", "pair", "ag_cohort_return", "ag_cohort_excess",
                                          "ag_cohort_rank", "ag_cohort_dispersion"])
            signatures.append((pair, self._signature(source)))
            source_rows.append(source[["date", "close"]].assign(**{pair: source["close"].pct_change(9)}).drop(columns="close"))
        signature = tuple(signatures)
        if self._cohort_cache is not None and self._cohort_cache[0] == signature:
            return self._cohort_cache[1]
        wide = source_rows[0]
        for row in source_rows[1:]:
            wide = wide.merge(row, on="date", how="outer", validate="one_to_one")
        returns = wide[list(pairs)]
        ranks = returns.rank(axis=1, method="average", pct=True)
        dispersion = returns.std(axis=1, ddof=0)
        complete = returns.notna().all(axis=1)
        dispersion = dispersion.where(complete)
        rows = []
        for pair in pairs:
            others = returns.drop(columns=pair)
            rows.append(DataFrame({"date": wide["date"], "pair": pair,
                "ag_cohort_return": returns[pair].where(complete),
                "ag_cohort_excess": (returns[pair] - others.median(axis=1)).where(complete),
                "ag_cohort_rank": ranks[pair].where(complete), "ag_cohort_dispersion": dispersion}))
        result = pd.concat(rows, ignore_index=True).sort_values(["pair", "date"], kind="stable")
        self._cohort_cache = (signature, result)
        return result

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        frame = dataframe.copy()
        frame["ag_atr5"] = ta.ATR(frame, timeperiod=14)
        frame["ag_prior_high5"] = frame["high"].shift(1).rolling(36, min_periods=24).max()
        frame["ag_prior_low5"] = frame["low"].shift(1).rolling(36, min_periods=24).min()
        for timeframe in _TIMEFRAMES:
            frame = merge_informative_pair(frame, self._higher(metadata["pair"], timeframe),
                                           "5m", timeframe, ffill=True)
        cohort = self._cohort_features()
        cohort = cohort.loc[cohort["pair"] == metadata["pair"]].drop(columns="pair")
        frame = frame.merge(cohort, on="date", how="left", validate="one_to_one")
        # BTC/ETH are context observations only; they never decide entry direction.
        for pair, prefix in (("BTC/USDT:USDT", "btc"), ("ETH/USDT:USDT", "eth")):
            leader = self.dp.get_pair_dataframe(pair=pair, timeframe="5m")[["date", "close"]].copy()
            leader[f"ag_{prefix}_return45"] = leader["close"].pct_change(9)
            frame = frame.merge(leader[["date", f"ag_{prefix}_return45"]], on="date", how="left",
                                validate="one_to_one")
        return frame

    def _signal_columns(self, frame: DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
        close = frame["close_15m"]
        high, low, opened = frame["high_15m"], frame["low_15m"], frame["open_15m"]
        atr = frame["ag_atr_15m"]
        pressure = frame["ag_pressure_15m"]
        relvol = frame["ag_relative_volume_15m"]
        day_high, day_low = frame["ag_day_range_high_1d"], frame["ag_day_range_low_1d"]
        daily_width = (day_high - day_low).replace(0, np.nan)
        daily_position = ((close - day_low) / daily_width).clip(0, 1)
        frame["ag_daily_position"] = daily_position
        valid = (close.notna() & atr.gt(0) & pressure.notna() & relvol.gt(0)
                 & frame["ag_atr5"].gt(0) & daily_position.notna())
        date = pd.to_datetime(frame["date"], utc=True)
        close_time = date + pd.Timedelta(minutes=5)
        cadence = close_time.dt.minute.mod(15).eq(0) & close_time.dt.second.eq(0)
        family = self.family
        route = pd.Series("best_guess", index=frame.index, dtype="object")
        reason = pd.Series("family-specific exploratory best guess", index=frame.index, dtype="object")

        if family == "vacuum":
            vah, val = frame["ag_vah_1h"], frame["ag_val_1h"]
            valid &= vah.notna() & val.notna()
            pressure_scaled = pressure.clip(-1, 1)
            thin_l = frame["ag_thin_above_1h"].fillna(0.0).clip(0, 1)
            thin_s = frame["ag_thin_below_1h"].fillna(0.0).clip(0, 1)
            above, below = close.gt(vah), close.lt(val)
            primary_l = above & pressure.gt(.22) & thin_l.ge(.5) & relvol.ge(.9)
            primary_s = below & pressure.lt(-.22) & thin_s.ge(.5) & relvol.ge(.9)
            loc = ((close - val) / (vah - val).replace(0, np.nan)).clip(0, 1).fillna(.5)
            score_l = (.34 + .20 * pressure_scaled.clip(lower=0) + .24 * thin_l + .22 * loc).clip(0, 1)
            score_s = (.34 + .20 * (-pressure_scaled).clip(lower=0) + .24 * thin_s + .22 * (1 - loc)).clip(0, 1)
            # Vacuum edges take priority, but non-ideal thinness stays exploratory.
            score_l = score_l.where(~above | pressure.le(0), score_l + .05)
            score_s = score_s.where(~below | pressure.ge(0), score_s + .05)
            reason[:] = "best-guess from local profile pressure, thinness and value location"
            reason[primary_l | primary_s] = "thin-profile edge acceptance with signed pressure"
            route[primary_l | primary_s] = "profile_acceptance"
        elif family == "reclaim":
            prior_high, prior_low = frame["ag_prior_high_15m"], frame["ag_prior_low_15m"]
            valid &= prior_high.notna() & prior_low.notna()
            sweep_l = (prior_low - low) / atr
            sweep_s = (high - prior_high) / atr
            swept_low = low.lt(prior_low - .10 * atr)
            swept_high = high.gt(prior_high + .10 * atr)
            reclaimed = close.gt(prior_low) & close.gt(opened)
            rejected = close.lt(prior_high) & close.lt(opened)
            change = frame["ag_pressure_change_15m"].fillna(0.0).clip(-1, 1)
            primary_l = swept_low & reclaimed & change.ge(.22) & sweep_l.ge(.18)
            primary_s = swept_high & rejected & change.le(-.22) & sweep_s.ge(.18)
            loc = ((close - prior_low) / (prior_high - prior_low).replace(0, np.nan)).clip(0, 1).fillna(.5)
            score_l = (.34 + .30 * (1 - loc) + .26 * change.clip(lower=0)).clip(0, 1)
            score_s = (.34 + .30 * loc + .26 * (-change).clip(lower=0)).clip(0, 1)
            score_l += (.08 * (swept_low & reclaimed)).astype(float)
            score_s += (.08 * (swept_high & rejected)).astype(float)
            reason[:] = "best-guess from failed-extreme proximity and pressure change"
            reason[primary_l | primary_s] = "failed extreme reclaimed with changing candle pressure"
            route[primary_l | primary_s] = "sweep_reclaim"
        elif family == "auction":
            vah, val, poc = frame["ag_vah_4h"], frame["ag_val_4h"], frame["ag_poc_4h"]
            valid &= vah.notna() & val.notna() & poc.notna()
            previous_close = frame["close_15m"].shift(1)
            relative_volume = relvol
            accepted_long = close.gt(vah + .1 * atr) & previous_close.gt(vah)
            accepted_short = close.lt(val - .1 * atr) & previous_close.lt(val)
            expansion = frame["ag_range_expansion_15m"].fillna(0.0)
            escape_l = accepted_long & (relative_volume.ge(1.1) | expansion.ge(1.0))
            escape_s = accepted_short & (relative_volume.ge(1.1) | expansion.ge(1.0))
            value_loc = ((close - val) / (vah - val).replace(0, np.nan)).clip(-.5, 1.5).fillna(.5)
            value_width = ((vah - val) / atr).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            reject_l = low.le(val + .15 * atr) & close.gt(val) & close.gt(opened)
            reject_s = high.ge(vah - .15 * atr) & close.lt(vah) & close.lt(opened)
            migration = frame["ag_poc_migration_4h"].fillna(0.0).clip(-.02, .02) / .02
            poc_gap = ((close - poc) / (vah - val).replace(0, np.nan)).clip(-1, 1).fillna(0.0)
            edge_l = (1 - value_loc).clip(0, 1)
            edge_s = value_loc.clip(0, 1)
            score_l = (.38 + .32 * (-poc_gap).clip(lower=0) + .30 * migration.clip(lower=0)).clip(0, 1)
            score_s = (.38 + .32 * poc_gap.clip(lower=0) + .30 * (-migration).clip(lower=0)).clip(0, 1)
            score_l = score_l.where(~reject_l, .46 + .30 * edge_l + .24 * value_width.clip(0, 1))
            score_s = score_s.where(~reject_s, .46 + .30 * edge_s + .24 * value_width.clip(0, 1))
            score_l = score_l.where(~escape_l, .52 + .22 * (relative_volume / 1.5).clip(0, 1)
                                    + .26 * (expansion / 1.5).clip(0, 1))
            score_s = score_s.where(~escape_s, .52 + .22 * (relative_volume / 1.5).clip(0, 1)
                                    + .26 * (expansion / 1.5).clip(0, 1))
            primary_l = escape_l & relative_volume.ge(1.1) & expansion.ge(1.0)
            primary_s = escape_s & relative_volume.ge(1.1) & expansion.ge(1.0)
            primary_l |= reject_l & edge_l.ge(.62) & value_width.ge(.6) & ~escape_l
            primary_s |= reject_s & edge_s.ge(.62) & value_width.ge(.6) & ~escape_s
            route[escape_l | escape_s] = "accepted_value_escape"
            route[(reject_l | reject_s) & ~(escape_l | escape_s)] = "value_edge_rotation"
            reason[:] = "best-guess from value-area location and POC migration"
            reason[reject_l | reject_s] = "value-edge rejection rotates toward prior point of control"
            reason[escape_l | escape_s] = "accepted value-area escape; rotation thesis invalidated"
        elif family == "rotation":
            valid &= (frame["ag_cohort_excess"].notna() & frame["ag_cohort_rank"].notna()
                      & frame["ag_cohort_dispersion"].notna() & frame["ag_cohort_return"].notna())
            excess = frame["ag_cohort_excess"].fillna(0.0)
            rank = frame["ag_cohort_rank"].fillna(.5)
            dispersion = frame["ag_cohort_dispersion"].fillna(0.0)
            own_return = frame["ag_cohort_return"].fillna(0.0)
            magnitude = (excess.abs() / dispersion.clip(lower=1e-6)).clip(0, 1)
            score_l = (.42 + .34 * magnitude + .24 * ((rank - .5).abs() * 2)).clip(0, 1)
            score_s = score_l.copy()
            primary_l = (excess.gt(0) & excess.ge(pd.concat([dispersion * .2,
                       pd.Series(.0008, index=frame.index)], axis=1).max(axis=1)) & rank.ge(.78))
            primary_s = (excess.lt(0) & (-excess).ge(pd.concat([dispersion * .2,
                       pd.Series(.0008, index=frame.index)], axis=1).max(axis=1)) & rank.le(.22))
            # Continuous cross-sectional best guess: strongest excess for long,
            # weakest excess for short; own return only breaks a neutral tie.
            long_preferred = excess.gt(0) | (excess.eq(0) & own_return.ge(0))
            score_l += (.01 * long_preferred).astype(float)
            score_s += (.01 * ~long_preferred).astype(float)
            reason[:] = "best-guess from same-close cohort excess/rank and own 45m return"
            reason[primary_l | primary_s] = "causal 45m return excess versus the eight-coin cohort"
            route[primary_l | primary_s] = "cohort_leader_laggard"
        else:
            raise ValueError(f"Unknown aggressive family: {family}")

        # Daily range location is a low-weight background feature; the distinct
        # family geometry remains the dominant direction/score driver.
        daily_tilt = (daily_position.fillna(.5) - .5) * .05
        score_l += daily_tilt
        score_s -= daily_tilt
        score_l = pd.Series(score_l, index=frame.index).astype(float).clip(0, 1)
        score_s = pd.Series(score_s, index=frame.index).astype(float).clip(0, 1)
        primary_l = pd.Series(primary_l, index=frame.index).fillna(False).astype(bool)
        primary_s = pd.Series(primary_s, index=frame.index).fillna(False).astype(bool)
        long_route = valid & cadence & (score_l >= score_s)
        short_route = valid & cadence & (score_s > score_l)
        long_mode = pd.Series(np.where(primary_l, "primary", "exploratory"), index=frame.index)
        short_mode = pd.Series(np.where(primary_s, "primary", "exploratory"), index=frame.index)
        return long_route, short_route, score_l.where(long_route, score_s.where(short_route)), \
            long_mode.where(long_route, short_mode.where(short_route)), route, reason

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        frame = dataframe
        frame["enter_long"], frame["enter_short"], frame["enter_tag"] = 0, 0, None
        long_route, short_route, score, mode, route, reason = self._signal_columns(frame)
        frame["ag_score"] = score
        frame["ag_mode"] = mode
        frame["ag_route"] = route
        frame["ag_decision_side"] = None
        frame["ag_reason"] = ""
        if long_route.any():
            frame.loc[long_route, "enter_long"] = 1
            frame.loc[long_route, "ag_decision_side"] = "long"
            frame.loc[long_route, "ag_reason"] = reason[long_route]
        if short_route.any():
            frame.loc[short_route, "enter_short"] = 1
            frame.loc[short_route, "ag_decision_side"] = "short"
            frame.loc[short_route, "ag_reason"] = reason[short_route]
        for side, mask in (("long", long_route), ("short", short_route)):
            for level in ("primary", "exploratory"):
                selected = mask & mode.eq(level)
                frame.loc[selected, "enter_tag"] = frame.loc[selected, "ag_route"].map(
                    lambda route_name: f"aggressive:{self.family}:{level}:{route_name}:{side}")
        frame["ag_signal"] = long_route | short_route
        return frame

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"], dataframe["exit_short"], dataframe["exit_tag"] = 0, 0, None
        return dataframe

    def _control(self, now: datetime):
        return load_aggressive_control(self.account_key, now)

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        control = self._control(current_time)
        state = (control.status, control.decision_id, control.bias, control.side_permission,
                 control.long_leverage_cap, control.short_leverage_cap, control.exposure)
        if state != self._last_control_state:
            LOG.info("aggressive_control account=%s status=%s decision=%s technical_only=%s bias=%s side=%s caps=(%s,%s) exposure=%s reason=%s",
                     self.account_key, control.status, control.decision_id, control.technical_only,
                     control.bias, control.side_permission, control.long_leverage_cap,
                     control.short_leverage_cap, control.exposure, control.reason)
            self._last_control_state = state

    def _current_row(self, pair: str, now: datetime, *, require_signal: bool = True):
        frame, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if frame.empty:
            raise ValueError(f"No analyzed aggressive candle for {pair}")
        row = frame.iloc[-1]
        closed_at = pd.Timestamp(row["date"]).to_pydatetime() + timedelta(minutes=5)
        if now.tzinfo is None or now.utcoffset() is None or not timedelta(0) <= now - closed_at < timedelta(minutes=10):
            raise ValueError(f"Aggressive candle is stale, future-dated, or has no aware clock for {pair}")
        if require_signal and not bool(row.get("ag_signal", False)):
            raise ValueError(f"No active completed 15m {self.family} decision for {pair}")
        return row

    @staticmethod
    def _row_value(row, name: str, *, positive: bool = False) -> float:
        return _finite(row[name], name, positive=positive)

    def _features(self, row) -> dict:
        names = ("ag_score", "ag_atr5", "ag_atr_15m", "ag_pressure_15m", "ag_pressure_change_15m",
                 "ag_relative_volume_15m", "ag_range_expansion_15m", "ag_vah_1h", "ag_val_1h",
                 "ag_thin_above_1h", "ag_thin_below_1h", "ag_prior_high_15m", "ag_prior_low_15m",
                 "ag_poc_4h", "ag_vah_4h", "ag_val_4h", "ag_poc_migration_4h",
                 "ag_range_high_1h", "ag_range_low_1h", "ag_range_high_4h", "ag_range_low_4h",
                 "ag_day_range_high_1d", "ag_day_range_low_1d", "ag_cohort_return", "ag_cohort_excess",
                 "ag_cohort_rank", "ag_cohort_dispersion", "ag_btc_return45", "ag_eth_return45",
                 "ag_daily_position")
        result = {}
        for name in names:
            value = row.get(name)
            if value is None or pd.isna(value):
                continue
            result[name.removeprefix("ag_")] = float(value)
        return result

    def _geometry(self, row, side: str, rate: float) -> tuple[float, float, float, str]:
        atr15 = self._row_value(row, "ag_atr_15m", positive=True)
        route = str(row["ag_route"])
        if self.family == "vacuum":
            anchor = self._row_value(row, "ag_vah_1h" if side == "long" else "ag_val_1h", positive=True)
            target_names = (("ag_lvn_above_1h", "1h_thin_node"), ("ag_range_high_1h", "1h_range_high"),
                            ("ag_lvn_above_4h", "4h_thin_node"), ("ag_vah_4h", "4h_value_high"),
                            ("ag_range_high_4h", "4h_range_high"), ("ag_day_range_high_1d", "prior_day_high")) if side == "long" else (
                            ("ag_lvn_below_1h", "1h_thin_node"), ("ag_range_low_1h", "1h_range_low"),
                            ("ag_lvn_below_4h", "4h_thin_node"), ("ag_val_4h", "4h_value_low"),
                            ("ag_range_low_4h", "4h_range_low"), ("ag_day_range_low_1d", "prior_day_low"))
        elif self.family == "reclaim":
            swept = route == "sweep_reclaim"
            anchor = self._row_value(row, ("low_15m" if swept else "ag_prior_low_15m") if side == "long"
                                     else ("high_15m" if swept else "ag_prior_high_15m"), positive=True)
            target_names = (("ag_prior_high_15m", "reclaimed_range_high"), ("ag_vah_1h", "1h_value_high"),
                            ("ag_range_high_1h", "1h_range_high"), ("ag_vah_4h", "4h_value_high"),
                            ("ag_range_high_4h", "4h_range_high")) if side == "long" else (
                            ("ag_prior_low_15m", "reclaimed_range_low"), ("ag_val_1h", "1h_value_low"),
                            ("ag_range_low_1h", "1h_range_low"), ("ag_val_4h", "4h_value_low"),
                            ("ag_range_low_4h", "4h_range_low"))
        elif self.family == "auction":
            if route == "accepted_value_escape":
                anchor = self._row_value(row, "ag_vah_4h" if side == "long" else "ag_val_4h", positive=True)
                target_names = (("ag_range_high_1h", "1h_range_high"), ("ag_range_high_4h", "4h_range_high"),
                                ("ag_day_range_high_1d", "prior_day_high")) if side == "long" else (
                                ("ag_range_low_1h", "1h_range_low"), ("ag_range_low_4h", "4h_range_low"),
                                ("ag_day_range_low_1d", "prior_day_low"))
            else:
                anchor = self._row_value(row, "ag_val_4h" if side == "long" else "ag_vah_4h", positive=True)
                target_names = (("ag_poc_4h", "4h_point_of_control"), ("ag_vah_4h", "4h_value_high"),
                                ("ag_range_high_4h", "4h_range_high")) if side == "long" else (
                                ("ag_poc_4h", "4h_point_of_control"), ("ag_val_4h", "4h_value_low"),
                                ("ag_range_low_4h", "4h_range_low"))
        else:
            anchor = self._row_value(row, "ag_prior_low5" if side == "long" else "ag_prior_high5", positive=True)
            target_names = (("ag_prior_high5", "5m_range_high"), ("ag_range_high_1h", "1h_range_high"),
                            ("ag_vah_4h", "4h_value_high"), ("ag_range_high_4h", "4h_range_high")) if side == "long" else (
                            ("ag_prior_low5", "5m_range_low"), ("ag_range_low_1h", "1h_range_low"),
                            ("ag_val_4h", "4h_value_low"), ("ag_range_low_4h", "4h_range_low"))
        direction = 1.0 if side == "long" else -1.0
        structural_distance = (rate - anchor) if side == "long" else (anchor - rate)
        distance = max(.75 * atr15, max(0.0, structural_distance) + .75 * atr15)
        distance = max(distance, 2.2 * AGGRESSIVE_LIMITS["fee_allowance"] * rate)
        stop = rate - direction * distance
        minimum_target_distance = max(1.5 * distance, 2.5 * AGGRESSIVE_LIMITS["fee_allowance"] * rate)
        candidates = []
        for name, label in target_names:
            value = row.get(name)
            if value is None or pd.isna(value):
                continue
            level = float(value)
            travel = direction * (level - rate)
            if level > 0 and travel >= minimum_target_distance:
                candidates.append((travel, level, label))
        if candidates:
            _, target, label = min(candidates)
            target_kind = f"next_observed_{label}"
        else:
            target = rate + direction * max(1.8 * distance, 2.5 * AGGRESSIVE_LIMITS["fee_allowance"] * rate)
            target_kind = "labelled_volatility_objective_1_8R"
        return stop, target, anchor, target_kind

    def _context_size_factor(self, control, side: str, row) -> float:
        sign = 1.0 if side == "long" else -1.0
        news = 1.0
        if not control.technical_only:
            news += .10 * sign * control.bias
        leaders = [row.get("ag_btc_return45"), row.get("ag_eth_return45")]
        leaders = [float(value) for value in leaders if value is not None and pd.notna(value)]
        if leaders:
            # BTC/ETH inform portfolio risk, but do not veto the alt's own route.
            leader_alignment = sum(math.tanh(value / .01) for value in leaders) / len(leaders)
            news += .15 * sign * leader_alignment
        return min(1.30, max(.65, news)) * control.size_factor

    def _automatic_leverage(self, row, side: str, rate: float, control, max_leverage: float) -> float:
        desired = control.desired_leverage(side, str(row["ag_mode"]), float(row["ag_score"]))
        if desired < 3.0:
            return 0.0
        stop, _, _, _ = self._geometry(row, side, rate)
        unit_distance = abs(rate - stop) / rate + AGGRESSIVE_LIMITS["fee_allowance"]
        stop_cap = AGGRESSIVE_LIMITS["emergency_margin_loss_pct"] / unit_distance
        if min(desired, max_leverage, stop_cap) < 3.0:
            return 0.0
        return min(desired, max_leverage, stop_cap, control.leverage_cap(side))

    def _automatic_stake_values(self, *, row, side: str, rate: float, control,
                                equity: float, committed: float, leverage: float,
                                min_stake: float | None, max_stake: float) -> dict | None:
        """Share the automatic callback's geometry, risk, and sizing rules with ranking."""
        rate = _finite(rate, "automatic candidate rate", positive=True)
        score = _finite(row["ag_score"], "automatic candidate score")
        stop, target, anchor, target_kind = self._geometry(row, side, rate)
        direction = 1.0 if side == "long" else -1.0
        if ((side == "long" and not stop < rate < target)
                or (side == "short" and not target < rate < stop)):
            return None
        fee = AGGRESSIVE_LIMITS["fee_allowance"]
        if direction * (target - rate) / rate < 2.5 * fee - 1e-12:
            return None
        unit_risk = leverage * (abs(rate - stop) / rate + fee)
        if unit_risk > AGGRESSIVE_LIMITS["emergency_margin_loss_pct"] + 1e-12:
            return None
        available_risk = min(equity * AGGRESSIVE_LIMITS["max_position_risk_pct"],
                             equity * AGGRESSIVE_LIMITS["max_combined_risk_pct"] - committed)
        if available_risk <= 0:
            return None
        context_factor = self._context_size_factor(control, side, row)
        _, margin_safe_stake = _margin_safe_entry_budget(
            equity, rate, row["ag_atr5"], manual=False)
        requested_stake = min(equity * AGGRESSIVE_LIMITS["max_margin_pct"] * context_factor,
                              _finite(max_stake, "maximum candidate stake", positive=True))
        stake = min(requested_stake, available_risk / (unit_risk * 1.10),
                    margin_safe_stake * control.size_factor)
        planned_risk = stake * unit_risk
        risk_reserve = planned_risk * 1.10
        if (not math.isfinite(stake) or stake <= 0 or risk_reserve <= 0
                or risk_reserve > equity * AGGRESSIVE_LIMITS["max_position_risk_pct"] + 1e-8
                or committed + risk_reserve > equity * AGGRESSIVE_LIMITS["max_combined_risk_pct"] + 1e-8
                or (min_stake is not None and stake < float(min_stake))):
            return None
        return {"stop": stop, "target": target, "anchor": anchor, "target_kind": target_kind,
                "stake": stake, "planned_risk": planned_risk, "risk_reserve": risk_reserve,
                "context_factor": context_factor, "score": score}

    def _committed_risk(self, equity: float, now: datetime, *, exclude_pending=None,
                        confirmed_only: bool = False) -> float:
        committed = self._open_risk_reserve(equity)
        for key, item in self._pending.items():
            if key == exclude_pending:
                continue
            if not timedelta(0) <= now - item["at"] <= timedelta(seconds=30):
                continue
            if confirmed_only and not item.get("confirmed"):
                continue
            committed += _finite(item["risk_reserve_usdt"], "pending risk reservation", positive=True)
        return committed

    def _failure_exit_reason(self, row, plan: dict, side: str) -> str | None:
        close = float(row["close_15m"])
        pressure = float(row["ag_pressure_15m"])
        route = str(plan["route"])
        family = plan["family"]
        if family == "vacuum":
            edge = float(row["ag_vah_1h"] if side == "long" else row["ag_val_1h"])
            failed = close <= edge and pressure < -.20 if side == "long" else close >= edge and pressure > .20
            return "vacuum_profile_acceptance_failed" if failed else None
        if family == "reclaim":
            level = float(plan["entry_reclaim_boundary"])
            failed = close <= level if side == "long" else close >= level
            return "reclaim_entry_boundary_lost" if failed else None
        if family == "auction":
            if route == "accepted_value_escape":
                edge = float(row["ag_vah_4h"] if side == "long" else row["ag_val_4h"])
                failed = close <= edge if side == "long" else close >= edge
                return "auction_accepted_escape_failed" if failed else None
            edge = float(row["ag_val_4h"] if side == "long" else row["ag_vah_4h"])
            failed = close <= edge if side == "long" else close >= edge
            return "auction_value_edge_rotation_failed" if failed else None
        excess, rank = float(row["ag_cohort_excess"]), float(row["ag_cohort_rank"])
        failed = excess < 0 and rank < .40 if side == "long" else excess > 0 and rank > .60
        return "rotation_relative_advantage_lost" if failed else None

    def _open_risk_reserve(self, equity: float) -> float:
        total = 0.0
        for trade in Trade.get_open_trades():
            if trade.strategy != self.__class__.__name__:
                raise RuntimeError("Foreign strategy position in isolated aggressive account")
            plan = trade.get_custom_data(PLAN_KEY)
            validate_aggressive_filled_plan(
                plan, actual_pair=trade.pair, actual_side="short" if trade.is_short else "long",
                actual_open_rate=float(trade.open_rate), actual_quantity=float(trade.amount),
                actual_stake=float(trade.stake_amount), actual_leverage=float(trade.leverage))
            total += float(plan["risk_reserve_usdt"])
        return total

    def leverage(self, pair, current_time, current_rate, proposed_leverage,
                 max_leverage, entry_tag, side, **kwargs):
        try:
            control = self._control(current_time)
            maximum = _finite(max_leverage, "exchange leverage maximum", positive=True)
            if isinstance(entry_tag, str) and entry_tag.startswith(MANUAL_TAG_PREFIX):
                plan = aggressive_entry_plan(entry_tag, current_time, self.account_key)
                validate_aggressive_manual_plan(plan, current_rate, self.account_key, control=control, now=current_time)
                requested = _finite(plan["leverage"], "approved leverage", positive=True)
            else:
                row = self._current_row(pair, current_time)
                return self._automatic_leverage(
                    row, side, float(current_rate), control, maximum)
            if requested < 3.0 or maximum < 3.0:
                return 0.0
            return min(requested, maximum, control.leverage_cap(side))
        except Exception as error:
            LOG.error("aggressive_leverage_refused account=%s pair=%s side=%s error=%s", self.account_key, pair, side, error)
            return 0.0

    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake,
                            min_stake, max_stake, leverage, entry_tag, side, **kwargs):
        try:
            pending_key = (pair, side, str(entry_tag))
            self._pending.pop(pending_key, None)
            self._pending = {key: value for key, value in self._pending.items()
                             if timedelta(0) <= current_time - value["at"] <= timedelta(seconds=30)}
            control = self._control(current_time)
            manual = isinstance(entry_tag, str) and entry_tag.startswith(MANUAL_TAG_PREFIX)
            row = self._current_row(pair, current_time, require_signal=not manual)
            equity = _finite(self.wallets.get_total_stake_amount(), "paper equity", positive=True)
            leverage = _finite(leverage, "callback leverage", positive=True)
            if not 3.0 <= leverage <= 10.0:
                return 0.0
            if manual:
                approval = aggressive_entry_plan(entry_tag, current_time, self.account_key)
                validate_aggressive_manual_plan(approval, current_rate, self.account_key,
                    equity=equity, actual_leverage=leverage, control=control, now=current_time)
                if (approval["pair"] != pair or approval["side"] != side
                        or not math.isclose(float(approval["leverage"]), leverage, abs_tol=1e-8)):
                    return 0.0
                stop, target = float(approval["stop_price"]), float(approval["take_profit_price"])
                anchor = stop
                target_kind = "main_agent_approved"
                requested_stake = equity * float(approval["stake_pct"])
                mode = "manual"
                score = 1.0
                reason = approval["reason"]
                features = {"decision_id": approval["decision_id"], "source": approval["source"],
                            **self._features(row)}
                deviation = abs(float(current_rate) - float(approval["reference_rate"]))
                if deviation > max(.15 * float(row["ag_atr5"]), .001 * float(current_rate)):
                    return 0.0
            else:
                if (str(row["ag_decision_side"]) != side or not str(entry_tag).startswith(f"aggressive:{self.family}:")):
                    return 0.0
                if not control.permits(side):
                    return 0.0
                mode = str(row["ag_mode"])
                if control.desired_leverage(side, mode, float(row["ag_score"])) < leverage - 1e-8:
                    return 0.0
                score = float(row["ag_score"])
                reason = str(row["ag_reason"])
                features = self._features(row)
            if manual:
                if ((side == "long" and not stop < current_rate < target)
                        or (side == "short" and not target < current_rate < stop)):
                    return 0.0
                unit_risk = leverage * (abs(float(current_rate) - stop) / float(current_rate)
                                        + AGGRESSIVE_LIMITS["fee_allowance"])
                if unit_risk > AGGRESSIVE_LIMITS["emergency_margin_loss_pct"]:
                    return 0.0
                committed = self._committed_risk(equity, current_time)
                available_risk = min(equity * AGGRESSIVE_LIMITS["max_position_risk_pct"],
                                     equity * AGGRESSIVE_LIMITS["max_combined_risk_pct"] - committed)
                _, margin_safe_stake = _margin_safe_entry_budget(
                    equity, float(current_rate), row["ag_atr5"], manual=True)
                requested_stake = min(equity * float(approval["stake_pct"]), float(max_stake),
                                      margin_safe_stake)
                stake = requested_stake
                planned_risk = stake * unit_risk
                risk_reserve = planned_risk * 1.10
                if planned_risk > available_risk / 1.10 + 1e-8:
                    return 0.0
            else:
                committed = self._committed_risk(equity, current_time)
                sized = self._automatic_stake_values(
                    row=row, side=side, rate=float(current_rate), control=control, equity=equity,
                    committed=committed, leverage=leverage, min_stake=min_stake, max_stake=max_stake)
                if sized is None:
                    return 0.0
                stop, target, anchor = sized["stop"], sized["target"], sized["anchor"]
                target_kind = sized["target_kind"]
                stake, planned_risk, risk_reserve = sized["stake"], sized["planned_risk"], sized["risk_reserve"]
                score = sized["score"]
                features = {**features, "context_size_factor": sized["context_factor"]}
            if (not math.isfinite(stake) or stake <= 0 or risk_reserve <= 0
                    or risk_reserve > equity * AGGRESSIVE_LIMITS["max_position_risk_pct"] + 1e-8
                    or committed + risk_reserve > equity * AGGRESSIVE_LIMITS["max_combined_risk_pct"] + 1e-8
                    or (min_stake is not None and stake < float(min_stake))):
                return 0.0
            reclaim_boundary = None
            if self.family == "reclaim":
                reclaim_boundary = self._row_value(
                    row, "ag_prior_low_15m" if side == "long" else "ag_prior_high_15m", positive=True)
            pending = {"at": current_time, "family": self.family, "mode": mode,
                "route": str(row["ag_route"]), "pair": pair, "side": side,
                "entry_tag": str(entry_tag), "provenance": "main_agent_force_entry" if manual else "automatic_signal",
                "reason": reason, "score": score, "features": features,
                "reference_rate": float(current_rate), "entry_atr": float(row["ag_atr5"]),
                "atr15": float(row["ag_atr_15m"]), "structural_anchor": float(anchor),
                "stop_price": float(stop), "target_price": float(target), "target_kind": target_kind,
                "planned_loss_usdt": planned_risk, "risk_reserve_usdt": risk_reserve,
                "entry_equity_usdt": equity, "stake_usdt": stake,
                "requested_leverage": leverage, "leverage": leverage,
                "control_status": control.status, "control_decision_id": control.decision_id,
                "technical_only": control.technical_only, "control_bias": control.bias,
                "control_side_permission": control.side_permission,
                "control_exposure": control.exposure, "score_at_entry": score}
            if reclaim_boundary is not None:
                pending["entry_reclaim_boundary"] = reclaim_boundary
            self._pending[pending_key] = pending
            LOG.info("aggressive_entry_plan account=%s pair=%s side=%s mode=%s score=%.3f route=%s control=%s technical_only=%s leverage=%.1f stake=%.2f reserve=%.2f reason=%s features=%s",
                     self.account_key, pair, side, mode, score, pending["route"], control.status,
                     control.technical_only, leverage, stake, risk_reserve, reason,
                     json.dumps(features, sort_keys=True, allow_nan=False))
            return stake
        except Exception as error:
            LOG.error("aggressive_stake_refused account=%s pair=%s side=%s error=%s", self.account_key, pair, side, error)
            self._pending.pop((pair, side, str(entry_tag)), None)
            return 0.0

    def _candidate_rank_feasible(self, pair: str, row, now: datetime, control) -> bool:
        try:
            side = row.get("ag_decision_side")
            if side not in {"long", "short"} or not control.permits(side):
                return False
            score = _finite(row.get("ag_score"), "candidate score")
            if not 0.0 <= score <= 1.0:
                return False
            mode = row.get("ag_mode")
            if mode not in {"primary", "exploratory"}:
                return False
            rate = _finite(row.get("close"), "candidate cached close", positive=True)
            exchange = getattr(self.dp, "_exchange", None)
            if exchange is None:
                return False
            proposed_stake = _finite(
                self.wallets.get_trade_stake_amount(pair, int(self.config["max_open_trades"]), update=False),
                "candidate proposed stake", positive=True)
            max_leverage = _finite(exchange.get_max_leverage(pair, proposed_stake),
                                   "candidate exchange leverage maximum", positive=True)
            leverage = self._automatic_leverage(row, side, rate, control, max_leverage)
            if leverage < 3.0:
                return False
            min_stake = exchange.get_min_pair_stake_amount(pair, rate, self.stoploss, leverage)
            max_stake = _finite(exchange.get_max_pair_stake_amount(pair, rate, leverage),
                                "candidate maximum stake", positive=True)
            max_stake = min(max_stake, _finite(self.wallets.get_available_stake_amount(),
                                               "available stake", positive=True))
            equity = _finite(self.wallets.get_total_stake_amount(), "paper equity", positive=True)
            committed = self._committed_risk(equity, now, confirmed_only=True)
            return self._automatic_stake_values(
                row=row, side=side, rate=rate, control=control, equity=equity,
                committed=committed, leverage=leverage, min_stake=min_stake,
                max_stake=max_stake) is not None
        except (FreqtradeException, KeyError, TypeError, ValueError, RuntimeError):
            return False

    def _top_rank_allows(self, pair: str, now: datetime, *, pending: dict | None = None,
                         rate: float | None = None, amount: float | None = None) -> bool:
        trades = Trade.get_open_trades()
        occupied = {trade.pair for trade in trades}
        if pair in occupied:
            return False
        free = max(0, int(self.config["max_open_trades"]) - len(trades))
        if free <= 0:
            return False
        own, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if own.empty:
            return False
        current_date = pd.Timestamp(own.iloc[-1]["date"])
        control = self._control(now)
        candidates = []
        for candidate_pair in sorted(AGGRESSIVE_PAIRS):
            if candidate_pair in occupied:
                continue
            analyzed, _ = self.dp.get_analyzed_dataframe(candidate_pair, self.timeframe)
            if analyzed.empty:
                continue
            row = analyzed.iloc[-1]
            signal = row.get("ag_signal", False)
            if pd.isna(signal) or not bool(signal) or pd.isna(row.get("ag_score")):
                continue
            candle_date = pd.Timestamp(row["date"])
            closed_at = candle_date.to_pydatetime() + timedelta(minutes=5)
            if not timedelta(0) <= now - closed_at < timedelta(minutes=10):
                continue
            if candle_date != current_date:
                continue
            side = str(row["ag_decision_side"])
            if candidate_pair == pair and pending is not None:
                if (side != pending.get("side") or pending.get("provenance") != "automatic_signal"
                        or rate is None or amount is None):
                    continue
                try:
                    aggressive_fill_protection(
                        side=side, rate=rate, approved_stop=pending["stop_price"],
                        approved_target=pending["target_price"], notional=float(amount) * float(rate),
                        leverage=pending["requested_leverage"],
                        risk_reserve=pending["risk_reserve_usdt"])
                except (TypeError, ValueError):
                    continue
                score = _finite(pending.get("score"), "pending candidate score")
            elif (not isinstance(row.get("enter_tag"), str)
                    or not row["enter_tag"].startswith(f"aggressive:{self.family}:")
                    or not self._candidate_rank_feasible(candidate_pair, row, now, control)):
                continue
            else:
                score = _finite(row.get("ag_score"), "candidate score")
            candidates.append({"pair": candidate_pair, "score": score})
        ranked = rank_aggressive_candidates(candidates, free)
        return pair in ranked

    def confirm_trade_entry(self, pair, order_type, amount, rate, time_in_force,
                            current_time, entry_tag, side, **kwargs) -> bool:
        try:
            key = (pair, side, str(entry_tag))
            pending = self._pending.get(key)
            if (pending is None or pending.get("confirmed")
                    or not timedelta(0) <= current_time - pending["at"] <= timedelta(seconds=30)):
                return False
            control = self._control(current_time)
            manual = pending["provenance"] == "main_agent_force_entry"
            if (control.decision_id != pending["control_decision_id"]
                    or (not manual and not control.permits(side))
                    or pending["requested_leverage"] > control.leverage_cap(side)):
                return False
            if control.technical_only and pending["requested_leverage"] > 3.0:
                return False
            upper_fill_rate, _ = _margin_safe_entry_budget(
                pending["entry_equity_usdt"], pending["reference_rate"],
                pending["entry_atr"], manual=manual)
            maximum_deviation = upper_fill_rate - pending["reference_rate"]
            rate = _finite(rate, "entry confirmation price", positive=True)
            amount = _finite(amount, "entry confirmation amount", positive=True)
            if abs(rate - pending["reference_rate"]) > maximum_deviation:
                return False
            upper_margin = amount * upper_fill_rate / pending["requested_leverage"]
            if upper_margin > pending["entry_equity_usdt"] * AGGRESSIVE_LIMITS["max_margin_pct"] + 1e-5:
                return False
            try:
                aggressive_fill_protection(
                    side=side, rate=rate, approved_stop=pending["stop_price"],
                    approved_target=pending["target_price"], notional=amount * rate,
                    leverage=pending["requested_leverage"], risk_reserve=pending["risk_reserve_usdt"])
            except ValueError:
                return False
            if not manual and not self._top_rank_allows(
                    pair, current_time, pending=pending, rate=rate, amount=amount):
                return False
            pending["confirmed"] = True
            return True
        except Exception as error:
            LOG.error("aggressive_entry_confirmation_refused account=%s pair=%s side=%s error=%s",
                      self.account_key, pair, side, error)
            return False

    def order_filled(self, pair, trade, order, current_time, **kwargs):
        if order.ft_order_side != trade.entry_side or trade.get_custom_data(PLAN_KEY) is not None:
            return
        side = "short" if trade.is_short else "long"
        key = (pair, side, str(trade.enter_tag))
        pending = self._pending.pop(key, None)
        try:
            if pending is None or not pending.get("confirmed"):
                raise RuntimeError("Aggressive fill lacks a confirmed durable risk reservation")
            contract_size = _finite(trade.contract_size, "actual contract size", positive=True)
            plan = build_aggressive_filled_plan(pending, pair=pair, side=side,
                open_rate=float(trade.open_rate), quantity=float(trade.amount),
                stake=float(trade.stake_amount), leverage=float(trade.leverage),
                contract_size=contract_size, filled_at=current_time)
            trade.set_custom_data(key=PLAN_KEY, value=plan)
        except Exception as error:
            LOG.critical("aggressive_entry_fill_protection_fault account=%s trade_id=%s pair=%s side=%s error=%s follow_up=emergency_stop_and_exit",
                         self.account_key, trade.id, pair, side, error)
            raise
        LOG.info("aggressive_entry_filled account=%s trade_id=%s pair=%s side=%s actual_rate=%s quantity=%s leverage=%s stop=%s target=%s planned_risk=%s reserve=%s provenance=%s",
                 self.account_key, trade.id, pair, side, plan["open_rate"], plan["quantity"],
                 plan["leverage"], plan["stop_price"], plan["target_price"],
                 plan["planned_loss_usdt"], plan["risk_reserve_usdt"], plan["provenance"])

    def _trade_plan(self, trade) -> dict:
        plan = trade.get_custom_data(PLAN_KEY)
        validate_aggressive_filled_plan(
            plan, actual_pair=trade.pair, actual_side="short" if trade.is_short else "long",
            actual_open_rate=float(trade.open_rate), actual_quantity=float(trade.amount),
            actual_stake=float(trade.stake_amount), actual_leverage=float(trade.leverage))
        if plan["family"] != self.family:
            raise ValueError("Aggressive durable protection plan belongs to another family")
        if not math.isclose(float(plan["contract_size"]), float(trade.contract_size), rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("Aggressive durable contract size differs from the open position")
        updated = apply_aggressive_protection_update(
            plan, self.account_key, int(trade.id), actual_pair=trade.pair,
            actual_side="short" if trade.is_short else "long", actual_open_rate=float(trade.open_rate),
            actual_quantity=float(trade.amount), actual_stake=float(trade.stake_amount),
            actual_leverage=float(trade.leverage))
        if updated != plan:
            trade.set_custom_data(key=PLAN_KEY, value=updated)
        return updated

    def _profit_protection(self, plan: dict, current_rate: float) -> tuple[dict, bool]:
        rate = _finite(current_rate, "current rate", positive=True)
        side = plan["side"]
        direction = 1.0 if side == "long" else -1.0
        entry = float(plan["open_rate"])
        unit_risk = abs(entry - float(plan["initial_stop_price"]))
        favorable = direction * (rate - entry)
        if unit_risk <= 0 or favorable < unit_risk:
            return plan, False
        cost_floor = 2.2 * AGGRESSIVE_LIMITS["fee_allowance"] * entry
        locked_gain = max(.75 * unit_risk, cost_floor) if favorable >= 2.0 * unit_risk else max(.25 * unit_risk, cost_floor)
        candidate = entry + direction * locked_gain
        previous = float(plan["stop_price"])
        tightened = max(previous, candidate) if side == "long" else min(previous, candidate)
        if ((side == "long" and (tightened <= previous or tightened >= rate or tightened >= float(plan["target_price"])))
                or (side == "short" and (tightened >= previous or tightened <= rate or tightened <= float(plan["target_price"])) )):
            return plan, False
        updated = dict(plan, stop_price=tightened,
                       profit_protection="cost_adjusted_0_25R" if favorable < 2.0 * unit_risk else "cost_adjusted_0_75R")
        validate_aggressive_filled_plan(updated, actual_pair=plan["pair"], actual_side=side,
            actual_open_rate=entry, actual_quantity=float(plan["quantity"]),
            actual_stake=float(plan["stake_usdt"]), actual_leverage=float(plan["leverage"]))
        return updated, True

    def custom_stoploss(self, pair, trade, current_time, current_rate,
                        current_profit, after_fill, **kwargs):
        try:
            plan = self._trade_plan(trade)
        except Exception as error:
            LOG.critical("aggressive_protection_missing account=%s trade_id=%s error=%s emergency_margin_loss_pct=%s",
                         self.account_key, trade.id, error, AGGRESSIVE_LIMITS["emergency_margin_loss_pct"])
            emergency_distance = AGGRESSIVE_LIMITS["emergency_margin_loss_pct"] / float(trade.leverage)
            stop = float(trade.open_rate) * (1.0 + emergency_distance if trade.is_short else 1.0 - emergency_distance)
            return stoploss_from_absolute(stop, float(current_rate), is_short=trade.is_short, leverage=trade.leverage)
        try:
            protected, changed = self._profit_protection(plan, current_rate)
            if changed:
                trade.set_custom_data(key=PLAN_KEY, value=protected)
                plan = protected
                LOG.info("aggressive_profit_protection account=%s trade_id=%s side=%s stop=%s lock=%s",
                         self.account_key, trade.id, plan["side"], plan["stop_price"], plan["profit_protection"])
        except Exception as error:
            # Keep the last durable stop; the explicit emergency stop remains
            # active if the monotonic profit-lock write cannot be completed.
            LOG.error("aggressive_profit_protection_write_failed account=%s trade_id=%s error=%s",
                      self.account_key, trade.id, error)
        return stoploss_from_absolute(float(plan["stop_price"]), float(current_rate),
                                      is_short=trade.is_short, leverage=trade.leverage)

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        try:
            plan = self._trade_plan(trade)
        except Exception as error:
            LOG.critical("aggressive_invalid_plan_emergency_exit account=%s trade_id=%s error=%s",
                         self.account_key, trade.id, error)
            return "aggressive_missing_or_invalid_protection_emergency_exit"
        age = current_time - trade.open_date_utc
        if age >= timedelta(hours=self.paper_hold_hours):
            return "aggressive_time_limit_24h"
        target = float(plan["target_price"])
        if ((trade.is_short and current_rate <= target)
                or (not trade.is_short and current_rate >= target)):
            return "aggressive_target"
        if age >= timedelta(minutes=15) and plan["provenance"] == "automatic_signal":
            try:
                row = self._current_row(pair, current_time, require_signal=False)
                close_at = pd.Timestamp(row["date"]).to_pydatetime() + timedelta(minutes=5)
                if close_at.minute % 15 == 0:
                    failure = self._failure_exit_reason(row, plan, "short" if trade.is_short else "long")
                    if failure:
                        return failure
            except (KeyError, TypeError, ValueError, RuntimeError):
                # Family data may be temporarily unavailable; the frozen
                # stop, target and 24h limit remain independent protection.
                pass
        return None


class PaperAggressiveVacuum(_PaperAggressiveAlt):
    account_key = "aggressive_vacuum"
    family = "vacuum"


class PaperAggressiveReclaim(_PaperAggressiveAlt):
    account_key = "aggressive_reclaim"
    family = "reclaim"


class PaperAggressiveAuction(_PaperAggressiveAlt):
    account_key = "aggressive_auction"
    family = "auction"


class PaperAggressiveRotation(_PaperAggressiveAlt):
    account_key = "aggressive_rotation"
    family = "rotation"
