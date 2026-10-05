"""Reconstruct timestamp-safe G1-B4 exit actions, trade states, and controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from sieve3_event_freqai_round1 import (
    USER_DATA_DIR,
    atomic_json,
    now_iso,
    ohlcv_path,
    parse_pairs,
    parse_windows,
)
from sieve3_event_generation1_b1_freqai import DEFAULT_WINDOWS
from sieve3_event_generation1_cache import atomic_parquet
from sieve3_event_reaction_research import nearest_neighbor_candidates
from user_data.strategies.sieve3_event_reaction_targets import (
    EXIT_ROLE_HORIZONS,
    EXIT_ROLE_SEQUENCE_HORIZONS,
    _atr,
    _future_extreme,
    _future_extreme_step,
    _strict_before_label,
)


ROOT = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "sieve3_event_reaction"
)
DEFAULT_G0 = ROOT / "generation0" / "full_v2"
DEFAULT_OUTPUT = ROOT / "generation1" / "g1-b4"
DEFAULT_PAIRS = (
    "BTC/USDT:USDT,ETH/USDT:USDT,BNB/USDT:USDT,SOL/USDT:USDT,"
    "XRP/USDT:USDT,ADA/USDT:USDT,DOGE/USDT:USDT,TRX/USDT:USDT,"
    "AVAX/USDT:USDT,LINK/USDT:USDT"
)

ALL_G0_GROUPS = (
    ("entry_target_full_or_zone_reversal", "long"),
    ("entry_target_full_or_zone_reversal", "short"),
    ("profit_ladder_three_stage_no_ratchet", "long"),
    ("profit_ladder_three_stage_no_ratchet", "short"),
    ("profit_ladder_three_stage_ratchet", "long"),
    ("profit_ladder_three_stage_ratchet", "short"),
    ("profit_level_full_or_ladder", "long"),
    ("profit_level_full_or_ladder", "short"),
    ("target_partial_invalidation_remainder", "long"),
    ("target_partial_invalidation_remainder", "short"),
)
ELIGIBLE_GROUPS = ALL_G0_GROUPS
REASON_CATEGORIES = (
    "partial_stage_1",
    "partial_stage_2",
    "target_partial",
    "profit_target_full",
    "entry_target",
    "source_invalidation",
    "max_hold",
    "stop_loss",
    "trailing_stop_loss",
    "force_exit",
    "other",
)
MATCH_FEATURES = (
    "current_profit",
    "holding_hours_log1p",
    "favourable_excursion_to_state",
    "adverse_excursion_to_state",
    "target_stage_before",
    "remaining_fraction_before",
    "target_progress_risk_units",
    "original_trigger_active",
    "atr_pct",
)


def pair_token(pair: str) -> str:
    return pair.split("/")[0].split(":")[0].lower()


def feature_token(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def group_token(family: str, side: str) -> str:
    return f"{feature_token(family)}__{side}"


def path_id(row: Series) -> str:
    raw = "|".join(
        (
            str(row["source_backtest_path"]),
            str(row["pair"]),
            pd.Timestamp(row["open_date"]).isoformat(),
            str(row["entry_id"]),
        )
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def order_list(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def order_time(order: dict[str, Any]) -> pd.Timestamp:
    return pd.to_datetime(
        order.get("order_filled_timestamp"), unit="ms", utc=True, errors="coerce"
    )


def reason_category(tag: str) -> str:
    value = feature_token(tag)
    if "partial_stage_1" in value:
        return "partial_stage_1"
    if "partial_stage_2" in value:
        return "partial_stage_2"
    if "target_partial" in value or "partial" in value:
        return "target_partial"
    if "profit_target" in value and "full" in value:
        return "profit_target_full"
    if "entry_target" in value:
        return "entry_target"
    if "source_invalidation" in value or "invalidation" in value:
        return "source_invalidation"
    if "max_hold" in value:
        return "max_hold"
    if "trailing_stop" in value:
        return "trailing_stop_loss"
    if "stop_loss" in value or value == "stoploss":
        return "stop_loss"
    if "force_exit" in value:
        return "force_exit"
    return "other"


def stage_from_tag(tag: str) -> int:
    value = feature_token(tag)
    match = re.search(r"stage_(\d+)", value)
    if match:
        return int(match.group(1))
    if "profit_target_3" in value:
        return 3
    if "target_partial" in value or "entry_target" in value:
        return 1
    return 0


def load_strategy_parameters(backtest_path: str) -> tuple[float | None, float | None]:
    """Return configured hard-stop and terminal ladder target as ratios."""
    path = Path(backtest_path)
    if not path.is_file():
        return None, None
    try:
        with zipfile.ZipFile(path) as archive:
            for name in archive.namelist():
                if not name.endswith(".json") or name.endswith("_config.json"):
                    continue
                if archive.getinfo(name).file_size > 100_000:
                    continue
                payload = json.loads(archive.read(name).decode("utf-8"))
                if "strategy_name" not in payload or "params" not in payload:
                    continue
                sell = payload.get("params", {}).get("sell", {})
                hard = pd.to_numeric(sell.get("hard_stop_percent"), errors="coerce")
                target_1 = pd.to_numeric(sell.get("target_1_percent"), errors="coerce")
                gap_units = pd.to_numeric(
                    sell.get("target_gap_half_percent_units"), errors="coerce"
                )
                hard_ratio = float(hard) / 100.0 if pd.notna(hard) else None
                terminal = None
                if pd.notna(target_1) and pd.notna(gap_units):
                    terminal = (float(target_1) + float(gap_units)) / 100.0
                return hard_ratio, terminal
    except (OSError, KeyError, ValueError, zipfile.BadZipFile, json.JSONDecodeError):
        return None, None
    return None, None


def load_market(pair: str) -> DataFrame:
    frame = pd.read_feather(ohlcv_path(pair))
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = (
        frame.dropna(subset=["date"])
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )
    close = pd.to_numeric(frame["close"], errors="coerce").replace(0.0, np.nan)
    frame["atr_pct"] = _atr(frame) / close
    frame["decision_time"] = frame["date"] + pd.Timedelta(hours=1)
    for horizon in EXIT_ROLE_HORIZONS:
        frame[f"future_close_{horizon}h"] = close.shift(-horizon)
        frame[f"future_high_{horizon}h"] = _future_extreme(
            frame["high"], horizon, "max"
        )
        frame[f"future_low_{horizon}h"] = _future_extreme(
            frame["low"], horizon, "min"
        )
        frame[f"future_high_step_{horizon}h"] = _future_extreme_step(
            frame["high"], horizon, "max"
        )
        frame[f"future_low_step_{horizon}h"] = _future_extreme_step(
            frame["low"], horizon, "min"
        )
    return frame


def load_entry_health(cache_dir: Path, pair: str) -> DataFrame:
    path = cache_dir / f"{pair_token(pair)}_sieve3_events_1h.parquet"
    frame = pd.read_parquet(path)
    frame["decision_time"] = pd.to_datetime(
        frame["decision_time"], utc=True, errors="coerce"
    )
    return (
        frame.dropna(subset=["decision_time"])
        .drop_duplicates("decision_time", keep="last")
        .set_index("decision_time")
    )


def classify_window(
    values: Series, windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]]
) -> Series:
    result = pd.Series("outside", index=values.index, dtype="object")
    for name, (start, end) in windows.items():
        result.loc[values.ge(start) & values.lt(end)] = name
    return result


def extract_action_orders(trades: DataFrame) -> tuple[DataFrame, dict[str, int]]:
    rows: list[dict[str, Any]] = []
    no_exit_orders = 0
    ambiguous_same_hour = 0
    for _, trade in trades.iterrows():
        orders = order_list(trade.get("orders"))
        exits = [item for item in orders if not bool(item.get("ft_is_entry"))]
        if not exits:
            no_exit_orders += 1
            continue
        timed = [(order_time(item), item) for item in exits]
        timed = [(time, item) for time, item in timed if pd.notna(time)]
        timed.sort(key=lambda item: item[0])
        counts = pd.Series([time.floor("h") for time, _ in timed]).value_counts()
        ambiguous_same_hour += int(counts.gt(1).sum())
        entry_amount = sum(
            float(item.get("amount") or 0.0)
            for item in orders
            if bool(item.get("ft_is_entry"))
        )
        cumulative_exit = 0.0
        for index, (filled_at, order) in enumerate(timed):
            amount = float(order.get("amount") or 0.0)
            tag = str(order.get("ft_order_tag") or "")
            if not tag and index == len(timed) - 1:
                tag = str(trade.get("exit_reason") or "")
            before = max(entry_amount - cumulative_exit, 0.0)
            cumulative_exit += amount
            after = max(entry_amount - cumulative_exit, 0.0)
            event_raw = "|".join(
                (
                    str(trade["trade_path_id"]),
                    filled_at.isoformat(),
                    str(index),
                    tag,
                )
            )
            rows.append(
                {
                    "event_id": hashlib.sha1(event_raw.encode("utf-8")).hexdigest()[:24],
                    "trade_path_id": trade["trade_path_id"],
                    "pair": trade["pair"],
                    "entry_id": trade["entry_id"],
                    "entry_side": trade["entry_side"],
                    "exit_family": trade["exit_family"],
                    "group": group_token(trade["exit_family"], trade["entry_side"]),
                    "source_backtest_path": trade["source_backtest_path"],
                    "open_date": trade["open_date"],
                    "action_time": filled_at,
                    "source_candle_time": filled_at.floor("h"),
                    "decision_time": filled_at.floor("h") + pd.Timedelta(hours=1),
                    "action_price": pd.to_numeric(order.get("safe_price"), errors="coerce"),
                    "action_amount": amount,
                    "action_fraction_of_entry": amount / entry_amount if entry_amount else np.nan,
                    "remaining_fraction_before": before / entry_amount if entry_amount else np.nan,
                    "remaining_fraction_after": after / entry_amount if entry_amount else np.nan,
                    "action_tag": tag,
                    "reason_category": reason_category(tag),
                    "target_stage_after": max(stage_from_tag(tag), 0),
                    "is_final_action": bool(index == len(timed) - 1 or after <= 1e-10),
                    "same_hour_action_count": int(counts.loc[filled_at.floor("h")]),
                }
            )
    return DataFrame(rows), {
        "trades_without_exit_orders": no_exit_orders,
        "trade_hours_with_multiple_exit_orders": ambiguous_same_hour,
    }


def build_trade_states(
    trades: DataFrame,
    actions: DataFrame,
    market_by_pair: dict[str, DataFrame],
    entry_cache_dir: Path,
    windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
) -> tuple[DataFrame, dict[str, int]]:
    action_groups = {
        key: group.sort_values("action_time")
        for key, group in actions.groupby("trade_path_id", observed=True)
    }
    health_by_pair = {
        pair: load_entry_health(entry_cache_dir, pair) for pair in market_by_pair
    }
    frames: list[DataFrame] = []
    missing_market_trades = 0
    missing_trigger_columns = 0
    for _, trade in trades.iterrows():
        pair = str(trade["pair"])
        market = market_by_pair[pair]
        start = pd.Timestamp(trade["open_date"]).floor("h")
        end = pd.Timestamp(trade["close_date"]).floor("h")
        path_market = market[market["date"].ge(start) & market["date"].le(end)].copy()
        if path_market.empty:
            missing_market_trades += 1
            continue
        sign = -1.0 if str(trade["entry_side"]).lower() == "short" else 1.0
        open_rate = float(trade["open_rate"])
        high = pd.to_numeric(path_market["high"], errors="coerce")
        low = pd.to_numeric(path_market["low"], errors="coerce")
        close = pd.to_numeric(path_market["close"], errors="coerce")
        cumulative_high = high.cummax()
        cumulative_low = low.cummin()
        path_actions = action_groups.get(str(trade["trade_path_id"]), DataFrame())
        action_map = (
            {key: group for key, group in path_actions.groupby("source_candle_time")}
            if not path_actions.empty
            else {}
        )
        orders = order_list(trade.get("orders"))
        entry_orders = [item for item in orders if bool(item.get("ft_is_entry"))]
        exit_orders = [item for item in orders if not bool(item.get("ft_is_entry"))]
        total_entry_amount = sum(float(item.get("amount") or 0.0) for item in entry_orders)
        exit_order_times = [(order_time(item), item) for item in exit_orders]
        hard_stop = trade.get("configured_hard_stop_ratio")
        terminal_target = trade.get("configured_terminal_target_ratio")
        entry_id = str(trade["entry_id"])
        active_column = f"entry_active__{entry_id}"
        onset_column = f"entry_onset__{entry_id}"
        health = health_by_pair[pair]
        if active_column not in health or onset_column not in health:
            missing_trigger_columns += 1
        base_rows: list[dict[str, Any]] = []
        for local_index, market_row in path_market.reset_index(drop=True).iterrows():
            source_time = pd.Timestamp(market_row["date"])
            decision_time = source_time + pd.Timedelta(hours=1)
            earlier_exits = [
                item
                for time, item in exit_order_times
                if pd.notna(time) and time.floor("h") < source_time
            ]
            prior_exit_amount = sum(float(item.get("amount") or 0.0) for item in earlier_exits)
            remaining_before = (
                max(total_entry_amount - prior_exit_amount, 0.0) / total_entry_amount
                if total_entry_amount
                else np.nan
            )
            stage_before = max(
                [stage_from_tag(str(item.get("ft_order_tag") or "")) for item in earlier_exits]
                or [0]
            )
            action_rows = action_map.get(source_time)
            row_variants = (
                action_rows.to_dict("records") if action_rows is not None else [None]
            )
            for action in row_variants:
                is_action = action is not None
                price = (
                    float(action["action_price"])
                    if is_action and pd.notna(action["action_price"])
                    else float(market_row["close"])
                )
                if is_action:
                    prior_high = (
                        float(cumulative_high.iloc[local_index - 1])
                        if local_index > 0
                        else open_rate
                    )
                    prior_low = (
                        float(cumulative_low.iloc[local_index - 1])
                        if local_index > 0
                        else open_rate
                    )
                    state_high = max(prior_high, price)
                    state_low = min(prior_low, price)
                else:
                    state_high = float(cumulative_high.iloc[local_index])
                    state_low = float(cumulative_low.iloc[local_index])
                if sign > 0:
                    favourable = max(state_high / open_rate - 1.0, 0.0)
                    adverse = max(1.0 - state_low / open_rate, 0.0)
                else:
                    favourable = max(1.0 - state_low / open_rate, 0.0)
                    adverse = max(state_high / open_rate - 1.0, 0.0)
                current_profit = sign * (price / open_rate - 1.0)
                health_row = health.reindex([decision_time])
                trigger_active = (
                    pd.to_numeric(health_row[active_column], errors="coerce").iloc[0]
                    if active_column in health_row and not health_row.empty
                    else np.nan
                )
                trigger_onset = (
                    pd.to_numeric(health_row[onset_column], errors="coerce").iloc[0]
                    if onset_column in health_row and not health_row.empty
                    else np.nan
                )
                record = {
                    "state_id": f"{trade['trade_path_id']}|{decision_time.isoformat()}|{action['event_id'] if is_action else 'control'}",
                    "trade_path_id": trade["trade_path_id"],
                    "event_id": action["event_id"] if is_action else None,
                    "pair": pair,
                    "entry_id": entry_id,
                    "entry_side": trade["entry_side"],
                    "exit_family": trade["exit_family"],
                    "group": group_token(trade["exit_family"], trade["entry_side"]),
                    "source_backtest_path": trade["source_backtest_path"],
                    "open_date": trade["open_date"],
                    "source_candle_time": source_time,
                    "decision_time": decision_time,
                    "state_step": int(local_index),
                    "is_exit_action": bool(is_action),
                    "action_tag": action["action_tag"] if is_action else None,
                    "reason_category": action["reason_category"] if is_action else None,
                    "is_final_action": bool(action["is_final_action"]) if is_action else False,
                    "action_fraction_of_entry": action["action_fraction_of_entry"] if is_action else 0.0,
                    "current_price": price,
                    "open_rate": open_rate,
                    "current_profit": current_profit,
                    "holding_hours": max((source_time - start).total_seconds() / 3600.0, 0.0),
                    "holding_hours_log1p": math.log1p(
                        max((source_time - start).total_seconds() / 3600.0, 0.0)
                    ),
                    "favourable_excursion_to_state": favourable,
                    "adverse_excursion_to_state": adverse,
                    "target_stage_before": int(stage_before),
                    "remaining_fraction_before": (
                        action["remaining_fraction_before"]
                        if is_action
                        else remaining_before
                    ),
                    "target_progress_risk_units": (
                        current_profit / float(hard_stop)
                        if pd.notna(hard_stop) and float(hard_stop) > 0.0
                        else np.nan
                    ),
                    "configured_target_progress": (
                        current_profit / float(terminal_target)
                        if pd.notna(terminal_target) and float(terminal_target) > 0.0
                        else np.nan
                    ),
                    "configured_target_available": float(
                        pd.notna(terminal_target) and float(terminal_target or 0.0) > 0.0
                    ),
                    "original_trigger_active": trigger_active,
                    "original_trigger_onset": trigger_onset,
                    "atr_pct": market_row["atr_pct"],
                }
                base_rows.append(record)
        if base_rows:
            frames.append(DataFrame(base_rows))
    states = pd.concat(frames, ignore_index=True) if frames else DataFrame()
    if not states.empty:
        states["window"] = classify_window(states["decision_time"], windows)
        states = states[~states["window"].eq("outside")].copy()
    return states, {
        "trades_without_market_state": missing_market_trades,
        "trades_missing_original_trigger_columns": missing_trigger_columns,
    }


def match_open_trade_controls(states: DataFrame) -> tuple[DataFrame, dict[str, int]]:
    events = states[states["is_exit_action"]].copy()
    controls = states[
        ~states["is_exit_action"]
        & states["state_step"].mod(4).eq(0)
        & pd.to_numeric(states["remaining_fraction_before"], errors="coerce").gt(0.0)
    ].copy()
    action_pair_times = pd.MultiIndex.from_frame(
        events[["pair", "decision_time"]].drop_duplicates()
    )
    control_pair_times = pd.MultiIndex.from_frame(
        controls[["pair", "decision_time"]]
    )
    contaminated_controls = control_pair_times.isin(action_pair_times)
    controls = controls[~contaminated_controls].copy()
    matched: list[DataFrame] = []
    unmatched = 0
    reused = 0
    for keys, event_group in events.groupby(
        ["group", "pair", "window"], dropna=False, observed=True
    ):
        group, pair, window = keys
        pool = controls[
            controls["group"].eq(group)
            & controls["pair"].eq(pair)
            & controls["window"].eq(window)
        ].copy()
        if len(pool) < 20:
            unmatched += len(event_group)
            continue
        if len(pool) > 50_000:
            positions = np.linspace(0, len(pool) - 1, 50_000, dtype="int64")
            pool = pool.iloc[positions].copy()
        combined = pd.concat(
            [pool.loc[:, MATCH_FEATURES], event_group.loc[:, MATCH_FEATURES]],
            ignore_index=True,
        ).apply(pd.to_numeric, errors="coerce")
        medians = combined.median(axis=0)
        filled = combined.fillna(medians).fillna(0.0)
        scale = (filled.quantile(0.75) - filled.quantile(0.25)).replace(0.0, 1.0)
        standardized = (filled - medians.fillna(0.0)) / scale
        pool_values = standardized.iloc[: len(pool)].to_numpy(dtype="float64")
        event_values = standardized.iloc[len(pool) :].to_numpy(dtype="float64")
        neighbour_count = min(len(pool), 200)
        distances, indices = nearest_neighbor_candidates(
            pool_values, event_values, neighbour_count
        )
        pool = pool.reset_index(drop=True)
        event_group = event_group.reset_index(drop=True)
        used: set[int] = set()
        selected: list[int] = []
        selected_distance: list[float] = []
        selected_separation: list[float] = []
        selected_reused: list[bool] = []
        selected_event_ids: list[str] = []
        selected_event_reasons: list[str] = []
        selected_event_roles: list[str] = []
        for row_index, candidates in enumerate(indices):
            event = event_group.iloc[row_index]
            choice: tuple[int, int, float] | None = None
            for require_unused, require_separated in (
                (True, True),
                (True, False),
                (False, True),
                (False, False),
            ):
                for rank, candidate in enumerate(candidates):
                    position = int(candidate)
                    control = pool.iloc[position]
                    if control["trade_path_id"] == event["trade_path_id"]:
                        continue
                    separation = abs(
                        (
                            pd.Timestamp(control["decision_time"])
                            - pd.Timestamp(event["decision_time"])
                        ).total_seconds()
                    ) / 3600.0
                    if require_unused and position in used:
                        continue
                    if require_separated and separation <= 72.0:
                        continue
                    choice = (position, rank, separation)
                    break
                if choice is not None:
                    break
            if choice is None:
                unmatched += 1
                continue
            position, rank, separation = choice
            selected.append(position)
            selected_distance.append(float(distances[row_index, rank]))
            selected_separation.append(float(separation))
            selected_reused.append(position in used)
            selected_event_ids.append(str(event["event_id"]))
            selected_event_reasons.append(str(event["reason_category"]))
            selected_event_roles.append(
                "final" if bool(event["is_final_action"]) else "partial"
            )
            reused += int(position in used)
            used.add(position)
        if selected:
            result = pool.iloc[selected].copy().reset_index(drop=True)
            result["matched_event_id"] = selected_event_ids
            result["analysis_reason_category"] = selected_event_reasons
            result["analysis_action_role"] = selected_event_roles
            result["control_match_distance"] = selected_distance
            result["control_match_time_separation_h"] = selected_separation
            result["control_match_reused"] = selected_reused
            matched.append(result)
    output = pd.concat(matched, ignore_index=True) if matched else DataFrame()
    return output, {
        "event_rows": int(len(events)),
        "candidate_control_rows": int(len(controls)),
        "excluded_control_rows_at_any_exit_pair_time": int(
            contaminated_controls.sum()
        ),
        "matched_control_rows": int(len(output)),
        "unmatched_event_rows": int(unmatched),
        "reused_control_rows": int(reused),
    }


def attach_outcomes(
    samples: DataFrame, market_by_pair: dict[str, DataFrame], sample_kind: str
) -> DataFrame:
    frames: list[DataFrame] = []
    for pair, group in samples.groupby("pair", observed=True):
        market = market_by_pair[str(pair)].set_index("date")
        aligned = market.reindex(pd.DatetimeIndex(group["source_candle_time"]))
        aligned.index = group.index
        base = pd.to_numeric(group["current_price"], errors="coerce").replace(0.0, np.nan)
        sign = np.where(group["entry_side"].eq("short"), -1.0, 1.0)
        for horizon in EXIT_ROLE_HORIZONS:
            terminal = pd.to_numeric(
                aligned[f"future_close_{horizon}h"], errors="coerce"
            )
            future_high = pd.to_numeric(
                aligned[f"future_high_{horizon}h"], errors="coerce"
            )
            future_low = pd.to_numeric(
                aligned[f"future_low_{horizon}h"], errors="coerce"
            )
            long_mask = sign > 0.0
            best = future_high.where(long_mask, future_low)
            worst = future_low.where(long_mask, future_high)
            favourable = Series(sign, index=group.index) * (best / base - 1.0)
            adverse_signed = Series(sign, index=group.index) * (worst / base - 1.0)
            high_step = pd.to_numeric(
                aligned[f"future_high_step_{horizon}h"], errors="coerce"
            )
            low_step = pd.to_numeric(
                aligned[f"future_low_step_{horizon}h"], errors="coerce"
            )
            favourable_step = high_step.where(long_mask, low_step)
            adverse_step = low_step.where(long_mask, high_step)
            outcome = group.copy()
            outcome["sample_kind"] = sample_kind
            outcome["horizon_hours"] = int(horizon)
            outcome["hold_instead_delta"] = Series(sign, index=group.index) * (
                terminal / base - 1.0
            )
            outcome["missed_additional_profit"] = favourable.clip(lower=0.0)
            outcome["avoided_loss_after_exit"] = (-adverse_signed).clip(lower=0.0)
            outcome["net_exit_regret"] = (
                outcome["missed_additional_profit"]
                - outcome["avoided_loss_after_exit"]
            )
            outcome["favourable_peak_step"] = favourable_step
            outcome["adverse_peak_step"] = adverse_step
            outcome["favourable_before_adverse"] = _strict_before_label(
                favourable_step, adverse_step
            )
            frames.append(outcome)
    return pd.concat(frames, ignore_index=True) if frames else DataFrame()


DIRECT_METRICS = (
    "hold_instead_delta",
    "missed_additional_profit",
    "avoided_loss_after_exit",
    "net_exit_regret",
    "favourable_peak_step",
    "adverse_peak_step",
    "favourable_before_adverse",
)


def validate_path_order_labels(outcomes: DataFrame) -> dict[str, int]:
    """Audit that OHLCV path-order labels exist only for strict candle order."""
    favourable_step = pd.to_numeric(
        outcomes["favourable_peak_step"], errors="coerce"
    )
    adverse_step = pd.to_numeric(outcomes["adverse_peak_step"], errors="coerce")
    order_label = pd.to_numeric(
        outcomes["favourable_before_adverse"], errors="coerce"
    )
    complete_order = favourable_step.notna() & adverse_step.notna()
    incomplete_order = ~complete_order
    same_candle_tie = complete_order & favourable_step.eq(adverse_step)
    strict_order = complete_order & favourable_step.ne(adverse_step)
    audit = {
        "complete_peak_step_rows": int(complete_order.sum()),
        "same_candle_indeterminate_rows": int(same_candle_tie.sum()),
        "same_candle_rows_with_binary_label": int(
            order_label[same_candle_tie].notna().sum()
        ),
        "incomplete_order_rows_with_binary_label": int(
            order_label[incomplete_order].notna().sum()
        ),
        "strict_order_rows": int(strict_order.sum()),
        "strict_order_rows_without_binary_label": int(
            order_label[strict_order].isna().sum()
        ),
        "nonbinary_observed_labels": int(
            (~order_label.dropna().isin([0.0, 1.0])).sum()
        ),
    }
    if any(
        audit[key]
        for key in (
            "same_candle_rows_with_binary_label",
            "incomplete_order_rows_with_binary_label",
            "strict_order_rows_without_binary_label",
            "nonbinary_observed_labels",
        )
    ):
        raise ValueError(f"Invalid B4 path-order labels: {audit}")
    return audit


def direct_summary(outcomes: DataFrame) -> tuple[DataFrame, DataFrame]:
    rows: list[dict[str, Any]] = []
    keys = [
        "group",
        "exit_family",
        "entry_side",
        "analysis_reason_category",
        "analysis_action_role",
        "pair",
        "window",
        "horizon_hours",
    ]
    for group_keys, group in outcomes.groupby(keys, dropna=False, observed=True):
        row = dict(zip(keys, group_keys, strict=True))
        event = group[group["sample_kind"].eq("event")]
        control = group[group["sample_kind"].eq("matched_control")]
        row.update(
            {
                "event_rows": int(len(event)),
                "control_rows": int(len(control)),
                "unique_event_paths": int(event["trade_path_id"].nunique()),
                "unique_event_times": int(event["decision_time"].nunique()),
                "unique_control_paths": int(control["trade_path_id"].nunique()),
                "control_reused_share": float(
                    control.get("control_match_reused", Series(dtype=bool))
                    .fillna(False)
                    .mean()
                )
                if len(control)
                else np.nan,
            }
        )
        for metric in DIRECT_METRICS:
            event_numeric = pd.to_numeric(event[metric], errors="coerce")
            control_numeric = pd.to_numeric(control[metric], errors="coerce")
            event_valid = event[event_numeric.notna()]
            control_valid = control[control_numeric.notna()]
            event_values = event_numeric.dropna()
            control_values = control_numeric.dropna()
            event_mean = event_values.mean() if len(event_values) else np.nan
            control_mean = control_values.mean() if len(control_values) else np.nan
            row[f"event_{metric}_rows"] = int(len(event_valid))
            row[f"control_{metric}_rows"] = int(len(control_valid))
            row[f"unique_event_{metric}_paths"] = int(
                event_valid["trade_path_id"].nunique()
            )
            row[f"unique_event_{metric}_times"] = int(
                event_valid["decision_time"].nunique()
            )
            row[f"event_{metric}_mean"] = event_mean
            row[f"control_{metric}_mean"] = control_mean
            row[f"event_minus_control_{metric}"] = event_mean - control_mean
        rows.append(row)
    summary = DataFrame(rows)
    portable_rows: list[dict[str, Any]] = []
    for keys_value, group in summary.groupby(
        [
            "group",
            "exit_family",
            "entry_side",
            "analysis_reason_category",
            "analysis_action_role",
            "horizon_hours",
        ],
        dropna=False,
        observed=True,
    ):
        row = dict(
            zip(
                [
                    "group",
                    "exit_family",
                    "entry_side",
                    "analysis_reason_category",
                    "analysis_action_role",
                    "horizon_hours",
                ],
                keys_value,
                strict=True,
            )
        )
        adequate = group[
            group["unique_event_paths"].ge(20)
            & group["unique_event_times"].ge(20)
            & group["control_rows"].ge(20)
        ]
        row.update(
            {
                "pair_window_cells": int(len(group)),
                "adequate_cells": int(len(adequate)),
                "distinct_adequate_pairs": int(adequate["pair"].nunique()),
                "distinct_adequate_windows": int(adequate["window"].nunique()),
            }
        )
        for metric in DIRECT_METRICS:
            metric_adequate = group[
                group[f"unique_event_{metric}_paths"].ge(20)
                & group[f"unique_event_{metric}_times"].ge(20)
                & group[f"event_{metric}_rows"].ge(20)
                & group[f"control_{metric}_rows"].ge(20)
            ]
            row[f"{metric}_adequate_cells"] = int(len(metric_adequate))
            row[f"{metric}_distinct_adequate_pairs"] = int(
                metric_adequate["pair"].nunique()
            )
            row[f"{metric}_distinct_adequate_windows"] = int(
                metric_adequate["window"].nunique()
            )
            column = f"event_minus_control_{metric}"
            values = pd.to_numeric(metric_adequate[column], errors="coerce").dropna()
            row[f"median_{column}"] = values.median() if len(values) else np.nan
            row[f"positive_share_{column}"] = values.gt(0.0).mean() if len(values) else np.nan
        portable_rows.append(row)
    return summary, DataFrame(portable_rows)


def regenerate_direct_outputs(output: Path) -> dict[str, Any]:
    """Rebuild direct summaries from the persisted outcome table only."""
    outcomes_path = output / "matched_exit_controls.parquet"
    if not outcomes_path.is_file():
        raise FileNotFoundError(outcomes_path)
    outcomes = pd.read_parquet(outcomes_path)
    path_order_audit = validate_path_order_labels(outcomes)
    summary, portability = direct_summary(outcomes)
    summary.to_csv(output / "direct_exit_role_summary.csv", index=False)
    portability.to_csv(output / "direct_exit_role_portability.csv", index=False)

    sequence = portability[
        portability["horizon_hours"].isin(EXIT_ROLE_SEQUENCE_HORIZONS)
    ]
    metric_adequacy = {
        metric: {
            "adequate_cells": int(portability[f"{metric}_adequate_cells"].sum()),
            "distinct_rows_with_adequacy": int(
                portability[f"{metric}_adequate_cells"].gt(0).sum()
            ),
        }
        for metric in DIRECT_METRICS
    }
    metric_adequacy["favourable_before_adverse"].update(
        {
            "sequence_horizon_raw_adequate_cells": int(sequence["adequate_cells"].sum()),
            "sequence_horizon_metric_adequate_cells": int(
                sequence["favourable_before_adverse_adequate_cells"].sum()
            ),
        }
    )

    contract_path = output / "feature_contract.json"
    if contract_path.is_file():
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        contract["direct_metric_adequacy_rule"] = (
            "Every direct outcome is portable only in pair/window cells with at "
            "least 20 non-null event rows, 20 non-null control rows, 20 unique "
            "event paths and 20 unique event times for that exact outcome."
        )
        contract["direct_outputs_rebuilt_at"] = now_iso()
        atomic_json(contract_path, contract)

    audit_path = output / "cache_audit.json"
    if audit_path.is_file():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        audit["direct_summary_rows"] = int(len(summary))
        audit["direct_portability_rows"] = int(len(portability))
        audit["path_order_audit"] = path_order_audit
        audit["direct_metric_adequacy"] = metric_adequacy
        audit["direct_outputs_rebuilt_at"] = now_iso()
        atomic_json(audit_path, audit)

    result = {
        "direct_summary_rows": int(len(summary)),
        "direct_portability_rows": int(len(portability)),
        "path_order_audit": path_order_audit,
        "metric_adequacy": metric_adequacy,
    }
    return result


def aggregate_cache(
    pair: str,
    states: DataFrame,
    actions: DataFrame,
    market: DataFrame,
) -> DataFrame:
    cache = market[["decision_time"]].copy()
    cache["coverage_present"] = 1.0
    selected_groups = {group_token(*item) for item in ELIGIBLE_GROUPS}
    pair_states = states[
        states["pair"].eq(pair) & states["group"].isin(selected_groups)
    ].copy()
    unique_states = pair_states.sort_values("is_exit_action").drop_duplicates(
        ["trade_path_id", "decision_time"], keep="last"
    )
    state_metrics = {
        "open_count": ("trade_path_id", "nunique"),
        "current_profit_mean": ("current_profit", "mean"),
        "current_profit_median": ("current_profit", "median"),
        "current_profit_min": ("current_profit", "min"),
        "current_profit_max": ("current_profit", "max"),
        "holding_log_mean": ("holding_hours_log1p", "mean"),
        "holding_log_median": ("holding_hours_log1p", "median"),
        "holding_log_max": ("holding_hours_log1p", "max"),
        "favourable_excursion_mean": ("favourable_excursion_to_state", "mean"),
        "favourable_excursion_max": ("favourable_excursion_to_state", "max"),
        "adverse_excursion_mean": ("adverse_excursion_to_state", "mean"),
        "adverse_excursion_max": ("adverse_excursion_to_state", "max"),
        "target_stage_mean": ("target_stage_before", "mean"),
        "target_stage_max": ("target_stage_before", "max"),
        "remaining_fraction_mean": ("remaining_fraction_before", "mean"),
        "remaining_fraction_min": ("remaining_fraction_before", "min"),
        "risk_unit_progress_mean": ("target_progress_risk_units", "mean"),
        "configured_target_progress_mean": ("configured_target_progress", "mean"),
        "configured_target_available_share": ("configured_target_available", "mean"),
        "trigger_active_share": ("original_trigger_active", "mean"),
        "trigger_onset_share": ("original_trigger_onset", "mean"),
    }
    for side in ("long", "short"):
        work = unique_states[unique_states["entry_side"].eq(side)]
        if work.empty:
            for metric in state_metrics:
                cache[f"trade_state__{side}__{metric}"] = 0.0
        else:
            grouped = work.groupby("decision_time", observed=True).agg(**state_metrics)
            grouped = grouped.add_prefix(f"trade_state__{side}__").reset_index()
            cache = cache.merge(
                grouped, on="decision_time", how="left", validate="one_to_one"
            )
    pair_actions = actions[
        actions["pair"].eq(pair) & actions["group"].isin(selected_groups)
    ].copy()
    for family, side in ELIGIBLE_GROUPS:
        name = f"exit_event__{group_token(family, side)}"
        dates = pair_actions[
            pair_actions["exit_family"].eq(family)
            & pair_actions["entry_side"].eq(side)
        ]["decision_time"]
        cache[name] = cache["decision_time"].isin(set(dates)).astype(float)
    for side in ("long", "short"):
        side_actions = pair_actions[pair_actions["entry_side"].eq(side)]
        cache[f"exit_event_any__{side}"] = cache["decision_time"].isin(
            set(side_actions["decision_time"])
        ).astype(float)
        partial_dates = side_actions[
            side_actions["is_final_action"].eq(False)
        ]["decision_time"]
        full_dates = side_actions[
            side_actions["is_final_action"].eq(True)
        ]["decision_time"]
        cache[f"exit_event_partial__{side}"] = cache["decision_time"].isin(
            set(partial_dates)
        ).astype(float)
        cache[f"exit_event_final__{side}"] = cache["decision_time"].isin(
            set(full_dates)
        ).astype(float)
    for category in REASON_CATEGORIES:
        dates = pair_actions[pair_actions["reason_category"].eq(category)]["decision_time"]
        cache[f"exit_reason__{category}"] = cache["decision_time"].isin(
            set(dates)
        ).astype(float)
    counts = pair_actions.groupby("decision_time", observed=True).size()
    cache["exit_event_action_count"] = (
        cache["decision_time"].map(counts).fillna(0.0).astype(float)
    )
    return cache


def concentration_table(actions: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in actions.groupby(
        ["group", "exit_family", "entry_side", "pair", "window"],
        dropna=False,
        observed=True,
    ):
        row = dict(
            zip(
                ["group", "exit_family", "entry_side", "pair", "window"],
                keys,
                strict=True,
            )
        )
        entry_counts = group["entry_id"].value_counts(normalize=True)
        reason_counts = group["reason_category"].value_counts(normalize=True)
        row.update(
            {
                "action_rows": int(len(group)),
                "unique_trade_paths": int(group["trade_path_id"].nunique()),
                "unique_decision_times": int(group["decision_time"].nunique()),
                "unique_entry_ids": int(group["entry_id"].nunique()),
                "max_entry_id_share": float(entry_counts.iloc[0]) if len(entry_counts) else np.nan,
                "max_reason_share": float(reason_counts.iloc[0]) if len(reason_counts) else np.nan,
                "partial_action_share": float((~group["is_final_action"]).mean()),
                "same_hour_multi_action_share": float(group["same_hour_action_count"].gt(1).mean()),
            }
        )
        rows.append(row)
    return DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g0-dir", type=Path, default=DEFAULT_G0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--windows-json", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--pairs", default=DEFAULT_PAIRS)
    parser.add_argument("--direct-only", action="store_true")
    args = parser.parse_args()

    g0 = args.g0_dir.resolve()
    output = args.output_dir.resolve()
    cache_dir = output / "cache"
    output.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if args.direct_only:
        print(json.dumps(regenerate_direct_outputs(output), indent=2))
        return 0
    pairs = parse_pairs(args.pairs)
    windows = parse_windows(args.windows_json)

    trades = pd.read_parquet(g0 / "cache" / "exit_events_long.parquet")
    trades["open_date"] = pd.to_datetime(trades["open_date"], utc=True, errors="coerce")
    trades["close_date"] = pd.to_datetime(trades["close_date"], utc=True, errors="coerce")
    eligible = set(ALL_G0_GROUPS)
    trades = trades[
        trades["pair"].isin(pairs)
        & Series(
            [
                (family, side) in eligible
                for family, side in zip(trades["exit_family"], trades["entry_side"])
            ],
            index=trades.index,
        )
    ].copy()
    trades = trades.dropna(subset=["open_date", "close_date", "open_rate"])
    eligible_trade_rows_before_deduplication = int(len(trades))
    trades["trade_path_id"] = trades.apply(path_id, axis=1)
    duplicate_paths = int(trades.duplicated("trade_path_id", keep=False).sum())
    trades = trades.drop_duplicates("trade_path_id", keep="last").reset_index(drop=True)

    parameter_cache: dict[str, tuple[float | None, float | None]] = {}
    for source in trades["source_backtest_path"].drop_duplicates():
        parameter_cache[str(source)] = load_strategy_parameters(str(source))
    trades["configured_hard_stop_ratio"] = trades["source_backtest_path"].map(
        lambda value: parameter_cache[str(value)][0]
    )
    trades["configured_terminal_target_ratio"] = trades["source_backtest_path"].map(
        lambda value: parameter_cache[str(value)][1]
    )

    actions, order_audit = extract_action_orders(trades)
    actions["window"] = classify_window(actions["decision_time"], windows)
    actions = actions[~actions["window"].eq("outside")].copy()
    market_by_pair = {pair: load_market(pair) for pair in pairs}
    states, state_audit = build_trade_states(
        trades, actions, market_by_pair, g0 / "cache", windows
    )
    matched, match_audit = match_open_trade_controls(states)

    events = states[states["is_exit_action"]].copy()
    matched_events = set(matched.get("matched_event_id", Series(dtype=str)))
    events = events[events["event_id"].isin(matched_events)].copy()
    events["analysis_reason_category"] = events["reason_category"]
    events["analysis_action_role"] = np.where(
        events["is_final_action"], "final", "partial"
    )
    event_outcomes = attach_outcomes(events, market_by_pair, "event")
    control_outcomes = attach_outcomes(matched, market_by_pair, "matched_control")
    outcomes = pd.concat([event_outcomes, control_outcomes], ignore_index=True)
    path_order_audit = validate_path_order_labels(outcomes)
    summary, portability = direct_summary(outcomes)
    concentration = concentration_table(actions)

    feature_columns: list[str] = []
    for pair in pairs:
        cache = aggregate_cache(pair, states, actions, market_by_pair[pair])
        path = cache_dir / f"{pair_token(pair)}_sieve3_events_1h.parquet"
        atomic_parquet(cache, path)
        feature_columns = [
            column
            for column in cache.columns
            if column not in {"decision_time", "coverage_present"}
        ]

    selected_groups = {group_token(*item) for item in ELIGIBLE_GROUPS}
    selected_actions = actions[actions["group"].isin(selected_groups)].copy()
    selected_action_ids = set(selected_actions["event_id"])
    action_scopes = selected_actions[
        [
            "event_id",
            "trade_path_id",
            "pair",
            "decision_time",
            "source_candle_time",
            "entry_id",
            "entry_side",
            "exit_family",
            "group",
            "reason_category",
            "is_final_action",
            "window",
        ]
    ].rename(columns={"event_id": "scope_id"})
    action_scopes["matched_event_id"] = action_scopes["scope_id"]
    action_scopes["action_role"] = np.where(
        action_scopes["is_final_action"], "final", "partial"
    )
    action_scopes["scope_kind"] = "exact_action"
    selected_controls = matched[
        matched["matched_event_id"].isin(selected_action_ids)
    ].copy()
    control_scopes = selected_controls[
        [
            "state_id",
            "matched_event_id",
            "trade_path_id",
            "pair",
            "decision_time",
            "source_candle_time",
            "entry_id",
            "entry_side",
            "exit_family",
            "group",
            "analysis_reason_category",
            "analysis_action_role",
            "window",
        ]
    ].rename(
        columns={
            "state_id": "scope_id",
            "analysis_reason_category": "reason_category",
            "analysis_action_role": "action_role",
        }
    )
    control_scopes["is_final_action"] = control_scopes["action_role"].eq("final")
    control_scopes["scope_kind"] = "matched_open_state"
    scope_columns = [
        "scope_id",
        "matched_event_id",
        "trade_path_id",
        "pair",
        "decision_time",
        "source_candle_time",
        "entry_id",
        "entry_side",
        "exit_family",
        "group",
        "reason_category",
        "action_role",
        "is_final_action",
        "scope_kind",
        "window",
    ]
    freqai_scopes = pd.concat(
        [action_scopes[scope_columns], control_scopes[scope_columns]],
        ignore_index=True,
    )
    all_action_scopes = actions.copy()
    all_action_scopes["action_role"] = np.where(
        all_action_scopes["is_final_action"], "final", "partial"
    )
    all_action_scopes["scope_kind"] = "exact_action"
    all_action_scopes["matched_event_id"] = all_action_scopes["event_id"]
    all_action_scopes = all_action_scopes.rename(columns={"event_id": "scope_id"})
    atomic_parquet(freqai_scopes, output / "event_scopes.parquet")
    atomic_parquet(
        all_action_scopes[scope_columns], output / "all_exit_action_scopes.parquet"
    )
    atomic_parquet(states, output / "exit_trade_states.parquet")
    atomic_parquet(outcomes, output / "matched_exit_controls.parquet")
    concentration.to_csv(output / "unique_path_concentration.csv", index=False)
    summary.to_csv(output / "direct_exit_role_summary.csv", index=False)
    portability.to_csv(output / "direct_exit_role_portability.csv", index=False)

    identity_columns = [
        column
        for column in feature_columns
        if column.startswith("exit_event__")
        or column.startswith("exit_event_any__")
        or column.startswith("exit_event_partial__")
        or column.startswith("exit_event_final__")
        or column.startswith("exit_reason__")
        or column == "exit_event_action_count"
    ]
    state_columns = [
        column for column in feature_columns if column.startswith("trade_state__")
    ]
    contract = {
        "generated_at": now_iso(),
        "event_time_boundary": (
            "An executed order in candle T is represented at T+1h after that candle "
            "is complete; all path outcomes begin with the next complete candle."
        ),
        "freqai_target_base": (
            "FreqAI exit-role targets use the completed event-candle close. Direct "
            "counterfactuals use the exact executed order price."
        ),
        "path_order_rule": (
            "Favourable-before-adverse is binary only when peak steps occur in "
            "different future candles; same-candle OHLCV order is indeterminate."
        ),
        "identity_columns": identity_columns,
        "trade_state_columns": state_columns,
        "all_feature_columns": feature_columns,
        "eligible_groups": [group_token(*item) for item in ELIGIBLE_GROUPS],
        "reconstructed_g0_groups": [group_token(*item) for item in ALL_G0_GROUPS],
        "reason_categories": list(REASON_CATEGORIES),
        "match_features": list(MATCH_FEATURES),
    }
    atomic_json(output / "feature_contract.json", contract)
    audit = {
        "generated_at": now_iso(),
        "eligible_trade_rows_before_deduplication": eligible_trade_rows_before_deduplication,
        "duplicate_trade_path_rows": duplicate_paths,
        "unique_trade_paths": int(trades["trade_path_id"].nunique()),
        "executed_exit_action_rows": int(len(actions)),
        "selected_g1_b4_action_rows": int(len(selected_actions)),
        "selected_g1_b4_matched_control_scope_rows": int(len(control_scopes)),
        "selected_g1_b4_freqai_scope_rows": int(len(freqai_scopes)),
        "unique_exit_action_ids": int(actions["event_id"].nunique()),
        "partial_action_rows": int((~actions["is_final_action"]).sum()),
        "final_action_rows": int(actions["is_final_action"].sum()),
        "state_rows": int(len(states)),
        "state_pairs": int(states["pair"].nunique()),
        "nonfinite_state_cells": int(
            np.isinf(states.select_dtypes(include=[np.number]).to_numpy()).sum()
        ),
        "duplicate_state_ids": int(states["state_id"].duplicated().sum()),
        "pairs_without_actions": sorted(set(pairs) - set(actions["pair"])),
        "adequately_supported_action_pairs_20plus": sorted(
            actions["pair"].value_counts().loc[lambda values: values.ge(20)].index
        ),
        "parameter_sources": int(len(parameter_cache)),
        "parameter_sources_with_hard_stop": int(
            sum(value[0] is not None for value in parameter_cache.values())
        ),
        "parameter_sources_with_terminal_target": int(
            sum(value[1] is not None for value in parameter_cache.values())
        ),
        "order_audit": order_audit,
        "state_audit": state_audit,
        "match_audit": match_audit,
        "path_order_audit": path_order_audit,
        "cache_feature_columns": int(len(feature_columns)),
        "identity_feature_columns": int(len(identity_columns)),
        "trade_state_feature_columns": int(len(state_columns)),
        "direct_summary_rows": int(len(summary)),
        "direct_portability_rows": int(len(portability)),
    }
    atomic_json(output / "cache_audit.json", audit)
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
