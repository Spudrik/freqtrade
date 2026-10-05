"""Build outcome-blind support for the frozen fresh-confirmation batch.

The command is deliberately gated by the complete-OHLCV coverage result.  Before
that gate passes it writes only a small waiting record on C: and never creates the
bulky D: support directory.  When coverage is complete it freezes all three
routes together before any future reaction values may be opened.
"""

from __future__ import annotations

# Bound native numerical pools before importing numpy/pandas.
# ruff: noqa: E402
import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYARROW_NUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_fresh_confirmation_freeze as freshz,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation20_state_attribution as g20s,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation22_cross_asset_context as g22d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation22_freqai_cache as g22cache,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_anchored_vwap as g24v,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_round_distribution_convergence as g24r,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation25_convergence_representation as g25conv,
)


ANALYSIS_PATH = Path(__file__).resolve()
DEFAULT_SUPPORT_ID = "fresh_confirmation_support_20260903a"
RECORD_ROOT = freshz.OUTPUT_ROOT / "support"
ARTIFACT_ROOT = Path(
    "D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones/fresh_confirmation"
)
OUTSIDE_PERIOD = "outside_fresh_confirmation"
ZONE_HALF_WIDTH_ATR = 0.25
COOLDOWN_HOURS = 6
MIN_ROWS_PER_PAIR_CONTROL_PERIOD = 10
MIN_ELIGIBLE_COINS = 5
MIN_TRAINING_ROWS = 720
MIN_PREDICTION_ROWS = 120
TARGET_PURGE_HOURS = freshz.MAXIMUM_OUTCOME_HOURS
STATE_READY_BLOCK = g22cache.READY_BLOCK
STATE_FEATURE_COLUMNS = tuple(g22cache.FEATURE_COLUMNS)
STATE_MODEL_FEATURE_COLUMNS = tuple(g22cache.STATE_FEATURES)
VWAP_LEVEL_NAME = "1d__current_session_through_previous_completed_candle__centre_0"
VWAP_SCOPE_VALUE = "1d__current_session_through_previous_completed_candle__centre_0.0"
VWAP_CONTROLS = ("actual", *g24v.CONTROLS)
CONVERGENCE_CONTROLS = ("actual", *g25conv.CONTROLS)


def artifact(path: Path) -> dict[str, Any]:
    """Return the standard immutable-artifact description."""
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


def support_result_path(support_id: str) -> Path:
    return RECORD_ROOT / support_id / "fresh_confirmation_support_result.json"


def coverage_result_path(coverage_run_id: str) -> Path:
    return (
        freshz.COVERAGE_ROOT
        / coverage_run_id
        / "fresh_confirmation_coverage_result.json"
    )


def load_inputs(
    coverage_run_id: str, *, refresh_coverage: bool
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify the pre-outcome freeze and obtain the timestamp-only coverage gate."""
    frozen = freshz.freeze()
    if refresh_coverage:
        coverage = freshz.run_coverage(coverage_run_id, overwrite=True)
    else:
        path = coverage_result_path(coverage_run_id)
        coverage = json.loads(path.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_fresh_confirmation_outcomes":
        raise ValueError("Fresh-confirmation question freeze is invalid.")
    if coverage.get("batch_id") != frozen.get("batch_id"):
        raise ValueError("Coverage result belongs to a different frozen batch.")
    if coverage.get("future_reaction_outcomes_read") is not False:
        raise ValueError("Coverage result is not outcome blind.")
    freeze_item = coverage.get("artifacts", {}).get("freeze", {})
    if freeze_item.get("sha256") != g0.sha256_file(freshz.FREEZE_PATH):
        raise ValueError("Coverage result does not match the frozen questions.")
    ready = coverage.get("status") == "ready_to_freeze_fresh_event_support"
    if ready != bool(coverage.get("coverage", {}).get("all_pair_coverage_pass")):
        raise ValueError("Coverage status and all-pair gate disagree.")
    validate_frozen_support_contract(frozen)
    return frozen, coverage


def validate_frozen_support_contract(frozen: dict[str, Any]) -> None:
    """Reject any drift between the question freeze and reused parent equations."""
    siblings = {str(item["id"]): item for item in frozen["siblings"]}
    if set(siblings) != {
        "fresh_market_state_activity",
        "fresh_meme_convergence_crossing",
        "fresh_daily_current_session_vwap",
    }:
        raise ValueError("The frozen three-route support batch changed.")
    state = siblings["fresh_market_state_activity"]
    state_profiles = [
        item for item in state["profiles"] if item["role"] == "market_state_only"
    ]
    shuffled_profiles = [
        item
        for item in state["profiles"]
        if item["role"] == "within_pair_time_shuffled_training_labels"
    ]
    if not state_profiles or not shuffled_profiles:
        raise ValueError("The frozen state/model control ladder is incomplete.")
    if any(
        tuple(item["feature_columns"]) != STATE_MODEL_FEATURE_COLUMNS
        for item in state_profiles
    ):
        raise ValueError("The frozen market-state feature columns drifted.")
    if any(tuple(item["feature_columns"]) != STATE_FEATURE_COLUMNS for item in shuffled_profiles):
        raise ValueError("The frozen shuffled-label control feature columns drifted.")
    vwap = siblings["fresh_daily_current_session_vwap"]
    if vwap["exact_question"].get("scope_value") != VWAP_SCOPE_VALUE:
        raise ValueError("The frozen daily VWAP question drifted.")
    if tuple(vwap["controls"]) != tuple(g24v.CONTROLS):
        raise ValueError("The frozen daily VWAP controls drifted.")
    convergence = siblings["fresh_meme_convergence_crossing"]
    if convergence["exact_question"] != {
        "scope_value": "all_60_frozen_definitions",
        "metric": "crossing_count",
        "horizon_hours": 2,
        "market_scope": "top_ten_memes",
    }:
        raise ValueError("The frozen convergence question drifted.")
    if tuple(convergence["controls"]) != tuple(g25conv.CONTROLS):
        raise ValueError("The frozen convergence controls drifted.")
    if not (
        float(g24v.ZONE_HALF_WIDTH_ATR)
        == float(g24r.ZONE_HALF_WIDTH_ATR)
        == ZONE_HALF_WIDTH_ATR
    ):
        raise ValueError("The reused level-zone width drifted.")


def _fresh_period_ids() -> tuple[str, ...]:
    return tuple(str(item["id"]) for item in freshz.FRESH_PERIODS)


def _normalise_source_open(source_open: Series, length: int) -> Series:
    source = pd.to_datetime(Series(source_open).reset_index(drop=True), utc=True, errors="coerce")
    if len(source) != length:
        raise ValueError("Level source timestamps do not align with the market frame.")
    return source


def fresh_support_events(
    base: DataFrame,
    *,
    pair: str,
    cohort: str,
    level_name: str,
    level: np.ndarray,
    control: str,
    event_kind: str,
    source_open: Series,
) -> DataFrame:
    """Find causal contact episodes using the two frozen fresh periods."""
    levels = np.asarray(level, dtype=float)
    if len(levels) != len(base):
        raise ValueError("Level values do not align with the market frame.")
    high = pd.to_numeric(base["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(base["low"], errors="coerce").to_numpy(dtype=float)
    atr = pd.to_numeric(base["base_atr"], errors="coerce").to_numpy(dtype=float)
    pre_close = pd.to_numeric(base["pre_close"], errors="coerce").to_numpy(dtype=float)
    width = ZONE_HALF_WIDTH_ATR * atr
    valid = np.isfinite(levels) & (levels > 0.0) & np.isfinite(width) & (width > 0.0)
    contact = valid & (high >= levels - width) & (low <= levels + width)
    if event_kind == "contact":
        condition = contact
    elif event_kind == "near_miss":
        expanded = valid & (high >= levels - 2.0 * width) & (low <= levels + 2.0 * width)
        condition = expanded & ~contact
    else:
        raise ValueError(f"Unknown event kind: {event_kind}")
    starts = g0.episode_start_mask(condition, levels, width, cooldown=COOLDOWN_HOURS)
    starts[max(len(starts) - freshz.MAXIMUM_OUTCOME_HOURS, 0) :] = False
    dates = pd.to_datetime(base["date"].reset_index(drop=True), utc=True, errors="raise")
    if not dates.is_monotonic_increasing or dates.duplicated().any():
        raise ValueError(f"Market dates must be unique and ordered for {pair}.")
    period = freshz.assign_fresh_period(dates)
    starts &= period.ne(OUTSIDE_PERIOD).to_numpy(dtype=bool)
    indexes = np.flatnonzero(starts)
    source = _normalise_source_open(source_open, len(base))
    future_source = source.iloc[indexes].notna() & source.iloc[indexes].gt(
        dates.iloc[indexes].to_numpy()
    )
    if future_source.any():
        raise ValueError(f"Future level source detected for {pair}/{level_name}.")
    approach = np.select(
        [
            pre_close[indexes] < levels[indexes] - width[indexes],
            pre_close[indexes] > levels[indexes] + width[indexes],
        ],
        ["from_below", "from_above"],
        default="already_inside_or_unclear",
    )
    return DataFrame(
        {
            "cohort": cohort,
            "pair": pair,
            "period": period.iloc[indexes].to_numpy(),
            "control": control,
            "event_time": dates.iloc[indexes].to_numpy(),
            "base_index": indexes,
            "level_family": "fresh_confirmation_level",
            "level_name": level_name,
            "level_price": levels[indexes],
            "zone_half_width": width[indexes],
            "zone_half_width_atr": ZONE_HALF_WIDTH_ATR,
            "base_atr": atr[indexes],
            "pre_distance_atr": np.abs(levels[indexes] - pre_close[indexes]) / atr[indexes],
            "approach_state": approach,
            "source_open": source.iloc[indexes].to_numpy(),
            "match_tier": "not_applicable",
        }
    )


def fresh_matched_random_time_support(
    base: DataFrame,
    *,
    pair: str,
    cohort: str,
    level_name: str,
    actual: DataFrame,
) -> DataFrame:
    """Create outcome-blind ordinary-time controls matched within each fresh block."""
    if actual.empty:
        return DataFrame()
    dates = pd.to_datetime(base["date"].reset_index(drop=True), utc=True, errors="raise")
    period = freshz.assign_fresh_period(dates).astype(str).to_numpy()
    pre_close = pd.to_numeric(base["pre_close"], errors="coerce").to_numpy(dtype=float)
    atr = pd.to_numeric(base["base_atr"], errors="coerce").to_numpy(dtype=float)
    high = pd.to_numeric(base["high"], errors="coerce").to_numpy(dtype=float)
    low = pd.to_numeric(base["low"], errors="coerce").to_numpy(dtype=float)
    width = ZONE_HALF_WIDTH_ATR * atr
    actual_indexes = actual["base_index"].to_numpy(dtype=int)
    if ((actual_indexes < 0) | (actual_indexes >= len(base))).any():
        raise ValueError("Actual event indexes fall outside the market frame.")
    excluded = np.zeros(len(base), dtype=bool)
    excluded[actual_indexes] = True
    excluded = (
        np.convolve(
            excluded.astype(np.int8),
            np.ones(2 * COOLDOWN_HOURS + 1, dtype=np.int8),
            mode="same",
        )
        > 0
    )
    eligible = (
        np.isfinite(pre_close)
        & np.isfinite(atr)
        & (atr > 0.0)
        & (np.arange(len(base)) < len(base) - freshz.MAXIMUM_OUTCOME_HOURS)
        & np.isin(period, _fresh_period_ids())
        & ~excluded
    )
    match = actual[["period", "approach_state", "pre_distance_atr"]].copy()
    match["distance_band"] = np.digitize(
        match["pre_distance_atr"].to_numpy(dtype=float),
        [0.10, 0.25, 0.50, 1.0, 2.0],
    )
    selected: list[tuple[int, float, str]] = []
    for key, group in match.groupby(
        ["period", "approach_state", "distance_band"],
        sort=False,
        dropna=False,
    ):
        wanted_period, approach, distance_band = key
        distance = float(group["pre_distance_atr"].median())
        if approach == "from_below":
            sign = 1.0
        elif approach == "from_above":
            sign = -1.0
        else:
            sign = 0.0
            distance = 0.0
        if sign:
            distance = max(distance, ZONE_HALF_WIDTH_ATR * 1.01)
        pseudo_level = pre_close + sign * distance * atr
        contact = (high >= pseudo_level - width) & (low <= pseudo_level + width)
        candidates = np.flatnonzero(eligible & contact & (period == str(wanted_period)))
        rng = np.random.default_rng(
            g0.stable_hash_int(
                f"g21-matched-random|{pair}|{level_name}|{wanted_period}|"
                f"{approach}|{distance_band}"
            )
        )
        blocked = np.zeros(len(base), dtype=bool)
        count = 0
        for position in rng.permutation(candidates):
            position = int(position)
            if blocked[position]:
                continue
            selected.append((position, float(pseudo_level[position]), "exact_geometry"))
            blocked[
                max(0, position - COOLDOWN_HOURS) : min(
                    len(base), position + COOLDOWN_HOURS + 1
                )
            ] = True
            count += 1
            if count >= len(group):
                break
    if not selected:
        return DataFrame()
    selected.sort(key=lambda item: item[0])
    indexes = np.asarray([item[0] for item in selected], dtype=int)
    levels = np.asarray([item[1] for item in selected], dtype=float)
    approach = np.select(
        [
            pre_close[indexes] < levels - width[indexes],
            pre_close[indexes] > levels + width[indexes],
        ],
        ["from_below", "from_above"],
        default="already_inside_or_unclear",
    )
    return DataFrame(
        {
            "cohort": cohort,
            "pair": pair,
            "period": period[indexes],
            "control": "matched_random_time",
            "event_time": dates.iloc[indexes].to_numpy(),
            "base_index": indexes,
            "level_family": "fresh_confirmation_level",
            "level_name": level_name,
            "level_price": levels,
            "zone_half_width": width[indexes],
            "zone_half_width_atr": ZONE_HALF_WIDTH_ATR,
            "base_atr": atr[indexes],
            "pre_distance_atr": np.abs(levels - pre_close[indexes]) / atr[indexes],
            "approach_state": approach,
            "source_open": pd.NaT,
            "match_tier": [item[2] for item in selected],
        }
    )


def _selected_vwap_surface(base: DataFrame, pair: str, control: str) -> dict[str, Any]:
    anchor = "1d"
    mode = "current_session_through_previous_completed_candle"
    multiplier = 0.0
    side = "centre"
    centre, _, source = g24v.weighted_session_stats(base, anchor, mode)
    level = centre.copy()
    source = pd.to_datetime(source, utc=True)
    atr = pd.to_numeric(base["base_atr"], errors="coerce")
    if control == "stale_definition":
        level, source = level.shift(24), source.shift(24)
    elif control == "random_recent_analogue":
        lag = (168, 336, 504, 672)[
            g0.stable_hash_int(
                f"g24-vwap-lag|{pair}|{anchor}|{mode}|{multiplier}|{side}"
            )
            % 4
        ]
        level, source = level.shift(lag), source.shift(lag)
    elif control == "price_shift":
        direction = (
            -1.0
            if g0.stable_hash_int(
                f"g24-vwap-shift|{pair}|{anchor}|{mode}|{multiplier}|{side}"
            )
            % 2
            else 1.0
        )
        level = level + direction * atr
    elif control != "actual":
        raise ValueError(f"Unknown VWAP surface control: {control}")
    return {
        "level_name": VWAP_LEVEL_NAME,
        "level": level.to_numpy(dtype=float),
        "source_open": source,
    }


def build_vwap_pair(base: DataFrame, pair: str) -> DataFrame:
    """Build the one frozen daily current-session VWAP question and its controls."""
    actual_surface = _selected_vwap_surface(base, pair, "actual")
    actual = fresh_support_events(
        base,
        pair=pair,
        cohort="normal",
        level_name=VWAP_LEVEL_NAME,
        level=actual_surface["level"],
        control="actual",
        event_kind="contact",
        source_open=actual_surface["source_open"],
    )
    parts = [
        actual,
        fresh_matched_random_time_support(
            base,
            pair=pair,
            cohort="normal",
            level_name=f"random_vwap_{VWAP_LEVEL_NAME}",
            actual=actual,
        ),
        fresh_support_events(
            base,
            pair=pair,
            cohort="normal",
            level_name=VWAP_LEVEL_NAME,
            level=actual_surface["level"],
            control="near_miss",
            event_kind="near_miss",
            source_open=actual_surface["source_open"],
        ),
    ]
    for control in ("random_recent_analogue", "stale_definition", "price_shift"):
        surface = _selected_vwap_surface(base, pair, control)
        parts.append(
            fresh_support_events(
                base,
                pair=pair,
                cohort="normal",
                level_name=VWAP_LEVEL_NAME,
                level=surface["level"],
                control=control,
                event_kind="contact",
                source_open=surface["source_open"],
            )
        )
    available = [part for part in parts if not part.empty]
    if not available:
        return DataFrame()
    support = pd.concat(available, ignore_index=True, sort=False)
    support["level_family"] = "anchored_vwap"
    support["anchor_timeframe"] = "1d"
    support["anchor_mode"] = "current_session_through_previous_completed_candle"
    support["band_side"] = "centre"
    support["band_multiplier"] = 0.0
    support.sort_values(["control", "period", "event_time"], inplace=True, kind="stable")
    support.drop_duplicates(["control", "period", "event_time"], inplace=True)
    support.reset_index(drop=True, inplace=True)
    return support


def _convergence_level_support(
    base: DataFrame,
    *,
    pair: str,
    surface: dict[str, Any],
    level: np.ndarray,
    source_open: Series,
    control: str,
    event_kind: str,
) -> DataFrame:
    events = fresh_support_events(
        base,
        pair=pair,
        cohort="meme",
        level_name=g24r.identity(surface),
        level=level,
        control=control,
        event_kind=event_kind,
        source_open=source_open,
    )
    return g24r.relabel(events, surface, control)


def build_convergence_pair(base: DataFrame, pair: str) -> tuple[DataFrame, list[dict[str, Any]]]:
    """Pool the 60 frozen round/distribution definitions without choosing a winner."""
    surfaces = g24r.combination_surfaces(base, pair)
    if len(surfaces) != g25conv.SOURCE_SURFACE_COUNT:
        raise ValueError(f"Expected 60 convergence definitions for {pair}, got {len(surfaces)}.")
    parts: list[DataFrame] = []
    for surface in surfaces:
        source = pd.to_datetime(surface["source_open"], utc=True)
        actual = _convergence_level_support(
            base,
            pair=pair,
            surface=surface,
            level=np.asarray(surface["level"], dtype=float),
            source_open=source,
            control="actual",
            event_kind="contact",
        )
        round_level = np.asarray(surface["round_level"], dtype=float).copy()
        distribution_level = np.asarray(surface["distribution_level"], dtype=float).copy()
        convergent = np.asarray(surface["convergent"], dtype=bool)
        round_level[convergent] = np.nan
        distribution_level[convergent] = np.nan
        isolated_round = _convergence_level_support(
            base,
            pair=pair,
            surface=surface,
            level=round_level,
            source_open=source,
            control="isolated_round_component",
            event_kind="contact",
        )
        isolated_distribution = _convergence_level_support(
            base,
            pair=pair,
            surface=surface,
            level=distribution_level,
            source_open=source,
            control="isolated_distribution_component",
            event_kind="contact",
        )
        matched = fresh_matched_random_time_support(
            base,
            pair=pair,
            cohort="meme",
            level_name=f"random_{g24r.identity(surface)}",
            actual=actual,
        )
        parts.extend(
            [
                actual,
                isolated_round,
                isolated_distribution,
                g24r.pseudo_convergence_support(
                    actual,
                    isolated_round,
                    isolated_distribution,
                    pair=pair,
                    surface=surface,
                ),
                g24r.relabel(matched, surface, "matched_random_time"),
                _convergence_level_support(
                    base,
                    pair=pair,
                    surface=surface,
                    level=np.asarray(surface["level"], dtype=float),
                    source_open=source,
                    control="near_miss",
                    event_kind="near_miss",
                ),
            ]
        )
        for control in ("random_recent_analogue", "stale_definition", "price_shift"):
            level, transformed_source = g24r.transformed_level(surface, base, pair, control)
            parts.append(
                _convergence_level_support(
                    base,
                    pair=pair,
                    surface=surface,
                    level=level,
                    source_open=transformed_source,
                    control=control,
                    event_kind="contact",
                )
            )
    available = [part for part in parts if not part.empty]
    if not available:
        return DataFrame(), []
    source_support = pd.concat(available, ignore_index=True, sort=False)
    source_support.sort_values(
        [
            "control",
            "period",
            "grid_step_multiplier",
            "lookback_hours",
            "quantile",
            "event_time",
            "pre_distance_atr",
        ],
        inplace=True,
        kind="stable",
    )
    source_support.drop_duplicates(
        [
            "control",
            "period",
            "grid_step_multiplier",
            "lookback_hours",
            "quantile",
            "event_time",
        ],
        keep="first",
        inplace=True,
    )
    source_support.reset_index(drop=True, inplace=True)
    return g25conv.equal_support_surface(source_support)


def _period_control_counts(frame: DataFrame, controls: Sequence[str]) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for period in _fresh_period_ids():
        cell = frame.loc[frame["period"].astype(str).eq(period)] if not frame.empty else frame
        counts = cell["control"].astype(str).value_counts() if not cell.empty else Series(dtype=int)
        output[period] = {control: int(counts.get(control, 0)) for control in controls}
    return output


def _pair_period_support_gate(counts: dict[str, dict[str, int]]) -> dict[str, bool]:
    return {
        period: bool(values and min(values.values()) >= MIN_ROWS_PER_PAIR_CONTROL_PERIOD)
        for period, values in counts.items()
    }


def _base_frames(frozen: dict[str, Any]) -> dict[tuple[str, str], DataFrame]:
    frames: dict[tuple[str, str], DataFrame] = {}
    for cohort, pairs in frozen["data_contract"]["cohort_pairs"].items():
        for pair in pairs:
            base, _ = g20s.base_and_state(str(pair), str(cohort))
            frames[(str(cohort), str(pair))] = base.reset_index(drop=True)
    return frames


def _cross_asset_context(
    frozen: dict[str, Any], bases: dict[tuple[str, str], DataFrame]
) -> DataFrame:
    inputs: dict[str, DataFrame] = {}
    for cohort in ("normal", "meme"):
        for pair in frozen["data_contract"]["cohort_pairs"][cohort]:
            if pair in inputs:
                continue
            base = bases[(cohort, pair)]
            selected = base[["date", "close", "volume", "base_atr"]].copy()
            selected["date"] = pd.to_datetime(selected["date"], utc=True, errors="raise")
            inputs[pair] = selected.set_index("date").sort_index()
    return g22d.causal_cross_asset_context(inputs)


def state_feature_surface(
    base: DataFrame, context: DataFrame, pair: str
) -> tuple[DataFrame, DataFrame]:
    """Reuse the frozen causal feature equations needed by the model and label control."""
    features, support = g22cache.feature_surface(base, context, pair)
    missing = sorted(set(STATE_FEATURE_COLUMNS).difference(features.columns))
    if missing:
        raise ValueError(f"State feature surface is incomplete for {pair}: {missing}")
    return features, support


def _state_pair_audit(
    feature: DataFrame, support: DataFrame, *, pair: str, cohort: str
) -> dict[str, Any]:
    dates = pd.to_datetime(feature["date"], utc=True, errors="raise")
    support_dates = pd.to_datetime(support["date"], utc=True, errors="raise")
    if not dates.equals(support_dates):
        raise ValueError(f"State feature/support dates do not align for {pair}.")
    ready = feature[f"ready__{STATE_READY_BLOCK}"].fillna(False).astype(bool)
    shuffled = pd.to_datetime(support["shuffled_source_date"], utc=True, errors="coerce")
    if (shuffled.loc[ready] >= dates.loc[ready]).any():
        raise ValueError(f"Shuffled labels are not sourced strictly from the past for {pair}.")
    available_dates = set(dates.to_numpy())
    periods = []
    for item in freshz.FRESH_PERIODS:
        start = pd.Timestamp(item["start_utc"])
        end = pd.Timestamp(item["end_utc_exclusive"])
        training_start = start - pd.Timedelta(days=freshz.STATE_TRAINING_DAYS[cohort])
        training_cutoff = start - pd.Timedelta(hours=TARGET_PURGE_HOURS)
        training = ready & dates.ge(training_start) & dates.lt(training_cutoff)
        prediction = ready & dates.ge(start) & dates.lt(end)
        shuffled_sources_present = int(
            sum(value in available_dates for value in shuffled.loc[training].dropna().to_numpy())
        )
        training_rows = int(training.sum())
        prediction_rows = int(prediction.sum())
        periods.append(
            {
                "period": str(item["id"]),
                "training_start_utc": training_start.isoformat(),
                "training_cutoff_exclusive_utc": training_cutoff.isoformat(),
                "training_rows": training_rows,
                "prediction_rows": prediction_rows,
                "shuffled_training_source_rows_present": shuffled_sources_present,
                "gate_pass": bool(
                    training_rows >= MIN_TRAINING_ROWS
                    and prediction_rows >= MIN_PREDICTION_ROWS
                    and shuffled_sources_present == training_rows
                ),
            }
        )
    return {
        "pair": pair,
        "cohort": cohort,
        "rows": len(feature),
        "ready_rows": int(ready.sum()),
        "periods": periods,
        "gate_pass": bool(periods and all(item["gate_pass"] for item in periods)),
    }


def _trim_state_inputs(
    feature: DataFrame, support: DataFrame, cohort: str
) -> tuple[DataFrame, DataFrame]:
    first, last = freshz.required_data_bounds(cohort)
    dates = pd.to_datetime(feature["date"], utc=True, errors="raise")
    mask = dates.ge(first) & dates.le(last)
    return (
        feature.loc[mask].reset_index(drop=True),
        support.loc[mask].reset_index(drop=True),
    )


def _write_state_support(
    frozen: dict[str, Any],
    bases: dict[tuple[str, str], DataFrame],
    output_root: Path,
) -> dict[str, Any]:
    context = _cross_asset_context(frozen, bases)
    context_path = output_root / "state" / "causal_cross_asset_context.parquet"
    context_path.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(context, context_path)
    inventory = []
    for cohort in ("normal", "meme"):
        for pair in frozen["data_contract"]["cohort_pairs"][cohort]:
            feature, support = state_feature_surface(bases[(cohort, pair)], context, pair)
            feature, support = _trim_state_inputs(feature, support, cohort)
            audit = _state_pair_audit(feature, support, pair=pair, cohort=cohort)
            stem = g0.pair_file_stem(pair)
            feature_path = output_root / "state" / cohort / "feature_cache" / f"{stem}.parquet"
            mapping_path = output_root / "state" / cohort / "label_mapping" / f"{stem}.parquet"
            feature_path.parent.mkdir(parents=True, exist_ok=True)
            mapping_path.parent.mkdir(parents=True, exist_ok=True)
            g0.atomic_write_parquet(feature, feature_path)
            g0.atomic_write_parquet(support, mapping_path)
            inventory.append(
                {
                    **audit,
                    "feature": artifact(feature_path),
                    "label_mapping": artifact(mapping_path),
                }
            )
    return {
        "route_id": "fresh_market_state_activity",
        "feature_columns_required_by_any_frozen_profile": list(STATE_FEATURE_COLUMNS),
        "main_model_feature_columns": list(STATE_MODEL_FEATURE_COLUMNS),
        "ready_block": STATE_READY_BLOCK,
        "minimum_training_rows": MIN_TRAINING_ROWS,
        "minimum_prediction_rows": MIN_PREDICTION_ROWS,
        "causal_cross_asset_context": artifact(context_path),
        "inventory": inventory,
        "support_gate_pass": bool(inventory and all(item["gate_pass"] for item in inventory)),
        "future_outcomes_read": False,
    }


def _write_level_support(
    frozen: dict[str, Any],
    bases: dict[tuple[str, str], DataFrame],
    output_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    vwap_inventory = []
    for pair in frozen["data_contract"]["cohort_pairs"]["normal"]:
        support = build_vwap_pair(bases[("normal", pair)], pair)
        counts = _period_control_counts(support, VWAP_CONTROLS)
        gates = _pair_period_support_gate(counts)
        path = output_root / "daily_vwap" / f"{g0.pair_file_stem(pair)}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        g0.atomic_write_parquet(support, path)
        vwap_inventory.append(
            {
                "pair": pair,
                "cohort": "normal",
                "rows": len(support),
                "period_control_counts": counts,
                "period_support_gate": gates,
                "support": artifact(path),
            }
        )
    convergence_inventory = []
    for pair in frozen["data_contract"]["cohort_pairs"]["meme"]:
        support, representation = build_convergence_pair(bases[("meme", pair)], pair)
        counts = _period_control_counts(support, CONVERGENCE_CONTROLS)
        gates = _pair_period_support_gate(counts)
        path = output_root / "meme_convergence" / f"{g0.pair_file_stem(pair)}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        g0.atomic_write_parquet(support, path)
        convergence_inventory.append(
            {
                "pair": pair,
                "cohort": "meme",
                "rows": len(support),
                "source_definition_count": g25conv.SOURCE_SURFACE_COUNT,
                "period_control_counts": counts,
                "period_support_gate": gates,
                "representation_audit": representation,
                "support": artifact(path),
            }
        )
    vwap_eligible = {
        period: sum(bool(item["period_support_gate"][period]) for item in vwap_inventory)
        for period in _fresh_period_ids()
    }
    convergence_eligible = {
        period: sum(
            bool(item["period_support_gate"][period]) for item in convergence_inventory
        )
        for period in _fresh_period_ids()
    }
    vwap = {
        "route_id": "fresh_daily_current_session_vwap",
        "level_name": VWAP_LEVEL_NAME,
        "controls": list(VWAP_CONTROLS),
        "inventory": vwap_inventory,
        "eligible_pairs_by_period": vwap_eligible,
        "support_gate_pass": all(
            count >= MIN_ELIGIBLE_COINS for count in vwap_eligible.values()
        ),
        "future_outcomes_read": False,
    }
    convergence = {
        "route_id": "fresh_meme_convergence_crossing",
        "scope_value": "all_60_frozen_definitions",
        "controls": list(CONVERGENCE_CONTROLS),
        "inventory": convergence_inventory,
        "eligible_pairs_by_period": convergence_eligible,
        "support_gate_pass": all(
            count >= MIN_ELIGIBLE_COINS for count in convergence_eligible.values()
        ),
        "future_outcomes_read": False,
    }
    return vwap, convergence


def _source_contracts(coverage_path: Path) -> dict[str, dict[str, Any]]:
    return {
        "fresh_question_freeze": artifact(freshz.FREEZE_PATH),
        "coverage_gate": artifact(coverage_path),
        "support_builder": artifact(ANALYSIS_PATH),
        "base_market_builder": artifact(Path(g20s.__file__)),
        "state_feature_builder": artifact(Path(g22cache.__file__)),
        "cross_asset_context_builder": artifact(Path(g22d.__file__)),
        "anchored_vwap_builder": artifact(Path(g24v.__file__)),
        "convergence_builder": artifact(Path(g24r.__file__)),
        "convergence_pool_builder": artifact(Path(g25conv.__file__)),
    }


def _verify_existing_result(result: dict[str, Any]) -> None:
    if result.get("status") != "frozen_all_three_supports_before_fresh_outcomes":
        return
    for route in result["routes"].values():
        if "causal_cross_asset_context" in route:
            items = [route["causal_cross_asset_context"]]
            items.extend(
                artifact_item
                for row in route["inventory"]
                for artifact_item in (row["feature"], row["label_mapping"])
            )
        else:
            items = [row["support"] for row in route["inventory"]]
        for item in items:
            path = Path(item["path"])
            if not path.is_file() or g0.sha256_file(path) != item["sha256"]:
                raise ValueError(f"Frozen fresh support artifact changed: {path}")


def prepare_support(
    support_id: str,
    coverage_run_id: str,
    *,
    refresh_coverage: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Apply the hard gate, then build and jointly freeze all outcome-blind support."""
    result_path = support_result_path(support_id)
    if result_path.is_file() and not overwrite:
        existing = json.loads(result_path.read_text(encoding="utf-8"))
        if existing.get("status") == "frozen_all_three_supports_before_fresh_outcomes":
            _verify_existing_result(existing)
            return existing
    frozen, coverage = load_inputs(
        coverage_run_id,
        refresh_coverage=refresh_coverage,
    )
    coverage_path = coverage_result_path(coverage_run_id)
    if coverage.get("status") != "ready_to_freeze_fresh_event_support":
        waiting = {
            "schema_version": 1,
            "batch_id": frozen["batch_id"],
            "support_id": support_id,
            "created_at_utc": g0.utc_now(),
            "status": "waiting_for_complete_fresh_ohlcv",
            "coverage": coverage.get("coverage", {}),
            "next_action": (
                "Do not build support or open outcomes until all 20 coverage cells pass."
            ),
            "bulky_support_directory_created": False,
            "future_reaction_outcomes_read": False,
            "future_signed_direction_read": False,
            "profit_read": False,
            "artifacts": {
                "fresh_question_freeze": artifact(freshz.FREEZE_PATH),
                "coverage_gate": artifact(coverage_path),
            },
        }
        result_path.parent.mkdir(parents=True, exist_ok=True)
        g0.atomic_write_json(waiting, result_path)
        return waiting

    output_root = ARTIFACT_ROOT / support_id
    output_root.mkdir(parents=True, exist_ok=True)
    bases = _base_frames(frozen)
    state = _write_state_support(frozen, bases, output_root)
    vwap, convergence = _write_level_support(frozen, bases, output_root)
    routes = {
        state["route_id"]: state,
        convergence["route_id"]: convergence,
        vwap["route_id"]: vwap,
    }
    all_pass = all(bool(route["support_gate_pass"]) for route in routes.values())
    result = {
        "schema_version": 1,
        "batch_id": frozen["batch_id"],
        "support_id": support_id,
        "created_at_utc": g0.utc_now(),
        "status": (
            "frozen_all_three_supports_before_fresh_outcomes"
            if all_pass
            else "insufficient_outcome_blind_support"
        ),
        "all_three_support_routes_built_together": True,
        "all_three_support_gates_pass": all_pass,
        "routes": routes,
        "source_contracts": _source_contracts(coverage_path),
        "future_reaction_outcomes_read": False,
        "future_signed_direction_read": False,
        "profit_read": False,
        "next_action": (
            "The separate outcome phase may now materialize only the frozen unsigned targets."
            if all_pass
            else "Stop before outcomes: at least one frozen route lacks enough support."
        ),
    }
    result_path.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(result, result_path)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-id", default=DEFAULT_SUPPORT_ID)
    parser.add_argument("--coverage-run-id", default=freshz.DEFAULT_COVERAGE_ID)
    parser.add_argument("--use-existing-coverage", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = prepare_support(
        args.support_id,
        args.coverage_run_id,
        refresh_coverage=not args.use_existing_coverage,
        overwrite=args.overwrite,
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
