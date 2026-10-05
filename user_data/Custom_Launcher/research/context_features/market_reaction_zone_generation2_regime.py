from __future__ import annotations

import os


# ruff: noqa: E402
# Keep this direct analysis to one numerical-library thread.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYARROW_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation0 import (
    LARGE_ARTIFACT_ROOT,
    OUTPUT_ROOT,
    atomic_write_json,
    atomic_write_parquet,
    future_path_matrices,
    load_manifest,
    normalize_dates,
    numeric_series,
    prepare_base_market_frame,
    sha256_file,
    utc_now,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation1_localization import (  # noqa: E501
    causal_market_context,
)


FROZEN_BATCH = OUTPUT_ROOT / "generation1_review" / "g2_frozen_branch_batch.json"
REPORT_ROOT = OUTPUT_ROOT / "generation2_branches" / "g2f_regime_modulation"
ARTIFACT_ROOT = (
    LARGE_ARTIFACT_ROOT / "generation2_branches" / "g2f_regime_modulation"
)
OUTPUT_SCHEMA_VERSION = 1
CAUSAL_QUANTILE_LOOKBACK_HOURS = 720
CAUSAL_QUANTILE_MINIMUM_HOURS = 168
MIN_PAIR_STATE_ROWS = 5
MIN_COHORT_COINS = 5
MIN_COHORT_STATE_ROWS = 50
MAX_STATE_SMD = 0.50
MAX_MEDIAN_STATE_SMD = 0.20

SOURCE_OUTCOMES = (
    "contact_range_ratio",
    "contact_volume_ratio",
    "abs_excursion_atr_h1",
    "range_ratio_h4",
    "volume_ratio_h4",
    "dwell_fraction_h4",
    "crossings_h4",
    "volume_ratio_h2",
    "crossings_h1",
    "range_ratio_h24",
    "volume_ratio_h24",
    "range_ratio_h48",
    "volume_ratio_h48",
)
PRESSURE_OUTCOMES = (
    "pressure_change_abs_h1",
    "pressure_change_abs_h4",
    "pressure_change_abs_h24",
)
OUTCOMES = (*SOURCE_OUTCOMES, *PRESSURE_OUTCOMES)


@dataclass(frozen=True)
class RegimeLens:
    name: str
    low_label: str
    high_label: str
    raw_state_columns: tuple[str, ...]


LENSES = (
    RegimeLens(
        "realized_volatility",
        "quiet_realized_volatility",
        "high_realized_volatility",
        ("state_realized_volatility_24h",),
    ),
    RegimeLens(
        "relative_volume",
        "low_relative_volume",
        "high_relative_volume",
        ("state_relative_volume_24h",),
    ),
    RegimeLens(
        "adx_trend_strength",
        "low_adx",
        "high_adx",
        ("state_adx14",),
    ),
    RegimeLens(
        "ema20_absolute_slope",
        "flat_ema20",
        "steep_ema20",
        ("state_ema20_abs_slope_6h_atr",),
    ),
    RegimeLens(
        "sma50_absolute_slope",
        "flat_sma50",
        "steep_sma50",
        ("state_sma50_abs_slope_12h_atr",),
    ),
    RegimeLens(
        "btc_absolute_return",
        "small_btc_move",
        "large_btc_move",
        ("state_btc_abs_return_24h",),
    ),
    RegimeLens(
        "btc_volatility",
        "low_btc_volatility",
        "high_btc_volatility",
        ("state_btc_atr_pct",),
    ),
    RegimeLens(
        "top10_breadth_alignment",
        "mixed_coin_directions",
        "aligned_coin_directions",
        ("state_top10_breadth_extremity",),
    ),
    RegimeLens(
        "top10_return_dispersion",
        "low_coin_dispersion",
        "high_coin_dispersion",
        ("state_top10_return_dispersion",),
    ),
    RegimeLens(
        "top10_common_volume",
        "low_common_volume",
        "high_common_volume",
        ("state_top10_common_volume",),
    ),
    RegimeLens(
        "quiet_vs_expanding",
        "quiet_low_volume_market",
        "expanding_high_volume_market",
        ("state_realized_volatility_24h", "state_relative_volume_24h"),
    ),
    RegimeLens(
        "range_vs_directional_trend",
        "range_like_market",
        "directional_trend_market",
        (
            "state_adx14",
            "state_ema20_abs_slope_6h_atr",
            "state_sma50_abs_slope_12h_atr",
        ),
    ),
    RegimeLens(
        "calm_vs_crypto_stress",
        "calm_crypto_wide_market",
        "stressed_crypto_wide_market",
        (
            "state_btc_abs_return_24h",
            "state_btc_atr_pct",
            "state_top10_breadth_extremity",
            "state_top10_return_dispersion",
            "state_top10_common_volume",
        ),
    ),
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generation 2F direct pre-contact regime interaction test. It measures "
            "direction-neutral changes in level reaction and does not optimize profit."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    validate_frozen_branch()
    source_path, source_record, source_integrity = validate_source_run(args.source_run)
    compact_dir = REPORT_ROOT / args.run_id
    bulky_dir = ARTIFACT_ROOT / args.run_id
    compact_dir.mkdir(parents=True, exist_ok=True)
    bulky_dir.mkdir(parents=True, exist_ok=True)
    record_path = compact_dir / "g2f_run_record.json"
    request = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "frozen_batch_sha256": sha256_file(FROZEN_BATCH),
        "source_event_pairs": str(source_path.resolve()),
        "source_event_pairs_sha256": sha256_file(source_path),
        "source_run_request_sha256": source_record.get("request_sha256"),
        "source_integrity_contract_passed": True,
        "source_integrity_audit": {
            key: source_integrity.get(key)
            for key in (
                "pair_failures",
                "matched_event_pairs_before_overlap_purge",
                "independent_event_pairs_after_horizon_purge",
                "pre_distance_atr_abs_difference_max",
                "event_separation_hours_min",
                "period_end_embargo_hours",
            )
        },
        "causal_quantile_lookback_hours": CAUSAL_QUANTILE_LOOKBACK_HOURS,
        "causal_quantile_minimum_hours": CAUSAL_QUANTILE_MINIMUM_HOURS,
        "regime_lenses": [lens.name for lens in LENSES],
        "outcomes": list(OUTCOMES),
        "direction_prediction": False,
        "profit_optimization": False,
    }
    request_sha256 = stable_json_sha256(request)
    record = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": args.run_id,
        "status": "running",
        "started_at_utc": utc_now(),
        "orchestrator_pid": os.getpid(),
        "command": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
        "request": request,
        "request_sha256": request_sha256,
        "baseline": (
            "The same level routes against their existing near-miss, no-level, "
            "shifted, stale, shuffled, and isolated controls in another causal regime."
        ),
        "hypothesis": (
            "A predeclared quiet, expanding, trend, or crypto-wide stress state changes "
            "the level-versus-control reaction difference in repeated periods."
        ),
        "pass_rule": (
            "A trader-readable interaction must repeat in at least two validation "
            "periods, retain at least five coins and fifty independent rows in each "
            "state, survive applicable controls and leave-one-coin-out checks, and "
            "retain acceptable continuous-state balance."
        ),
        "direction_prediction": False,
        "profit_optimization": False,
    }
    atomic_write_json(record, record_path)
    try:
        source = pd.read_parquet(source_path)
        validate_source_events(source, manifest)
        context = causal_market_context(manifest)
        enriched_frames: list[DataFrame] = []
        time_violations = 0
        for pair, pair_events in source.groupby("pair", observed=True, sort=True):
            base = prepare_base_market_frame(str(pair), manifest)
            state = causal_regime_state(base, context)
            enriched, violations = attach_pair_regimes(
                pair_events.reset_index(drop=True),
                base=base,
                state=state,
            )
            enriched_frames.append(enriched)
            time_violations += violations
        enriched = pd.concat(enriched_frames, ignore_index=True)
        enriched["output_schema_version"] = OUTPUT_SCHEMA_VERSION
        enriched["run_request_sha256"] = request_sha256
        event_output = bulky_dir / "g2f_event_pairs_with_regimes.parquet"
        atomic_write_parquet(enriched, event_output)

        pair_period = pair_period_interactions(enriched)
        cohort_period = cohort_period_interactions(pair_period)
        leave_one_out = leave_one_coin_out_interactions(pair_period)
        atomic_write_parquet(
            pair_period,
            compact_dir / "g2f_pair_period_interactions.parquet",
        )
        atomic_write_parquet(
            cohort_period,
            compact_dir / "g2f_cohort_period_interactions.parquet",
        )
        atomic_write_parquet(
            leave_one_out,
            compact_dir / "g2f_leave_one_coin_out.parquet",
        )
        integrity = integrity_record(
            source=source,
            enriched=enriched,
            pair_period=pair_period,
            time_violations=time_violations,
        )
        atomic_write_json(integrity, compact_dir / "g2f_integrity.json")
        if not integrity["passed"]:
            raise ValueError("G2F integrity checks failed; inspect g2f_integrity.json.")
        record.update(
            {
                "status": "completed",
                "completed_at_utc": utc_now(),
                "source_event_pairs": len(source),
                "enriched_event_pairs": len(enriched),
                "pair_period_rows": len(pair_period),
                "cohort_period_rows": len(cohort_period),
                "leave_one_coin_out_rows": len(leave_one_out),
                "event_output": str(event_output),
                "integrity": str(compact_dir / "g2f_integrity.json"),
            }
        )
        atomic_write_json(record, record_path)
    except Exception as exc:
        record.update(
            {
                "status": "failed",
                "failed_at_utc": utc_now(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        atomic_write_json(record, record_path)
        raise
    return 0


def validate_frozen_branch() -> None:
    payload = json.loads(FROZEN_BATCH.read_text(encoding="utf-8"))
    branches = {item.get("id") for item in payload.get("branches", [])}
    if "g2f_market_regime_modulation_of_level_reaction" not in branches:
        raise ValueError("The frozen Generation 2 batch does not contain G2F.")


def validate_source_run(source_run: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    record_path = source_run / "g2ab_localization_run_record.json"
    integrity_path = source_run / "g2ab_integrity.json"
    events_path = source_run / "g2ab_independent_event_pairs.parquet"
    for path in (record_path, integrity_path, events_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required frozen G2AB source is missing: {path}")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    integrity = json.loads(integrity_path.read_text(encoding="utf-8"))
    if record.get("status") != "completed":
        raise ValueError("G2F source G2AB run is not completed.")
    validate_source_integrity_contract(integrity)
    return events_path, record, integrity


def validate_source_integrity_contract(integrity: dict[str, Any]) -> None:
    """Validate the explicit G2AB audit fields; that schema has no pass flag."""
    issues: list[str] = []
    if int(integrity.get("pair_failures", -1)) != 0:
        issues.append("pair_failures is not zero")
    matched = int(integrity.get("matched_event_pairs_before_overlap_purge", 0))
    independent = int(
        integrity.get("independent_event_pairs_after_horizon_purge", 0)
    )
    if independent <= 0 or matched < independent:
        issues.append("independent event-pair counts are invalid")
    distance = float(integrity.get("pre_distance_atr_abs_difference_max", np.inf))
    if not np.isfinite(distance) or distance > 0.100001:
        issues.append("pre-distance matching exceeded the frozen 0.10 ATR caliper")
    if float(integrity.get("event_separation_hours_min", 0.0)) <= 0.0:
        issues.append("event separation is not positive")
    if int(integrity.get("period_end_embargo_hours", 0)) < 48:
        issues.append("period-end embargo is shorter than the longest source outcome")
    if not integrity.get("periods_present"):
        issues.append("no chronological periods are present")
    if not integrity.get("routes_present"):
        issues.append("no level routes are present")
    if integrity.get("direction_prediction") is not False:
        issues.append("direction boundary is not preserved")
    if integrity.get("profit_optimization") is not False:
        issues.append("profit boundary is not preserved")
    if issues:
        raise ValueError("G2F source G2AB integrity is unusable: " + "; ".join(issues))


def validate_source_events(frame: DataFrame, manifest: dict[str, Any]) -> None:
    required = {
        "pair",
        "route_id",
        "period",
        "actual_scope",
        "control",
        "actual_base_index",
        "control_base_index",
        "actual_event_time",
        "control_event_time",
        "direction_prediction",
        "profit_optimization",
        *(f"delta__{outcome}" for outcome in SOURCE_OUTCOMES),
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"G2F source event pairs lack columns: {missing}")
    if frame.empty:
        raise ValueError("G2F source event-pair table is empty.")
    if set(frame["pair"].astype(str)).difference(manifest["data"]["pairs"]):
        raise ValueError("G2F source contains a pair outside the frozen manifest.")
    if frame["direction_prediction"].astype(bool).any():
        raise ValueError("G2F source crosses the direction-prediction boundary.")
    if frame["profit_optimization"].astype(bool).any():
        raise ValueError("G2F source crosses the profit-optimization boundary.")


def causal_regime_state(base: DataFrame, context: DataFrame) -> DataFrame:
    high = numeric_series(base["high"])
    low = numeric_series(base["low"])
    close = numeric_series(base["close"])
    volume = numeric_series(base["volume"]).clip(lower=0.0)
    known_atr = numeric_series(base["base_atr"]).replace(0.0, np.nan)
    previous_close = close.shift(1)
    true_range = pd.concat(
        (
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ),
        axis=1,
    ).max(axis=1)
    upward = high.diff()
    downward = -low.diff()
    plus_dm = upward.where((upward > downward) & (upward > 0.0), 0.0)
    minus_dm = downward.where((downward > upward) & (downward > 0.0), 0.0)
    smoothed_tr = true_range.ewm(
        alpha=1.0 / 14.0, min_periods=14, adjust=False
    ).mean()
    plus_di = (
        100.0
        * plus_dm.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
        / smoothed_tr.replace(0.0, np.nan)
    )
    minus_di = (
        100.0
        * minus_dm.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean()
        / smoothed_tr.replace(0.0, np.nan)
    )
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(
        0.0, np.nan
    )
    ema20 = close.ewm(span=20, min_periods=20, adjust=False).mean()
    sma50 = close.rolling(50, min_periods=50).mean()
    state = DataFrame({"date": normalize_dates(base["date"])})
    state["state_realized_volatility_24h"] = (
        close.pct_change().rolling(24, min_periods=12).std(ddof=0).shift(1)
    )
    state["state_relative_volume_24h"] = volume.shift(1).div(
        volume.shift(2).rolling(24, min_periods=12).median().replace(0.0, np.nan)
    )
    state["state_adx14"] = (
        dx.ewm(alpha=1.0 / 14.0, min_periods=14, adjust=False).mean().shift(1)
        / 100.0
    )
    state["state_ema20_abs_slope_6h_atr"] = (
        (ema20.shift(1) - ema20.shift(7)).abs().div(6.0 * known_atr)
    )
    state["state_sma50_abs_slope_12h_atr"] = (
        (sma50.shift(1) - sma50.shift(13)).abs().div(12.0 * known_atr)
    )
    context_columns = [
        "date",
        "state_btc_return_24h",
        "state_btc_atr_pct",
        "state_top10_breadth",
        "state_top10_return_dispersion",
        "state_top10_common_volume",
    ]
    available_context = context[
        [column for column in context_columns if column in context.columns]
    ].copy()
    state = state.merge(
        available_context,
        on="date",
        how="left",
        validate="one_to_one",
    )
    required_context = set(context_columns).difference({"date"})
    missing_context = sorted(required_context.difference(state.columns))
    if missing_context:
        raise ValueError(f"G2F market context lacks columns: {missing_context}")
    state["state_btc_abs_return_24h"] = pd.to_numeric(
        state["state_btc_return_24h"], errors="coerce"
    ).abs()
    state["state_top10_breadth_extremity"] = (
        pd.to_numeric(state["state_top10_breadth"], errors="coerce").sub(0.5).abs()
        * 2.0
    )

    raw_columns = sorted(
        {
            column
            for lens in LENSES
            for column in lens.raw_state_columns
        }
    )
    for column in raw_columns:
        state[f"regime_component__{column}"] = causal_tercile_bucket(state[column])

    state["regime__realized_volatility"] = state[
        "regime_component__state_realized_volatility_24h"
    ]
    state["regime__relative_volume"] = state[
        "regime_component__state_relative_volume_24h"
    ]
    state["regime__adx_trend_strength"] = state[
        "regime_component__state_adx14"
    ]
    state["regime__ema20_absolute_slope"] = state[
        "regime_component__state_ema20_abs_slope_6h_atr"
    ]
    state["regime__sma50_absolute_slope"] = state[
        "regime_component__state_sma50_abs_slope_12h_atr"
    ]
    state["regime__btc_absolute_return"] = state[
        "regime_component__state_btc_abs_return_24h"
    ]
    state["regime__btc_volatility"] = state[
        "regime_component__state_btc_atr_pct"
    ]
    state["regime__top10_breadth_alignment"] = state[
        "regime_component__state_top10_breadth_extremity"
    ]
    state["regime__top10_return_dispersion"] = state[
        "regime_component__state_top10_return_dispersion"
    ]
    state["regime__top10_common_volume"] = state[
        "regime_component__state_top10_common_volume"
    ]
    state["regime__quiet_vs_expanding"] = coherent_pair_regime(
        state["regime__realized_volatility"],
        state["regime__relative_volume"],
    )
    trend_votes = state[
        [
            "regime__adx_trend_strength",
            "regime__ema20_absolute_slope",
            "regime__sma50_absolute_slope",
        ]
    ].to_numpy(dtype=np.int8)
    state["regime__range_vs_directional_trend"] = vote_regime(
        trend_votes,
        required=2,
    )
    stress_votes = state[
        [
            "regime__btc_absolute_return",
            "regime__btc_volatility",
            "regime__top10_breadth_alignment",
            "regime__top10_return_dispersion",
            "regime__top10_common_volume",
        ]
    ].to_numpy(dtype=np.int8)
    state["regime__calm_vs_crypto_stress"] = vote_regime(
        stress_votes,
        required=3,
    )
    return state.replace([np.inf, -np.inf], np.nan)


def causal_tercile_bucket(
    values: Series,
    *,
    lookback: int = CAUSAL_QUANTILE_LOOKBACK_HOURS,
    minimum: int = CAUSAL_QUANTILE_MINIMUM_HOURS,
) -> Series:
    numeric = pd.to_numeric(values, errors="coerce")
    history = numeric.shift(1)
    lower = history.rolling(lookback, min_periods=minimum).quantile(1.0 / 3.0)
    upper = history.rolling(lookback, min_periods=minimum).quantile(2.0 / 3.0)
    output = pd.Series(0, index=values.index, dtype="int8")
    valid = numeric.notna() & lower.notna() & upper.notna()
    output.loc[valid & numeric.le(lower)] = -1
    output.loc[valid & numeric.ge(upper)] = 1
    return output


def coherent_pair_regime(left: Series, right: Series) -> Series:
    output = pd.Series(0, index=left.index, dtype="int8")
    output.loc[left.eq(-1) & right.eq(-1)] = -1
    output.loc[left.eq(1) & right.eq(1)] = 1
    return output


def vote_regime(votes: np.ndarray, *, required: int) -> np.ndarray:
    if votes.ndim != 2 or required < 1 or required > votes.shape[1]:
        raise ValueError("Invalid coherent-regime vote contract.")
    output = np.zeros(votes.shape[0], dtype=np.int8)
    output[(votes == -1).sum(axis=1) >= required] = -1
    output[(votes == 1).sum(axis=1) >= required] = 1
    return output


def attach_pair_regimes(
    events: DataFrame,
    *,
    base: DataFrame,
    state: DataFrame,
) -> tuple[DataFrame, int]:
    actual_index = events["actual_base_index"].to_numpy(dtype=np.int64)
    control_index = events["control_base_index"].to_numpy(dtype=np.int64)
    if (
        actual_index.min(initial=0) < 0
        or control_index.min(initial=0) < 0
        or actual_index.max(initial=-1) >= len(base)
        or control_index.max(initial=-1) >= len(base)
    ):
        raise ValueError("G2F source base index is outside the causal market frame.")
    base_dates = normalize_dates(base["date"])
    actual_dates = normalize_dates(events["actual_event_time"])
    control_dates = normalize_dates(events["control_event_time"])
    violations = int((base_dates.iloc[actual_index].to_numpy() != actual_dates.to_numpy()).sum())
    violations += int(
        (base_dates.iloc[control_index].to_numpy() != control_dates.to_numpy()).sum()
    )
    output = events.copy()
    raw_columns = sorted(
        {
            column
            for lens in LENSES
            for column in lens.raw_state_columns
        }
    )
    for column in raw_columns:
        values = pd.to_numeric(state[column], errors="coerce").to_numpy(dtype=float)
        output[f"actual_state__{column}"] = values[actual_index]
        output[f"control_state__{column}"] = values[control_index]
    for lens in LENSES:
        values = state[f"regime__{lens.name}"].to_numpy(dtype=np.int8)
        actual = values[actual_index]
        control = values[control_index]
        agreed = (actual == control) & np.isin(actual, (-1, 1))
        output[f"actual_regime__{lens.name}"] = actual
        output[f"control_regime__{lens.name}"] = control
        output[f"regime_agreement__{lens.name}"] = agreed
        output[f"pair_regime__{lens.name}"] = np.where(agreed, actual, 0).astype(
            np.int8
        )

    paths = future_path_matrices(base, max_horizon=24)
    pre_pressure = pd.to_numeric(
        base["pre_pressure_mean_24"], errors="coerce"
    ).to_numpy(dtype=float)
    pressure = DataFrame(paths["pressure"])
    for horizon in (1, 4, 24):
        changed = pressure.iloc[:, :horizon].mean(axis=1).to_numpy(dtype=float)
        changed = np.abs(changed - pre_pressure)
        actual_value = changed[actual_index]
        control_value = changed[control_index]
        outcome = f"pressure_change_abs_h{horizon}"
        output[f"actual__{outcome}"] = actual_value
        output[f"control__{outcome}"] = control_value
        output[f"delta__{outcome}"] = actual_value - control_value
    return output, violations


def pair_period_interactions(events: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    group_columns = [
        "route_id",
        "actual_scope",
        "control",
        "pair",
        "period",
    ]
    for lens in LENSES:
        regime_column = f"pair_regime__{lens.name}"
        selected = events.loc[events[regime_column].isin((-1, 1))]
        for key, group in selected.groupby(
            group_columns,
            observed=True,
            sort=False,
        ):
            for outcome in OUTCOMES:
                values = pd.to_numeric(
                    group[f"delta__{outcome}"], errors="coerce"
                )
                low = values.loc[group[regime_column].eq(-1) & values.notna()]
                high = values.loc[group[regime_column].eq(1) & values.notna()]
                if low.empty or high.empty:
                    continue
                balance = continuous_state_balance(group, lens)
                rows.append(
                    {
                        "route_id": key[0],
                        "actual_scope": key[1],
                        "control": key[2],
                        "pair": key[3],
                        "period": key[4],
                        "regime_lens": lens.name,
                        "low_state_label": lens.low_label,
                        "high_state_label": lens.high_label,
                        "outcome": outcome,
                        "low_state_independent_pairs": len(low),
                        "high_state_independent_pairs": len(high),
                        "pair_period_coverage_eligible": (
                            len(low) >= MIN_PAIR_STATE_ROWS
                            and len(high) >= MIN_PAIR_STATE_ROWS
                        ),
                        "low_state_delta_mean": float(low.mean()),
                        "high_state_delta_mean": float(high.mean()),
                        "interaction_high_minus_low": float(high.mean() - low.mean()),
                        "low_state_positive_fraction": float(low.gt(0.0).mean()),
                        "high_state_positive_fraction": float(high.gt(0.0).mean()),
                        "max_absolute_state_smd": balance["max_absolute_smd"],
                        "median_absolute_state_smd": balance[
                            "median_absolute_smd"
                        ],
                        "state_features_scored": balance["features_scored"],
                        "direction_prediction": False,
                        "profit_optimization": False,
                    }
                )
    return DataFrame(rows)


def continuous_state_balance(group: DataFrame, lens: RegimeLens) -> dict[str, Any]:
    values: list[float] = []
    for column in lens.raw_state_columns:
        actual = pd.to_numeric(
            group[f"actual_state__{column}"], errors="coerce"
        ).to_numpy(dtype=float)
        control = pd.to_numeric(
            group[f"control_state__{column}"], errors="coerce"
        ).to_numpy(dtype=float)
        valid = np.isfinite(actual) & np.isfinite(control)
        if valid.sum() < 3:
            continue
        left = actual[valid]
        right = control[valid]
        pooled = np.sqrt((np.var(left) + np.var(right)) / 2.0)
        if np.isfinite(pooled) and pooled > 1e-12:
            values.append(float((np.mean(left) - np.mean(right)) / pooled))
    absolute = np.abs(values)
    return {
        "features_scored": len(values),
        "max_absolute_smd": float(np.max(absolute)) if len(absolute) else np.nan,
        "median_absolute_smd": (
            float(np.median(absolute)) if len(absolute) else np.nan
        ),
    }


def cohort_period_interactions(pair_period: DataFrame) -> DataFrame:
    if pair_period.empty:
        return DataFrame()
    eligible = pair_period.loc[pair_period["pair_period_coverage_eligible"]].copy()
    rows: list[dict[str, Any]] = []
    group_columns = [
        "route_id",
        "actual_scope",
        "control",
        "period",
        "regime_lens",
        "low_state_label",
        "high_state_label",
        "outcome",
    ]
    for key, group in eligible.groupby(
        group_columns,
        observed=True,
        sort=False,
    ):
        interactions = pd.to_numeric(
            group["interaction_high_minus_low"], errors="coerce"
        )
        low_count = int(group["low_state_independent_pairs"].sum())
        high_count = int(group["high_state_independent_pairs"].sum())
        coins = sorted(group["pair"].astype(str).unique())
        maximum = pd.to_numeric(
            group["max_absolute_state_smd"], errors="coerce"
        ).max()
        median = pd.to_numeric(
            group["median_absolute_state_smd"], errors="coerce"
        ).median()
        coverage = (
            len(coins) >= MIN_COHORT_COINS
            and low_count >= MIN_COHORT_STATE_ROWS
            and high_count >= MIN_COHORT_STATE_ROWS
        )
        balance = (
            np.isfinite(maximum)
            and np.isfinite(median)
            and maximum <= MAX_STATE_SMD
            and median <= MAX_MEDIAN_STATE_SMD
        )
        low_weight = pd.to_numeric(
            group["low_state_independent_pairs"], errors="coerce"
        )
        high_weight = pd.to_numeric(
            group["high_state_independent_pairs"], errors="coerce"
        )
        pooled_low = float(
            np.average(group["low_state_delta_mean"], weights=low_weight)
        )
        pooled_high = float(
            np.average(group["high_state_delta_mean"], weights=high_weight)
        )
        rows.append(
            {
                "route_id": key[0],
                "actual_scope": key[1],
                "control": key[2],
                "period": key[3],
                "regime_lens": key[4],
                "low_state_label": key[5],
                "high_state_label": key[6],
                "outcome": key[7],
                "eligible_coin_count": len(coins),
                "eligible_coins": ";".join(coins),
                "low_state_independent_pairs": low_count,
                "high_state_independent_pairs": high_count,
                "coverage_gate_passed": coverage,
                "state_balance_usable": balance,
                "evidence_eligible": coverage and balance,
                "equal_coin_interaction_median": float(interactions.median()),
                "equal_coin_interaction_q25": float(interactions.quantile(0.25)),
                "equal_coin_interaction_q75": float(interactions.quantile(0.75)),
                "equal_coin_positive_fraction": float(interactions.gt(0.0).mean()),
                "pooled_low_state_delta_mean": pooled_low,
                "pooled_high_state_delta_mean": pooled_high,
                "pooled_interaction_high_minus_low": pooled_high - pooled_low,
                "max_absolute_state_smd": float(maximum),
                "median_absolute_state_smd": float(median),
                "direction_prediction": False,
                "profit_optimization": False,
            }
        )
    return DataFrame(rows)


def leave_one_coin_out_interactions(pair_period: DataFrame) -> DataFrame:
    if pair_period.empty:
        return DataFrame()
    eligible = pair_period.loc[pair_period["pair_period_coverage_eligible"]].copy()
    rows: list[dict[str, Any]] = []
    group_columns = [
        "route_id",
        "actual_scope",
        "control",
        "period",
        "regime_lens",
        "low_state_label",
        "high_state_label",
        "outcome",
    ]
    for key, group in eligible.groupby(
        group_columns,
        observed=True,
        sort=False,
    ):
        ordered = group.sort_values("pair", kind="stable").reset_index(drop=True)
        if len(ordered) < 2:
            continue
        for omitted_index, omitted in enumerate(ordered["pair"].astype(str)):
            retained = ordered.drop(index=omitted_index)
            interactions = pd.to_numeric(
                retained["interaction_high_minus_low"], errors="coerce"
            )
            low_count = int(retained["low_state_independent_pairs"].sum())
            high_count = int(retained["high_state_independent_pairs"].sum())
            rows.append(
                {
                    "route_id": key[0],
                    "actual_scope": key[1],
                    "control": key[2],
                    "period": key[3],
                    "regime_lens": key[4],
                    "low_state_label": key[5],
                    "high_state_label": key[6],
                    "outcome": key[7],
                    "omitted_pair": omitted,
                    "remaining_coin_count": len(retained),
                    "remaining_low_state_pairs": low_count,
                    "remaining_high_state_pairs": high_count,
                    "coverage_gate_passed": (
                        len(retained) >= MIN_COHORT_COINS
                        and low_count >= MIN_COHORT_STATE_ROWS
                        and high_count >= MIN_COHORT_STATE_ROWS
                    ),
                    "equal_coin_interaction_median": float(interactions.median()),
                    "equal_coin_positive_fraction": float(
                        interactions.gt(0.0).mean()
                    ),
                }
            )
    return DataFrame(rows)


def integrity_record(
    *,
    source: DataFrame,
    enriched: DataFrame,
    pair_period: DataFrame,
    time_violations: int,
) -> dict[str, Any]:
    missing_outcomes = [
        outcome
        for outcome in OUTCOMES
        if f"delta__{outcome}" not in enriched.columns
    ]
    lens_agreement_counts = {
        lens.name: int(enriched[f"regime_agreement__{lens.name}"].sum())
        for lens in LENSES
    }
    return {
        "created_at_utc": utc_now(),
        "passed": (
            len(source) == len(enriched)
            and time_violations == 0
            and not missing_outcomes
            and not pair_period.empty
            and all(value > 0 for value in lens_agreement_counts.values())
        ),
        "source_event_pairs": len(source),
        "enriched_event_pairs": len(enriched),
        "source_pair_count": int(enriched["pair"].nunique()),
        "source_route_count": int(enriched["route_id"].nunique()),
        "source_control_count": int(enriched["control"].nunique()),
        "event_time_to_base_index_violations": time_violations,
        "missing_outcomes": missing_outcomes,
        "regime_agreement_counts": lens_agreement_counts,
        "pair_period_rows": len(pair_period),
        "causal_quantile_lookback_hours": CAUSAL_QUANTILE_LOOKBACK_HOURS,
        "causal_quantile_minimum_hours": CAUSAL_QUANTILE_MINIMUM_HOURS,
        "current_contact_candle_in_state_calculation": False,
        "direction_prediction": False,
        "profit_optimization": False,
    }


def stable_json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
