"""Run Generation 18's frozen direct confirmation siblings A through D."""

from __future__ import annotations

# Bind numerical pools before pandas/numpy imports.
# ruff: noqa: E402
import argparse
import hashlib
import json
import os
import sys
from collections.abc import Iterable, Sequence
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
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation16_direct_attribution as g16d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation17_level_source_atlas as g17l,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation18_freeze as g18z,
)


DEFAULT_RUN_ID = "g18_direct_confirmation_20260823a"
RECORD_ROOT = g18z.OUTPUT_ROOT / "direct_confirmation"
CONTEXT_REGISTRY_PATH = RECORD_ROOT / "g18_context_registry_freeze.json"
ATLAS_EVENT_ROOT = g17l.ARTIFACT_ROOT / g17l.DEFAULT_RUN_ID / "pair_events"
HORIZONS = g18z.HORIZONS_HOURS
MIN_PAIR_ROWS = 10
BOOTSTRAP_SAMPLES = 2000
RATIONAL_FAMILIES = (
    "adaptive_volume_profile_nodes",
    "rolling_vwap_deviation_bands",
    "donchian_boundaries",
    "weekly_pivot_grid",
)
CONFIRMATION_FAMILIES = (
    "adaptive_volume_profile_nodes",
    "donchian_boundaries",
)
LEVEL_CONFIRMATION_NAMES = frozenset(
    (*g18z.VP_CONFIRMATION_SPECS, *g18z.DONCHIAN_CONFIRMATION_SPECS)
)

RAW_COLUMNS = (
    "cohort",
    "pair",
    "control",
    "event_time",
    "level_family",
    "level_name",
    "pre_distance_atr",
    "zone_half_width_atr",
    "contact_range_ratio",
    "contact_volume_ratio",
    "contact_pressure_change",
    *(
        column
        for horizon in HORIZONS
        for column in (
            f"abs_excursion_atr_h{horizon}",
            f"away_excursion_atr_h{horizon}",
            f"through_excursion_atr_h{horizon}",
            f"volume_ratio_h{horizon}",
            f"dwell_fraction_h{horizon}",
            f"crossings_h{horizon}",
        )
    ),
)

CONTEXT_DEFINITIONS = (
    {
        "context_id": "local_volume_range_activity",
        "high_state": "high",
        "low_state": "low",
        "definition": (
            "mean log of prior completed-candle volume and range ratios: high >= "
            "log(1.25), low <= log(0.80)"
        ),
    },
    {
        "context_id": "local_volatility",
        "high_state": "high",
        "low_state": "low",
        "definition": (
            "prior ATR/price divided by its trailing 720h median: high >= 1.25, "
            "low <= 0.80"
        ),
    },
    {
        "context_id": "bollinger_compression",
        "high_state": "expanded",
        "low_state": "compressed",
        "definition": (
            "prior 20h Bollinger width divided by its trailing 720h median: expanded "
            ">= 1.25, compressed <= 0.75"
        ),
    },
    {
        "context_id": "rsi_extreme",
        "high_state": "extreme",
        "low_state": "neutral",
        "definition": "prior RSI14 <= 35 or >= 65 versus prior RSI14 between 45 and 55",
    },
    {
        "context_id": "adx_strength",
        "high_state": "strong",
        "low_state": "weak",
        "definition": "prior ADX14 >= 25 versus <= 18",
    },
    {
        "context_id": "wider_crypto_activity",
        "high_state": "high",
        "low_state": "low",
        "definition": (
            "equal-weight normal-coin median absolute prior 8h return divided by its "
            "trailing 720h median: high >= 1.25, low <= 0.80"
        ),
    },
    {
        "context_id": "crypto_breadth_imbalance",
        "high_state": "one_sided",
        "low_state": "balanced",
        "definition": (
            "absolute distance of prior 8h positive-coin breadth from 50%: one-sided "
            ">= 0.30, balanced <= 0.10"
        ),
    },
    {
        "context_id": "crypto_dispersion",
        "high_state": "high",
        "low_state": "low",
        "definition": (
            "cross-coin dispersion of prior 8h returns divided by its trailing 720h "
            "median: high >= 1.25, low <= 0.80"
        ),
    },
    {
        "context_id": "local_wider_trend_relationship",
        "high_state": "agreement",
        "low_state": "conflict",
        "definition": (
            "sign of the coin's prior 8h return agrees with or opposes the equal-weight "
            "normal-coin median prior 8h return; zero signs are unavailable"
        ),
    },
    {
        "context_id": "source_ready_orderbook",
        "high_state": "current_state",
        "low_state": "stale_or_shuffled",
        "definition": (
            "only timestamp-ready orderbook rows in both frozen holdout halves; park "
            "when that common surface is absent"
        ),
    },
)


def artifact(path: Path) -> dict[str, Any]:
    return g18z.artifact(path)


def load_freeze() -> dict[str, Any]:
    frozen = json.loads(g18z.FREEZE_PATH.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_generation18_outcomes":
        raise ValueError("Generation 18 branch layer is not frozen.")
    return frozen


def freeze_context_registry() -> dict[str, Any]:
    frozen = load_freeze()
    branch = next(
        item
        for item in frozen["branches"]
        if item["branch_id"] == "g18d_context_and_market_regimes"
    )
    registry = {
        "schema_version": 1,
        "generation": 18,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation18_context_outcomes",
        "branch_id": branch["branch_id"],
        "definitions": list(CONTEXT_DEFINITIONS),
        "maximum_active_blocks_per_interaction": 2,
        "outcome_selected_thresholds": False,
        "future_signed_direction_used": False,
        "profit_used": False,
        "source_contracts": {"generation18_freeze": artifact(g18z.FREEZE_PATH)},
    }
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    if CONTEXT_REGISTRY_PATH.is_file():
        existing = json.loads(CONTEXT_REGISTRY_PATH.read_text(encoding="utf-8"))
        left = {key: value for key, value in existing.items() if key != "created_at_utc"}
        right = {key: value for key, value in registry.items() if key != "created_at_utc"}
        if left != right:
            raise ValueError("Generation 18 context registry changed after its freeze.")
        return existing
    g0.atomic_write_json(registry, CONTEXT_REGISTRY_PATH)
    return registry


def assign_confirmation_period(dates: Series, cohort: str) -> Series:
    output = pd.Series("outside_g18_confirmation", index=dates.index, dtype="object")
    for period in g18z.CONFIRMATION_PERIODS[cohort]:
        start = pd.Timestamp(period["start_utc"])
        stop = pd.Timestamp(period["end_utc_exclusive"])
        output.loc[(dates >= start) & (dates < stop)] = period["id"]
    return output


def confirmation_period_ids(market_scope: str) -> set[str]:
    cohort = "meme" if market_scope == "top_ten_memes" else "normal"
    return {item["id"] for item in g18z.CONFIRMATION_PERIODS[cohort]}


def read_holdout(path: Path) -> DataFrame:
    frame = pd.read_parquet(path, columns=list(RAW_COLUMNS))
    if frame.empty:
        return frame
    cohort_values = frame["cohort"].dropna().astype(str).unique()
    if len(cohort_values) != 1:
        raise ValueError(f"Unexpected cohort values in {path}: {cohort_values}")
    cohort = cohort_values[0]
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True, errors="raise")
    frame["g18_period"] = assign_confirmation_period(frame["event_time"], cohort)
    return frame.loc[frame["g18_period"].ne("outside_g18_confirmation")].copy()


def add_reaction_metrics(frame: DataFrame) -> None:
    threshold = pd.to_numeric(frame["zone_half_width_atr"], errors="coerce").clip(
        lower=0.5
    )
    for horizon in HORIZONS:
        crossings = pd.to_numeric(frame[f"crossings_h{horizon}"], errors="coerce")
        away = pd.to_numeric(frame[f"away_excursion_atr_h{horizon}"], errors="coerce")
        through = pd.to_numeric(
            frame[f"through_excursion_atr_h{horizon}"], errors="coerce"
        )
        excursion = pd.to_numeric(
            frame[f"abs_excursion_atr_h{horizon}"], errors="coerce"
        )
        volume = pd.to_numeric(frame[f"volume_ratio_h{horizon}"], errors="coerce")
        frame[f"metric__crossings_h{horizon}"] = crossings
        frame[f"metric__volume_ratio_h{horizon}"] = volume
        frame[f"metric__dwell_fraction_h{horizon}"] = pd.to_numeric(
            frame[f"dwell_fraction_h{horizon}"], errors="coerce"
        )
        frame[f"metric__unsigned_reaction_h{horizon}"] = binary_metric(
            (excursion >= threshold) & (volume >= 1.25), excursion, volume, threshold
        )
        frame[f"metric__any_recross_h{horizon}"] = binary_metric(
            crossings >= 1.0, crossings
        )
        frame[f"metric__repeated_recross_h{horizon}"] = binary_metric(
            crossings >= 2.0, crossings
        )
        frame[f"metric__two_sided_traversal_h{horizon}"] = binary_metric(
            (away >= threshold) & (through >= threshold), away, through, threshold
        )
        frame[f"metric__one_sided_rejection_h{horizon}"] = binary_metric(
            (away >= threshold) & (through < threshold), away, through, threshold
        )
        frame[f"metric__one_sided_breakthrough_h{horizon}"] = binary_metric(
            (through >= threshold) & (away < threshold), away, through, threshold
        )


def binary_metric(mask: Series, *required: Series) -> Series:
    output = mask.astype(float)
    unavailable = pd.Series(False, index=output.index)
    for values in required:
        unavailable |= pd.to_numeric(values, errors="coerce").isna()
    output.loc[unavailable] = np.nan
    return output


def nearest_anchor(frame: DataFrame, keys: Sequence[str]) -> DataFrame:
    selected = frame.copy()
    selected["_anchor_distance"] = pd.to_numeric(
        selected["pre_distance_atr"], errors="coerce"
    ).abs()
    return (
        selected.sort_values([*keys, "_anchor_distance", "level_name"], kind="stable")
        .drop_duplicates(list(keys), keep="first")
        .drop(columns="_anchor_distance")
        .reset_index(drop=True)
    )


def rank_activity_residual(frame: DataFrame, value_column: str) -> tuple[np.ndarray, np.ndarray]:
    activity_columns = (
        "contact_volume_ratio",
        "contact_range_ratio",
        "contact_pressure_change",
    )
    work = frame[[value_column, *activity_columns]].apply(pd.to_numeric, errors="coerce")
    valid = work.notna().all(axis=1)
    work = work.loc[valid].copy()
    if len(work) < MIN_PAIR_ROWS:
        return np.array([]), np.array([])
    ranks = DataFrame(index=work.index)
    ranks["volume"] = work["contact_volume_ratio"].rank(pct=True)
    ranks["range"] = work["contact_range_ratio"].rank(pct=True)
    ranks["pressure"] = work["contact_pressure_change"].abs().rank(pct=True)
    activity = ranks.mean(axis=1)
    strata = np.minimum((activity * 5).astype(int), 4)
    outcome = pd.to_numeric(work[value_column], errors="coerce")
    outcome_residual = outcome - outcome.groupby(strata).transform("mean")
    return work.index.to_numpy(), outcome_residual.to_numpy(dtype=float)


def spearman_effect(frame: DataFrame, component: str, target: str) -> tuple[float, int]:
    indexes, outcome = rank_activity_residual(frame, target)
    if not len(indexes):
        return np.nan, 0
    values = pd.to_numeric(frame.loc[indexes, component], errors="coerce")
    component_residual = values
    valid = np.isfinite(component_residual.to_numpy()) & np.isfinite(outcome)
    if valid.sum() < MIN_PAIR_ROWS or component_residual.loc[valid].nunique() < 2:
        return np.nan, int(valid.sum())
    left = Series(component_residual.to_numpy()[valid]).rank(method="average")
    right = Series(outcome[valid]).rank(method="average")
    return float(left.corr(right)), int(valid.sum())


def density_pair_effects(events: DataFrame) -> DataFrame:
    rational = events.loc[events["level_family"].isin(RATIONAL_FAMILIES)].copy()
    keys = ["cohort", "pair", "g18_period", "control", "event_time"]
    grouped = rational.groupby(keys, observed=True, sort=False)
    counts = grouped.agg(
        density_independent_families=("level_family", "nunique"),
        density_convergent_variants=("level_name", "nunique"),
    ).reset_index()
    anchor = nearest_anchor(rational, keys).merge(counts, on=keys, validate="one_to_one")
    anchor["density_cross_family_flag"] = anchor[
        "density_independent_families"
    ].ge(2).astype(float)
    anchor["density_proximity_score"] = -pd.to_numeric(
        anchor["pre_distance_atr"], errors="coerce"
    ).abs()
    anchor["density_zone_width"] = pd.to_numeric(
        anchor["zone_half_width_atr"], errors="coerce"
    )
    components = (
        "density_independent_families",
        "density_convergent_variants",
        "density_cross_family_flag",
        "density_proximity_score",
        "density_zone_width",
    )
    rows: list[dict[str, Any]] = []
    for key, cell in anchor.groupby(
        ["cohort", "pair", "g18_period", "control"], observed=True, sort=False
    ):
        for component in components:
            for horizon in HORIZONS:
                target = f"metric__crossings_h{horizon}"
                effect, count = spearman_effect(cell, component, target)
                rows.append(
                    {
                        "cohort": key[0],
                        "pair": key[1],
                        "period": key[2],
                        "control": key[3],
                        "component": component,
                        "metric": "crossings",
                        "horizon_hours": horizon,
                        "partial_spearman": effect,
                        "event_rows": count,
                        "component_unique_values": int(cell[component].nunique(dropna=True)),
                    }
                )
    return DataFrame.from_records(rows)


def scoped_level_events(events: DataFrame, *, semantics: bool) -> DataFrame:
    keys = ["cohort", "pair", "g18_period", "control", "event_time"]
    selected = events.loc[events["level_family"].isin(CONFIRMATION_FAMILIES)].copy()
    frames: list[DataFrame] = []
    family = nearest_anchor(selected, [*keys, "level_family"])
    family["scope_kind"] = "family_any_level"
    family["scope_value"] = family["level_family"]
    frames.append(family)
    combined = nearest_anchor(selected, keys)
    combined["scope_kind"] = "retained_family_surface"
    combined["scope_value"] = "adaptive_vp_or_donchian"
    frames.append(combined)
    if not semantics:
        exact = selected.loc[selected["level_name"].isin(LEVEL_CONFIRMATION_NAMES)].copy()
        exact["scope_kind"] = "single_level"
        exact["scope_value"] = exact["level_name"]
        frames.append(exact)
        rivals = events.loc[events["level_family"].isin(g18z.RIVAL_FAMILIES)].copy()
        rival_family = nearest_anchor(rivals, [*keys, "level_family"])
        rival_family["scope_kind"] = "rival_family"
        rival_family["scope_value"] = rival_family["level_family"]
        frames.append(rival_family)
    return pd.concat(frames, ignore_index=True, sort=False)


def mean_summary(
    frame: DataFrame,
    *,
    metrics: Iterable[str],
    controls: set[str],
) -> DataFrame:
    selected = frame.loc[frame["control"].isin({"actual", *controls})]
    keys = [
        "cohort",
        "pair",
        "g18_period",
        "scope_kind",
        "scope_value",
        "control",
    ]
    rows: list[DataFrame] = []
    for metric in metrics:
        for horizon in HORIZONS:
            column = f"metric__{metric}_h{horizon}"
            grouped = (
                selected[keys + [column]]
                .dropna(subset=[column])
                .groupby(keys, observed=True, sort=False)[column]
                .agg(["mean", "size"])
                .reset_index()
                .rename(columns={"mean": "outcome_mean", "size": "event_rows"})
            )
            grouped["metric"] = metric
            grouped["horizon_hours"] = horizon
            rows.append(grouped)
    return pd.concat(rows, ignore_index=True, sort=False)


def paired_contrasts(summary: DataFrame, controls: Sequence[str]) -> DataFrame:
    keys = [
        "cohort",
        "pair",
        "g18_period",
        "scope_kind",
        "scope_value",
        "metric",
        "horizon_hours",
    ]
    actual = summary.loc[summary["control"].eq("actual")].drop(columns="control")
    rows: list[DataFrame] = []
    for control in controls:
        baseline = summary.loc[summary["control"].eq(control)].drop(columns="control")
        merged = actual.merge(
            baseline,
            on=keys,
            how="inner",
            suffixes=("_actual", "_control"),
            validate="one_to_one",
        )
        merged["comparison"] = control
        merged["difference"] = merged["outcome_mean_actual"] - merged["outcome_mean_control"]
        merged["eligible_pair"] = (
            merged["event_rows_actual"].ge(MIN_PAIR_ROWS)
            & merged["event_rows_control"].ge(MIN_PAIR_ROWS)
        )
        rows.append(merged)
    return pd.concat(rows, ignore_index=True, sort=False)


def density_contrasts(effects: DataFrame) -> DataFrame:
    keys = ["cohort", "pair", "period", "component", "metric", "horizon_hours"]
    actual = effects.loc[effects["control"].eq("actual")].drop(columns="control")
    rows: list[DataFrame] = []
    for control in g18z.CONTROLS:
        baseline = effects.loc[effects["control"].eq(control)].drop(columns="control")
        merged = actual.merge(
            baseline,
            on=keys,
            how="inner",
            suffixes=("_actual", "_control"),
            validate="one_to_one",
        )
        merged["comparison"] = control
        merged["difference"] = (
            merged["partial_spearman_actual"] - merged["partial_spearman_control"]
        )
        merged["eligible_pair"] = (
            merged["event_rows_actual"].ge(MIN_PAIR_ROWS)
            & merged["event_rows_control"].ge(MIN_PAIR_ROWS)
            & merged["component_unique_values_actual"].ge(2)
            & merged["component_unique_values_control"].ge(2)
        )
        rows.append(merged)
    return pd.concat(rows, ignore_index=True, sort=False)


def bootstrap_coins(values: np.ndarray, key: str) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if not len(values):
        return np.nan, np.nan
    seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(BOOTSTRAP_SAMPLES, len(values)), replace=True).mean(
        axis=1
    )
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def expand_market_scopes(frame: DataFrame) -> DataFrame:
    rows: list[DataFrame] = []
    for (cohort, pair), cell in frame.groupby(["cohort", "pair"], observed=True, sort=False):
        for market_scope in g16d.group_for_pair(str(cohort), str(pair)):
            copy = cell.copy()
            copy["market_scope"] = market_scope
            rows.append(copy)
    return pd.concat(rows, ignore_index=True, sort=False)


def period_scores(contrasts: DataFrame, question_keys: Sequence[str]) -> DataFrame:
    source = contrasts.loc[contrasts["eligible_pair"]].copy()
    if "period" not in source.columns and "g18_period" in source.columns:
        source.rename(columns={"g18_period": "period"}, inplace=True)
    expanded = expand_market_scopes(source)
    keys = [*question_keys, "market_scope", "period", "comparison"]
    rows: list[dict[str, Any]] = []
    for key, cell in expanded.groupby(keys, observed=True, sort=False):
        values = pd.to_numeric(cell["difference"], errors="coerce").dropna()
        lower, upper = bootstrap_coins(values.to_numpy(), "|".join(map(str, key)))
        absolute = values.abs()
        dominance = (
            float(absolute.max() / absolute.sum())
            if len(absolute) and absolute.sum() > 0.0
            else np.nan
        )
        market_scope = str(key[-3])
        required = g16d.GROUP_MIN_COINS[market_scope]
        fraction = float(values.gt(0).mean()) if len(values) else np.nan
        point = bool(len(values) >= required and values.mean() > 0 and fraction >= 0.70)
        strict = bool(
            point
            and np.isfinite(lower)
            and lower > 0
            and (
                market_scope == "btc_separate"
                or (
                    np.isfinite(dominance)
                    and dominance <= g16d.MAX_COIN_ABSOLUTE_SHARE
                )
            )
        )
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "eligible_coins": len(values),
                "required_coins": required,
                "positive_coins": int(values.gt(0).sum()),
                "positive_coin_fraction": fraction,
                "equal_coin_difference": float(values.mean()) if len(values) else np.nan,
                "bootstrap_lower_95": lower,
                "bootstrap_upper_95": upper,
                "maximum_one_coin_absolute_share": dominance,
                "point_period_pass": point,
                "strict_period_pass": strict,
            }
        )
    return DataFrame.from_records(rows).sort_values(keys).reset_index(drop=True)


def whole_decisions(
    scores: DataFrame,
    question_keys: Sequence[str],
    controls: Sequence[str],
) -> DataFrame:
    keys = [*question_keys, "market_scope"]
    rows: list[dict[str, Any]] = []
    for key, cell in scores.groupby(keys, observed=True, sort=False):
        market_scope = str(key[-1])
        periods = confirmation_period_ids(market_scope)
        selected = cell.loc[cell["period"].isin(periods)]
        complete = bool(
            len(selected) == len(periods) * len(controls)
            and set(selected["period"]) == periods
            and set(selected["comparison"]) == set(controls)
        )
        point = bool(complete and selected["point_period_pass"].astype(bool).all())
        strict = bool(complete and selected["strict_period_pass"].astype(bool).all())
        rows.append(
            {
                **dict(zip(keys, key, strict=True)),
                "status": (
                    "strict_holdout_confirmation"
                    if strict
                    else "point_holdout_confirmation"
                    if point
                    else "not_retained"
                ),
                "complete_control_period_ladder": complete,
                "minimum_equal_coin_difference": (
                    float(selected["equal_coin_difference"].min())
                    if len(selected)
                    else np.nan
                ),
                "minimum_bootstrap_lower": (
                    float(selected["bootstrap_lower_95"].min())
                    if len(selected)
                    else np.nan
                ),
            }
        )
    return DataFrame.from_records(rows).sort_values(keys).reset_index(drop=True)


def ratio_to_trailing_median(values: Series, window: int = 720) -> Series:
    numeric = pd.to_numeric(values, errors="coerce")
    baseline = numeric.rolling(window, min_periods=168).median().shift(1)
    return numeric.div(baseline.replace(0.0, np.nan))


def rsi(close: Series, period: int = 14) -> Series:
    change = pd.to_numeric(close, errors="coerce").diff()
    gain = change.clip(lower=0.0).ewm(alpha=1.0 / period, adjust=False).mean()
    loss = (-change.clip(upper=0.0)).ewm(alpha=1.0 / period, adjust=False).mean()
    relative = gain.div(loss.replace(0.0, np.nan))
    return 100.0 - 100.0 / (1.0 + relative)


def adx(frame: DataFrame, period: int = 14) -> Series:
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    up = high.diff()
    down = -low.diff()
    plus_dm = Series(np.where((up > down) & (up > 0), up, 0.0), index=frame.index)
    minus_dm = Series(np.where((down > up) & (down > 0), down, 0.0), index=frame.index)
    true_range = pd.concat(
        [(high - low).abs(), (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    smoothed_range = true_range.ewm(alpha=1.0 / period, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1.0 / period, adjust=False).mean().div(
        smoothed_range.replace(0.0, np.nan)
    )
    minus_di = 100.0 * minus_dm.ewm(alpha=1.0 / period, adjust=False).mean().div(
        smoothed_range.replace(0.0, np.nan)
    )
    dx = 100.0 * (plus_di - minus_di).abs().div(
        (plus_di + minus_di).replace(0.0, np.nan)
    )
    return dx.ewm(alpha=1.0 / period, adjust=False).mean()


def build_wider_context() -> DataFrame:
    manifest = json.loads(g0.DEFAULT_MANIFEST.read_text(encoding="utf-8"))
    returns: list[Series] = []
    for pair in manifest["data"]["pairs"]:
        source = g0.load_ohlcv(g0.ohlcv_path(str(pair), "1h"))
        source = source.sort_values("date").drop_duplicates("date")
        close = pd.to_numeric(source["close"], errors="coerce")
        prior_return = close.shift(1).div(close.shift(9)).sub(1.0)
        returns.append(Series(prior_return.to_numpy(), index=source["date"], name=str(pair)))
    matrix = pd.concat(returns, axis=1).sort_index()
    wider = DataFrame(index=matrix.index)
    wider["wider_median_return_8h"] = matrix.median(axis=1)
    wider["wider_median_absolute_return_8h"] = matrix.abs().median(axis=1)
    wider["wider_breadth_positive_8h"] = matrix.gt(0.0).mean(axis=1)
    wider["wider_dispersion_8h"] = matrix.std(axis=1, ddof=0)
    wider["wider_activity_ratio"] = ratio_to_trailing_median(
        wider["wider_median_absolute_return_8h"]
    )
    wider["wider_dispersion_ratio"] = ratio_to_trailing_median(
        wider["wider_dispersion_8h"]
    )
    wider["wider_breadth_imbalance"] = (
        wider["wider_breadth_positive_8h"] - 0.5
    ).abs()
    return wider.reset_index(names="event_time")


def build_local_context(pair: str, wider: DataFrame) -> DataFrame:
    frame = g0.load_ohlcv(g0.ohlcv_path(pair, "1h"))
    frame = frame.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce")
    candle_range = high - low
    prior_volume = volume.shift(1)
    prior_range = candle_range.shift(1)
    volume_baseline = volume.shift(2).rolling(24, min_periods=24).median()
    range_baseline = candle_range.shift(2).rolling(24, min_periods=24).median()
    volume_ratio = prior_volume.div(volume_baseline.replace(0.0, np.nan))
    range_ratio = prior_range.div(range_baseline.replace(0.0, np.nan))
    activity_score = (
        np.log(volume_ratio.clip(lower=1e-9)) + np.log(range_ratio.clip(lower=1e-9))
    ) / 2.0
    atr_fraction = g0.wilder_atr(frame, 14).shift(1).div(close.shift(1).abs())
    middle = close.rolling(20, min_periods=20).mean()
    deviation = close.rolling(20, min_periods=20).std(ddof=0)
    bb_width = (4.0 * deviation).div(middle.abs().replace(0.0, np.nan)).shift(1)
    local = DataFrame(
        {
            "event_time": frame["date"],
            "local_return_8h": close.shift(1).div(close.shift(9)).sub(1.0),
            "local_activity_score": activity_score,
            "local_volatility_ratio": ratio_to_trailing_median(atr_fraction),
            "local_bollinger_width_ratio": ratio_to_trailing_median(bb_width),
            "local_rsi14": rsi(close, 14).shift(1),
            "local_adx14": adx(frame, 14).shift(1),
        }
    )
    output = local.merge(wider, on="event_time", how="left", validate="one_to_one")
    return output


def context_state_frame(pair: str, wider: DataFrame) -> DataFrame:
    frame = build_local_context(pair, wider)
    states = DataFrame({"event_time": frame["event_time"]})
    states["state__local_volume_range_activity"] = np.select(
        [
            frame["local_activity_score"] >= np.log(1.25),
            frame["local_activity_score"] <= np.log(0.80),
        ],
        ["high", "low"],
        default="unavailable",
    )
    states["state__local_volatility"] = np.select(
        [frame["local_volatility_ratio"] >= 1.25, frame["local_volatility_ratio"] <= 0.80],
        ["high", "low"],
        default="unavailable",
    )
    states["state__bollinger_compression"] = np.select(
        [
            frame["local_bollinger_width_ratio"] >= 1.25,
            frame["local_bollinger_width_ratio"] <= 0.75,
        ],
        ["expanded", "compressed"],
        default="unavailable",
    )
    states["state__rsi_extreme"] = np.select(
        [
            (frame["local_rsi14"] <= 35.0) | (frame["local_rsi14"] >= 65.0),
            frame["local_rsi14"].between(45.0, 55.0, inclusive="both"),
        ],
        ["extreme", "neutral"],
        default="unavailable",
    )
    states["state__adx_strength"] = np.select(
        [frame["local_adx14"] >= 25.0, frame["local_adx14"] <= 18.0],
        ["strong", "weak"],
        default="unavailable",
    )
    states["state__wider_crypto_activity"] = np.select(
        [frame["wider_activity_ratio"] >= 1.25, frame["wider_activity_ratio"] <= 0.80],
        ["high", "low"],
        default="unavailable",
    )
    states["state__crypto_breadth_imbalance"] = np.select(
        [
            frame["wider_breadth_imbalance"] >= 0.30,
            frame["wider_breadth_imbalance"] <= 0.10,
        ],
        ["one_sided", "balanced"],
        default="unavailable",
    )
    states["state__crypto_dispersion"] = np.select(
        [frame["wider_dispersion_ratio"] >= 1.25, frame["wider_dispersion_ratio"] <= 0.80],
        ["high", "low"],
        default="unavailable",
    )
    local_sign = np.sign(frame["local_return_8h"])
    wider_sign = np.sign(frame["wider_median_return_8h"])
    states["state__local_wider_trend_relationship"] = np.select(
        [
            (local_sign != 0) & (local_sign == wider_sign),
            (local_sign != 0) & (wider_sign != 0) & (local_sign != wider_sign),
        ],
        ["agreement", "conflict"],
        default="unavailable",
    )
    return states


def context_pair_modulation(events: DataFrame, context: DataFrame) -> tuple[DataFrame, DataFrame]:
    controls = ("matched_random_time", "near_miss")
    scoped = scoped_level_events(events, semantics=True)
    scoped = scoped.loc[
        scoped["scope_kind"].eq("retained_family_surface")
        & scoped["control"].isin({"actual", *controls})
    ].merge(context, on="event_time", how="left", validate="many_to_one")
    definitions = {
        item["context_id"]: (item["high_state"], item["low_state"])
        for item in CONTEXT_DEFINITIONS
        if item["context_id"] != "source_ready_orderbook"
    }
    rows: list[DataFrame] = []
    coverage: list[dict[str, Any]] = []
    base_keys = ["cohort", "pair", "g18_period", "control"]
    for context_id, (high_state, low_state) in definitions.items():
        state_column = f"state__{context_id}"
        coverage.append(
            {
                "cohort": str(scoped["cohort"].iloc[0]) if len(scoped) else "unknown",
                "pair": str(scoped["pair"].iloc[0]) if len(scoped) else "unknown",
                "source": context_id,
                "eligible_rows": int(scoped[state_column].isin([high_state, low_state]).sum()),
                "status": (
                    "tested"
                    if scoped[state_column].isin([high_state, low_state]).any()
                    else "parked_no_rows"
                ),
            }
        )
        for metric in ("unsigned_reaction", "crossings", "volume_ratio"):
            for horizon in HORIZONS:
                column = f"metric__{metric}_h{horizon}"
                selected = scoped.loc[
                    scoped[state_column].isin([high_state, low_state]),
                    [*base_keys, state_column, column],
                ].dropna(subset=[column])
                grouped = (
                    selected.groupby(
                        [*base_keys, state_column], observed=True, sort=False
                    )[column]
                    .agg(["mean", "size"])
                    .reset_index()
                    .rename(
                        columns={
                            state_column: "context_state",
                            "mean": "outcome_mean",
                            "size": "event_rows",
                        }
                    )
                )
                grouped["context_id"] = context_id
                grouped["high_state"] = high_state
                grouped["low_state"] = low_state
                grouped["metric"] = metric
                grouped["horizon_hours"] = horizon
                rows.append(grouped)
    summary = pd.concat(rows, ignore_index=True, sort=False)
    key_columns = [
        "cohort",
        "pair",
        "g18_period",
        "context_id",
        "high_state",
        "low_state",
        "metric",
        "horizon_hours",
        "context_state",
    ]
    actual = summary.loc[summary["control"].eq("actual")].drop(columns="control")
    contrasts: list[DataFrame] = []
    for control in controls:
        baseline = summary.loc[summary["control"].eq(control)].drop(columns="control")
        merged = actual.merge(
            baseline,
            on=key_columns,
            suffixes=("_actual", "_control"),
            validate="one_to_one",
        )
        merged["level_effect"] = merged["outcome_mean_actual"] - merged["outcome_mean_control"]
        merged["minimum_state_rows"] = merged[
            ["event_rows_actual", "event_rows_control"]
        ].min(axis=1)
        pivot = merged.pivot(
            index=[column for column in key_columns if column != "context_state"],
            columns="context_state",
            values=["level_effect", "minimum_state_rows"],
        ).reset_index()
        pivot.columns = [
            "__".join(str(part) for part in column if str(part))
            if isinstance(column, tuple)
            else str(column)
            for column in pivot.columns
        ]
        records: list[dict[str, Any]] = []
        for row in pivot.to_dict("records"):
            high_state = str(row["high_state"])
            low_state = str(row["low_state"])
            high_effect = row.get(f"level_effect__{high_state}", np.nan)
            low_effect = row.get(f"level_effect__{low_state}", np.nan)
            high_rows = row.get(f"minimum_state_rows__{high_state}", 0)
            low_rows = row.get(f"minimum_state_rows__{low_state}", 0)
            records.append(
                {
                    **{
                        key: row[key]
                        for key in (
                            "cohort",
                            "pair",
                            "g18_period",
                            "context_id",
                            "metric",
                            "horizon_hours",
                        )
                    },
                    "comparison": control,
                    "high_state": high_state,
                    "low_state": low_state,
                    "high_state_level_effect": high_effect,
                    "low_state_level_effect": low_effect,
                    "difference": high_effect - low_effect,
                    "eligible_pair": bool(
                        np.isfinite(high_effect)
                        and np.isfinite(low_effect)
                        and high_rows >= MIN_PAIR_ROWS
                        and low_rows >= MIN_PAIR_ROWS
                    ),
                    "minimum_state_rows": min(high_rows, low_rows),
                }
            )
        contrasts.append(DataFrame.from_records(records))
    return pd.concat(contrasts, ignore_index=True, sort=False), DataFrame.from_records(coverage)


def output_paths(run_dir: Path) -> dict[str, Path]:
    return {
        "density_pair_effects": run_dir / "g18_density_pair_effects.csv",
        "density_contrasts": run_dir / "g18_density_control_contrasts.csv",
        "density_scores": run_dir / "g18_density_period_scores.csv",
        "density_decisions": run_dir / "g18_density_decisions.csv",
        "semantic_contrasts": run_dir / "g18_semantic_pair_contrasts.csv",
        "semantic_scores": run_dir / "g18_semantic_period_scores.csv",
        "semantic_decisions": run_dir / "g18_semantic_decisions.csv",
        "level_contrasts": run_dir / "g18_level_pair_contrasts.csv",
        "level_scores": run_dir / "g18_level_period_scores.csv",
        "level_decisions": run_dir / "g18_level_decisions.csv",
        "context_modulation": run_dir / "g18_context_pair_modulation.csv",
        "context_scores": run_dir / "g18_context_period_scores.csv",
        "context_decisions": run_dir / "g18_context_decisions.csv",
        "coverage": run_dir / "g18_direct_coverage.csv",
        "result": run_dir / "g18_direct_confirmation_result.json",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    load_freeze()
    freeze_context_registry()
    run_dir = RECORD_ROOT / args.run_id
    paths = output_paths(run_dir)
    if paths["result"].is_file() and not args.overwrite:
        print(paths["result"].read_text(encoding="utf-8"))
        return 0
    run_dir.mkdir(parents=True, exist_ok=True)

    wider = build_wider_context()
    density_parts: list[DataFrame] = []
    semantic_parts: list[DataFrame] = []
    level_parts: list[DataFrame] = []
    context_parts: list[DataFrame] = []
    coverage_parts: list[DataFrame] = []
    inventories: list[dict[str, Any]] = []
    event_paths = sorted(ATLAS_EVENT_ROOT.glob("*.parquet"))
    if len(event_paths) != 20:
        raise ValueError(f"Expected 20 Generation 17 pair-event files, got {len(event_paths)}.")
    for index, path in enumerate(event_paths, start=1):
        events = read_holdout(path)
        if events.empty:
            raise ValueError(f"No frozen Generation 18 holdout rows in {path}.")
        add_reaction_metrics(events)
        cohort = str(events["cohort"].iloc[0])
        pair = str(events["pair"].iloc[0])
        density_parts.append(density_pair_effects(events))
        semantic_parts.append(
            mean_summary(
                scoped_level_events(events, semantics=True),
                metrics=g18z.REACTION_FORMS,
                controls={"matched_random_time", "near_miss"},
            )
        )
        level_parts.append(
            mean_summary(
                scoped_level_events(events, semantics=False),
                metrics=("crossings", "unsigned_reaction"),
                controls=set(g18z.CONTROLS),
            )
        )
        context = context_state_frame(pair, wider)
        context_contrasts, context_coverage = context_pair_modulation(events, context)
        context_parts.append(context_contrasts)
        coverage_parts.append(context_coverage)
        inventories.append(
            {
                "cohort": cohort,
                "pair": pair,
                "holdout_rows": len(events),
                "first_event_utc": str(events["event_time"].min()),
                "last_event_utc": str(events["event_time"].max()),
                "source_path": str(path.resolve()),
                "source_sha256": g0.sha256_file(path),
            }
        )
        print(
            json.dumps(
                {
                    "phase": "g18_direct_pair",
                    "processed": index,
                    "total": len(event_paths),
                    "pair": pair,
                    "cohort": cohort,
                }
            ),
            flush=True,
        )

    density_effects = pd.concat(density_parts, ignore_index=True, sort=False)
    density_control = density_contrasts(density_effects)
    density_scores = period_scores(
        density_control,
        ("component", "metric", "horizon_hours"),
    )
    density_decisions = whole_decisions(
        density_scores,
        ("component", "metric", "horizon_hours"),
        g18z.CONTROLS,
    )

    semantic_summary = pd.concat(semantic_parts, ignore_index=True, sort=False)
    semantic_control = paired_contrasts(
        semantic_summary, ("matched_random_time", "near_miss")
    )
    semantic_scores = period_scores(
        semantic_control,
        ("scope_kind", "scope_value", "metric", "horizon_hours"),
    )
    semantic_decisions = whole_decisions(
        semantic_scores,
        ("scope_kind", "scope_value", "metric", "horizon_hours"),
        ("matched_random_time", "near_miss"),
    )

    level_summary = pd.concat(level_parts, ignore_index=True, sort=False)
    level_control = paired_contrasts(level_summary, g18z.CONTROLS)
    level_scores = period_scores(
        level_control,
        ("scope_kind", "scope_value", "metric", "horizon_hours"),
    )
    level_decisions = whole_decisions(
        level_scores,
        ("scope_kind", "scope_value", "metric", "horizon_hours"),
        g18z.CONTROLS,
    )

    context_control = pd.concat(context_parts, ignore_index=True, sort=False)
    context_scores = period_scores(
        context_control,
        ("context_id", "metric", "horizon_hours"),
    )
    context_decisions = whole_decisions(
        context_scores,
        ("context_id", "metric", "horizon_hours"),
        ("matched_random_time", "near_miss"),
    )

    coverage = pd.concat(coverage_parts, ignore_index=True, sort=False)
    orderbook_rows = {
        "cohort": "all",
        "pair": "all",
        "source": "source_ready_orderbook",
        "eligible_rows": 0,
        "status": "parked_no_timestamp_ready_rows_in_both_g18_holdout_halves",
    }
    coverage = pd.concat([coverage, DataFrame([orderbook_rows])], ignore_index=True)

    outputs = {
        "density_pair_effects": density_effects,
        "density_contrasts": density_control,
        "density_scores": density_scores,
        "density_decisions": density_decisions,
        "semantic_contrasts": semantic_control,
        "semantic_scores": semantic_scores,
        "semantic_decisions": semantic_decisions,
        "level_contrasts": level_control,
        "level_scores": level_scores,
        "level_decisions": level_decisions,
        "context_modulation": context_control,
        "context_scores": context_scores,
        "context_decisions": context_decisions,
        "coverage": coverage,
    }
    for name, frame in outputs.items():
        g0.atomic_write_csv(frame, paths[name])

    strict_counts = {
        "density": int(
            density_decisions["status"].eq("strict_holdout_confirmation").sum()
        ),
        "semantics": int(
            semantic_decisions["status"].eq("strict_holdout_confirmation").sum()
        ),
        "levels": int(
            level_decisions["status"].eq("strict_holdout_confirmation").sum()
        ),
        "context": int(
            context_decisions["status"].eq("strict_holdout_confirmation").sum()
        ),
    }
    result = {
        "schema_version": 1,
        "generation": 18,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation18_direct_confirmation",
        "branches_completed": [
            "g18a_density_geometry_confirmation",
            "g18b_reaction_form_confirmation",
            "g18c_level_source_confirmation",
            "g18d_context_and_market_regimes",
        ],
        "pair_tasks_completed": len(inventories),
        "strict_holdout_question_rows": strict_counts,
        "orderbook_status": orderbook_rows["status"],
        "research_boundary": {
            "profit_used": False,
            "future_signed_direction_used": False,
            "direction_branch_remained_parked": True,
        },
        "source_contracts": {
            "generation18_freeze": artifact(g18z.FREEZE_PATH),
            "context_registry": artifact(CONTEXT_REGISTRY_PATH),
        },
        "inventory": inventories,
        "artifacts": {name: artifact(paths[name]) for name in outputs},
        "result_path": str(paths["result"].resolve()),
    }
    g0.atomic_write_json(result, paths["result"])
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
