"""Test generic multi-timeframe indicator states around a fixed level basket."""

from __future__ import annotations

# Bound numerical pools before pandas/numpy imports.
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
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation17_level_source_atlas as g17l,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation18_direct_confirmation as g18d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation22_cross_asset_context as g22d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation22_period_landmarks as g22a,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation23_continuous_context as g23b,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_common as g24c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation24_freeze as g24z,
)


ANALYSIS_PATH = Path(__file__).resolve()
BRANCH_ID = "g24e_generic_multitimeframe_indicator_context"
DEFAULT_RUN_ID = "g24_generic_indicator_context_20260828a"
DEFAULT_SUPPORT_ID = "g24_generic_indicator_context_support_20260828a"
RECORD_ROOT = g24z.OUTPUT_ROOT / "generic_indicator_context"
SUPPORT_ROOT = (
    Path(
        "D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones/"
        "generation24_branches/g24_broad_siblings/generic_indicator_context"
    )
    / DEFAULT_SUPPORT_ID
)
SUPPORT_MANIFEST = RECORD_ROOT / "g24_generic_indicator_context_support_freeze.json"
TIMEFRAMES = ("1h", "4h", "1d")
TIMEFRAME_HOURS = {"1h": 1, "4h": 4, "1d": 24}
INDICATORS = (
    "rsi_14",
    "bollinger_position_20_2",
    "bollinger_width_20_2",
    "macd_histogram_12_26_9_over_atr",
    "ema_20_50_spread_over_atr",
    "ema_20_slope_over_atr",
    "atr_fraction_14",
)
BANDS = ("low", "middle", "high")
AGREEMENTS = ("all_low", "all_high")
CONTROLS = (*g24z.LEVEL_CONTROLS, "same_state_no_level_time")
CALIBRATION_END_EXCLUSIVE = g22d.CALIBRATION_END_EXCLUSIVE
NO_LEVEL_MINIMUM_SPACING_HOURS = 2
ZONE_HALF_WIDTH_ATR = 0.25


def artifact(path: Path) -> dict[str, Any]:
    return g24z.artifact(path)


def load_freeze() -> dict[str, Any]:
    frozen, branch = g24c.load_branch(BRANCH_ID)
    if tuple(branch["timeframes"]) != TIMEFRAMES:
        raise ValueError("Generation 24 generic-indicator timeframes drifted.")
    if tuple(branch["indicator_inputs"]) != INDICATORS:
        raise ValueError("Generation 24 generic-indicator registry drifted.")
    if tuple(branch["bands"]) != BANDS:
        raise ValueError("Generation 24 generic-indicator bands drifted.")
    if tuple(branch["cross_timeframe_agreement"]) != AGREEMENTS:
        raise ValueError("Generation 24 generic-indicator agreements drifted.")
    if tuple(branch["controls"]) != CONTROLS:
        raise ValueError("Generation 24 generic-indicator controls drifted.")
    if branch.get("arbitrary_multi_indicator_conjunctions_allowed") is not False:
        raise ValueError("Generation 24 arbitrary indicator conjunctions became enabled.")
    return frozen


def true_range(frame: DataFrame) -> Series:
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    previous_close = pd.to_numeric(frame["close"], errors="coerce").shift(1)
    return pd.concat(
        [(high - low).abs(), (high - previous_close).abs(), (low - previous_close).abs()],
        axis=1,
    ).max(axis=1)


def indicator_values(candles: DataFrame) -> DataFrame:
    """Conventional causal indicator values on already-completed candles."""
    close = pd.to_numeric(candles["close"], errors="coerce")
    delta = close.diff()
    average_gain = delta.clip(lower=0.0).ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
    average_loss = (
        (-delta.clip(upper=0.0)).ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
    )
    relative_strength = average_gain / average_loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + relative_strength)
    rsi = rsi.where(average_loss.ne(0.0), 100.0)

    middle = close.rolling(20, min_periods=20).mean()
    deviation = close.rolling(20, min_periods=20).std(ddof=0)
    lower = middle - 2.0 * deviation
    upper = middle + 2.0 * deviation
    band_range = (upper - lower).replace(0.0, np.nan)

    atr = true_range(candles).ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
    ema12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    ema20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ema50 = close.ewm(span=50, adjust=False, min_periods=50).mean()
    denominator = atr.replace(0.0, np.nan)
    output = DataFrame(index=candles.index)
    output["rsi_14"] = rsi
    output["bollinger_position_20_2"] = (close - lower) / band_range
    output["bollinger_width_20_2"] = band_range / middle.abs().replace(0.0, np.nan)
    output["macd_histogram_12_26_9_over_atr"] = (macd - signal) / denominator
    output["ema_20_50_spread_over_atr"] = (ema20 - ema50) / denominator
    output["ema_20_slope_over_atr"] = ema20.diff() / denominator
    output["atr_fraction_14"] = atr / close.abs().replace(0.0, np.nan)
    return output


def completed_timeframe_features(base: DataFrame, timeframe: str) -> DataFrame:
    """Align only the preceding completed informative candle to each hourly row."""
    hours = TIMEFRAME_HOURS[timeframe]
    dates = pd.to_datetime(base["date"], utc=True, errors="raise")
    bucket = dates.dt.floor(f"{hours}h")
    work = base[["open", "high", "low", "close", "volume"]].copy()
    work["bucket"] = bucket
    candles = work.groupby("bucket", observed=True, sort=True).agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    completed = indicator_values(candles).shift(1)
    aligned = completed.reindex(bucket.to_numpy()).reset_index(drop=True)
    aligned.index = base.index
    aligned.columns = [f"{timeframe}__{column}" for column in aligned.columns]
    return aligned


def causal_indicator_frame(
    base: DataFrame,
) -> tuple[DataFrame, list[dict[str, Any]]]:
    output = DataFrame({"date": pd.to_datetime(base["date"], utc=True, errors="raise")})
    for timeframe in TIMEFRAMES:
        output = pd.concat([output, completed_timeframe_features(base, timeframe)], axis=1)
    calibration = output["date"].lt(CALIBRATION_END_EXCLUSIVE)
    registry: list[dict[str, Any]] = []
    for timeframe in TIMEFRAMES:
        for indicator in INDICATORS:
            column = f"{timeframe}__{indicator}"
            reference = (
                pd.to_numeric(output.loc[calibration, column], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if len(reference) < 500:
                raise ValueError(f"Insufficient indicator calibration rows: {column}")
            lower = float(reference.quantile(1.0 / 3.0))
            upper = float(reference.quantile(2.0 / 3.0))
            band_column = f"band__{column}"
            output[band_column] = g23b.band_for_values(output[column], lower, upper)
            registry.append(
                {
                    "timeframe": timeframe,
                    "indicator": indicator,
                    "value_column": column,
                    "band_column": band_column,
                    "calibration_rows": len(reference),
                    "lower_tertile": lower,
                    "upper_tertile": upper,
                }
            )
    return output, registry


def attach_indicator_states(
    events: DataFrame, states: DataFrame, registry: list[dict[str, Any]]
) -> DataFrame:
    if events.empty:
        return events
    state_columns = [
        column
        for item in registry
        for column in (str(item["value_column"]), str(item["band_column"]))
    ]
    lookup = states[["date", *state_columns]].rename(columns={"date": "event_time"})
    merged = events.merge(lookup, on="event_time", how="left", validate="many_to_one")
    parts: list[DataFrame] = []
    for item in registry:
        band_column = str(item["band_column"])
        value_column = str(item["value_column"])
        copy = merged.loc[merged[band_column].isin(BANDS)].copy()
        copy["scope_kind"] = "single_indicator_timeframe_band"
        copy["generic_indicator"] = str(item["indicator"])
        copy["indicator_timeframe"] = str(item["timeframe"])
        copy["indicator_state"] = copy[band_column].astype(str)
        copy["indicator_value"] = pd.to_numeric(copy[value_column], errors="coerce")
        copy["scope_value"] = (
            str(item["timeframe"]) + "__" + str(item["indicator"]) + "__" + copy["indicator_state"]
        )
        parts.append(copy)
    for indicator in INDICATORS:
        columns = [f"band__{timeframe}__{indicator}" for timeframe in TIMEFRAMES]
        for agreement, band in (("all_low", "low"), ("all_high", "high")):
            mask = merged[columns].eq(band).all(axis=1)
            copy = merged.loc[mask].copy()
            if copy.empty:
                continue
            copy["scope_kind"] = "same_indicator_cross_timeframe_agreement"
            copy["generic_indicator"] = indicator
            copy["indicator_timeframe"] = "1h_4h_1d"
            copy["indicator_state"] = agreement
            copy["indicator_value"] = np.nan
            copy["scope_value"] = f"1h_4h_1d__{indicator}__{agreement}"
            parts.append(copy)
    if not parts:
        return DataFrame()
    output = pd.concat(parts, ignore_index=True, sort=False)
    output.sort_values(
        ["control", "scope_kind", "scope_value", "event_time", "pre_distance_atr"],
        inplace=True,
        kind="stable",
    )
    return output.drop_duplicates(
        ["control", "scope_kind", "scope_value", "event_time"], keep="first"
    ).reset_index(drop=True)


def state_masks(states: DataFrame) -> dict[tuple[str, str], np.ndarray]:
    masks: dict[tuple[str, str], np.ndarray] = {}
    for timeframe in TIMEFRAMES:
        for indicator in INDICATORS:
            column = f"band__{timeframe}__{indicator}"
            values = states[column].astype(str).to_numpy()
            for band in BANDS:
                value = f"{timeframe}__{indicator}__{band}"
                masks[("single_indicator_timeframe_band", value)] = values == band
    for indicator in INDICATORS:
        columns = [f"band__{timeframe}__{indicator}" for timeframe in TIMEFRAMES]
        frame = states[columns].astype(str)
        masks[
            (
                "same_indicator_cross_timeframe_agreement",
                f"1h_4h_1d__{indicator}__all_low",
            )
        ] = frame.eq("low").all(axis=1).to_numpy()
        masks[
            (
                "same_indicator_cross_timeframe_agreement",
                f"1h_4h_1d__{indicator}__all_high",
            )
        ] = frame.eq("high").all(axis=1).to_numpy()
    return masks


def same_state_no_level_events(
    base: DataFrame,
    *,
    pair: str,
    cohort: str,
    actual_scoped: DataFrame,
    states: DataFrame,
    has_level_contact: np.ndarray,
) -> DataFrame:
    dates = pd.to_datetime(base["date"], utc=True, errors="raise")
    periods = g18d.assign_confirmation_period(dates, cohort).astype(str).to_numpy()
    atr = pd.to_numeric(base["base_atr"], errors="coerce").to_numpy(dtype=float)
    close = pd.to_numeric(base["close"], errors="coerce").to_numpy(dtype=float)
    masks = state_masks(states)
    indexes = np.arange(len(base), dtype=int)
    rows: list[dict[str, Any]] = []
    keys = ["scope_kind", "scope_value", "period"]
    for identity_values, cell in actual_scoped.groupby(keys, observed=True, sort=False):
        scope_kind, scope_value, period_name = (str(value) for value in identity_values)
        eligible = np.flatnonzero(
            masks[(scope_kind, scope_value)]
            & (periods == period_name)
            & ~has_level_contact
            & np.isfinite(atr)
            & (atr > 0.0)
            & (indexes < len(base) - max(g24z.HORIZONS_HOURS))
        )
        rng = np.random.default_rng(
            g0.stable_hash_int(
                f"g24-indicator-no-level|{pair}|{scope_kind}|{scope_value}|{period_name}"
            )
        )
        blocked = np.zeros(len(base), dtype=bool)
        selected: list[int] = []
        for candidate in rng.permutation(eligible):
            index = int(candidate)
            if blocked[index]:
                continue
            selected.append(index)
            blocked[max(0, index - 1) : min(len(base), index + 2)] = True
            if len(selected) >= len(cell):
                break
        exemplar = cell.iloc[0]
        for index in selected:
            rows.append(
                {
                    "cohort": cohort,
                    "pair": pair,
                    "period": period_name,
                    "control": "same_state_no_level_time",
                    "event_time": dates.iloc[index],
                    "base_index": index,
                    "level_family": "generic_indicator_fixed_level_basket",
                    "source_family": "no_level",
                    "level_name": "same_state_current_close_anchor",
                    "level_price": close[index],
                    "zone_half_width": ZONE_HALF_WIDTH_ATR * atr[index],
                    "zone_half_width_atr": ZONE_HALF_WIDTH_ATR,
                    "base_atr": atr[index],
                    "pre_distance_atr": 0.0,
                    "approach_state": "already_inside_or_unclear",
                    "source_open": pd.NaT,
                    "match_tier": "same_indicator_state_period_minimum_2h_spacing",
                    "scope_kind": scope_kind,
                    "scope_value": scope_value,
                    "generic_indicator": str(exemplar["generic_indicator"]),
                    "indicator_timeframe": str(exemplar["indicator_timeframe"]),
                    "indicator_state": str(exemplar["indicator_state"]),
                    "indicator_value": np.nan,
                }
            )
    return DataFrame.from_records(rows)


def pair_support(
    pair: str, cohort: str, *, overwrite: bool
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output = SUPPORT_ROOT / f"{cohort}__{g0.pair_file_stem(pair)}.parquet"
    base = g17l.level_surface(pair, g20_manifest(cohort))
    states, registry = causal_indicator_frame(base)
    if output.is_file() and not overwrite:
        existing = pd.read_parquet(output, columns=["control"])
        return (
            {
                "pair": pair,
                "cohort": cohort,
                "rows": len(existing),
                "control_counts": existing["control"].value_counts().to_dict(),
                "path": str(output.resolve()),
                "sha256": g0.sha256_file(output),
                "status": "existing",
            },
            [{"pair": pair, "cohort": cohort, **item} for item in registry],
        )
    raw, surfaces = g23b.raw_level_controls(base, pair=pair, cohort=cohort)
    expanded = attach_indicator_states(raw, states, registry)
    actual_scoped = expanded.loc[expanded["control"].eq("actual")].copy()
    no_level = same_state_no_level_events(
        base,
        pair=pair,
        cohort=cohort,
        actual_scoped=actual_scoped,
        states=states,
        has_level_contact=g22d.actual_contact_mask(base, surfaces),
    )
    support = pd.concat([expanded, no_level], ignore_index=True, sort=False)
    support.sort_values(
        ["control", "scope_kind", "scope_value", "period", "event_time"],
        inplace=True,
        kind="stable",
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_parquet(support, output)
    return (
        {
            "pair": pair,
            "cohort": cohort,
            "rows": len(support),
            "control_counts": support["control"].value_counts().to_dict(),
            "path": str(output.resolve()),
            "sha256": g0.sha256_file(output),
            "status": "built",
        },
        [{"pair": pair, "cohort": cohort, **item} for item in registry],
    )


def g20_manifest(cohort: str) -> dict[str, Any]:
    path = g0.DEFAULT_MANIFEST if cohort == "normal" else g17l.MEME_MANIFEST
    return json.loads(path.read_text(encoding="utf-8"))


def freeze_support(*, overwrite: bool = False) -> dict[str, Any]:
    load_freeze()
    if SUPPORT_MANIFEST.is_file() and not overwrite:
        manifest = json.loads(SUPPORT_MANIFEST.read_text(encoding="utf-8"))
        if manifest.get("status") != "frozen_before_generation24_indicator_context_outcomes":
            raise ValueError("Invalid Generation 24 indicator-context support.")
        return manifest
    inventory: list[dict[str, Any]] = []
    state_registry: list[dict[str, Any]] = []
    pairs = g22a.cohort_pairs()
    for number, (cohort, pair) in enumerate(pairs, start=1):
        item, registry = pair_support(pair, cohort, overwrite=overwrite)
        inventory.append(item)
        state_registry.extend(registry)
        print(
            json.dumps(
                {"phase": "g24_indicator_support", "processed": number, "total": len(pairs)}
            ),
            flush=True,
        )
    manifest = {
        "schema_version": 1,
        "generation": 24,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation24_indicator_context_outcomes",
        "branch_id": BRANCH_ID,
        "inventory": inventory,
        "state_registry": state_registry,
        "future_outcome_values_read": False,
        "winner_filtering_used": False,
        "arbitrary_multi_indicator_conjunctions_used": False,
        "indicator_definition": {
            "alignment": "preceding_completed_candle_only",
            "calibration_end_exclusive_utc": CALIBRATION_END_EXCLUSIVE.isoformat(),
            "bands": "pair_timeframe_indicator training-tertiles",
            "cross_timeframe_agreement": list(AGREEMENTS),
            "no_level_minimum_spacing_hours": NO_LEVEL_MINIMUM_SPACING_HOURS,
        },
        "source_contracts": {
            "generation24_freeze": artifact(g24z.FREEZE_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
            "generation24_common": artifact(g24c.ANALYSIS_PATH),
            "fixed_level_basket_builder": artifact(g23b.ANALYSIS_PATH),
        },
    }
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(manifest, SUPPORT_MANIFEST)
    return manifest


def scoped_events(events: DataFrame) -> DataFrame:
    return events.copy()


def execute(run_id: str, *, overwrite: bool = False) -> int:
    manifest = freeze_support(overwrite=False)
    batch = g24c.require_all_frozen_supports()
    run_dir = RECORD_ROOT / run_id
    result_path = run_dir / "g24_generic_indicator_context_result.json"
    if result_path.is_file() and not overwrite:
        return 0
    events = g24c.open_frozen_support_outcomes(
        manifest, phase="g24_indicator_context_outcomes", scope_events=scoped_events
    )
    contrasts, scores, decisions = g24c.score_control_ladders(events, controls=CONTROLS)
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = g24c.write_scored_tables(
        run_dir,
        prefix="g24_generic_indicator_context",
        contrasts=contrasts,
        scores=scores,
        decisions=decisions,
    )
    result = {
        "schema_version": 1,
        "generation": 24,
        "created_at_utc": g0.utc_now(),
        "status": "completed_generation24_generic_indicator_context",
        "strict_rows": int(decisions["status"].eq("strict_holdout_confirmation").sum()),
        "point_rows": int(decisions["status"].eq("point_holdout_confirmation").sum()),
        "research_boundary": {
            "profit_used": False,
            "signed_direction_used": False,
            "winner_filtering_used": False,
            "arbitrary_multi_indicator_conjunctions_used": False,
        },
        "source_contracts": {
            "outcome_blind_support": artifact(SUPPORT_MANIFEST),
            "all_generation24_sibling_supports": batch,
        },
        "artifacts": {name: artifact(path) for name, path in paths.items()},
        "result_path": str(result_path.resolve()),
    }
    g0.atomic_write_json(result, result_path)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--prepare-support", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.prepare_support:
        print(json.dumps(freeze_support(overwrite=args.overwrite), indent=2))
        return 0
    return execute(args.run_id, overwrite=args.overwrite)


if __name__ == "__main__":
    raise SystemExit(main())
