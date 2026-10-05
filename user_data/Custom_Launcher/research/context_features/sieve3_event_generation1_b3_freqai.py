"""Run direct and FreqAI tests for the frozen G1-B3 mixed-level activity lead."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from sieve3_event_freqai_round1 import (
    REPO_ROOT,
    USER_DATA_DIR,
    append_log,
    atomic_json,
    load_predictions,
    now_iso,
    ohlcv_path,
    parse_pairs,
    parse_windows,
    score_values,
)
from sieve3_event_generation1_b1_freqai import (
    DEFAULT_BASE_CONFIG,
    DEFAULT_WINDOWS,
    INCLUDE_TIMEFRAMES,
    METRICS,
    add_majority_consistency,
    profile_config,
    run_manifest,
)
from user_data.strategies.sieve3_event_reaction_targets import (
    build_sieve3_reaction_targets,
)


DEFAULT_OUTPUT = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "sieve3_event_reaction"
    / "generation1"
    / "g1-b3"
    / "freqai"
)
DEFAULT_CACHE = DEFAULT_OUTPUT.parent / "cache"
DEFAULT_SCOPES = DEFAULT_OUTPUT.parent / "event_scopes.parquet"
DEFAULT_PYTHON = (
    REPO_ROOT / "runtime" / "venvs" / "freqtrade-backtest-06" / "Scripts" / "python.exe"
)
BANDS = (0.005, 0.01)
SIDES = ("long", "short")


def band_token(band: float) -> str:
    return f"{int(round(band * 10000.0))}bp"


def fields(prefix: str, names: tuple[str, ...]) -> list[str]:
    return [
        f"{prefix}__{side}__{band_token(band)}__{name}"
        for side in SIDES
        for band in BANDS
        for name in names
    ]


RELATIONSHIP_FIELDS = (
    "relationship_mixed",
    "relationship_same_direction_only",
    "relationship_opposing_only",
    "relationship_crossed_or_displaced_only",
    "relationship_no_nearby_level",
)
COUNT_FIELDS = ("same_count", "opposing_count")
TIMEFRAME_FIELDS = (
    "same_max_tf_72h",
    "opposing_max_tf_72h",
    "one_sided_max_tf_72h",
    "mixed_same_higher",
    "mixed_opposing_higher",
)
EXACT_FIELDS = ("mixed_equal_highest",)
COMPONENTS = fields("level", (*RELATIONSHIP_FIELDS, *COUNT_FIELDS, *TIMEFRAME_FIELDS))
EXACT = fields("level", EXACT_FIELDS)
FULL = [*COMPONENTS, *EXACT]
STALE_FULL = fields(
    "stale_level", (*RELATIONSHIP_FIELDS, *COUNT_FIELDS, *TIMEFRAME_FIELDS, *EXACT_FIELDS)
)
NO_MIXED = fields("level", (*COUNT_FIELDS, *TIMEFRAME_FIELDS))
NO_TIMEFRAME = fields("level", (*RELATIONSHIP_FIELDS, *COUNT_FIELDS))

PROFILES: dict[str, dict[str, Any]] = {
    "fixed_indicator_control": {
        "strategy": "Sieve3EventReactionFixedIndicatorControlFreqAIResearchStrategy",
        "shift_hours": 0,
        "feature_columns": None,
        "theory": "Fixed price and conventional-indicator state without level proximity.",
    },
    "level_components": {
        "strategy": "Sieve3EventReactionFixedIndicatorIdentityFreqAIResearchStrategy",
        "shift_hours": 0,
        "feature_columns": COMPONENTS,
        "theory": "Current mixed/one-sided state, counts and timeframe balance without the explicit mixed-equal interaction flag.",
    },
    "exact_mixed_equal": {
        "strategy": "Sieve3EventReactionFixedIndicatorIdentityFreqAIResearchStrategy",
        "shift_hours": 0,
        "feature_columns": EXACT,
        "theory": "Only the explicit mixed equal-highest-timeframe state at 0.5% and 1% is added.",
    },
    "stale_full_state": {
        "strategy": "Sieve3EventReactionFixedIndicatorIdentityFreqAIResearchStrategy",
        "shift_hours": 0,
        "feature_columns": STALE_FULL,
        "theory": "The identical relationship representation recomputed from levels made stale by 168 hours is the placebo.",
    },
    "full_current_state": {
        "strategy": "Sieve3EventReactionFixedIndicatorIdentityFreqAIResearchStrategy",
        "shift_hours": 0,
        "feature_columns": FULL,
        "theory": "Current components plus the explicit mixed-equal interaction test live incremental level-state information.",
    },
    "components_no_mixed": {
        "strategy": "Sieve3EventReactionFixedIndicatorIdentityFreqAIResearchStrategy",
        "shift_hours": 0,
        "feature_columns": NO_MIXED,
        "theory": "Counts and timeframe fields without categorical mixed-state identity form the mixed-state ablation.",
    },
    "components_no_timeframe": {
        "strategy": "Sieve3EventReactionFixedIndicatorIdentityFreqAIResearchStrategy",
        "shift_hours": 0,
        "feature_columns": NO_TIMEFRAME,
        "theory": "Relationship state and counts without timeframe balance form the timeframe ablation.",
    },
}


def targets() -> list[str]:
    output = [
        "&-future_reaction_magnitude_2h_atr",
        "&-future_reaction_magnitude_4h_atr",
        "&-future_volume_ratio_2h",
        "&-future_volume_ratio_24h",
        "&-future_volume_ratio_48h",
        "&-future_return_2h",
        "&-future_return_4h",
        "&-future_path_balance_2h_atr",
        "&-future_path_balance_4h_atr",
    ]
    for horizon in (4, 8, 12, 24, 48):
        output.extend(
            [
                f"&-one_atr_touch_observed_{horizon}h",
                f"&-first_1atr_touch_step_censored_{horizon}h",
            ]
        )
    return output


TARGETS = targets()


def target_metadata(target: str) -> tuple[str, int]:
    match = re.search(r"_(\d+)h(?:_|$)", target)
    if not match:
        raise ValueError(target)
    horizon = int(match.group(1))
    if "reaction_magnitude" in target:
        family = "reaction_magnitude"
    elif "volume_ratio" in target:
        family = "volume_activity"
    elif "one_atr_touch_observed" in target:
        family = "unconditional_one_atr_hit"
    elif "touch_step_censored" in target:
        family = "censored_reaction_timing"
    elif "future_return" in target:
        family = "terminal_direction_falsification"
    elif "path_balance" in target:
        family = "path_direction_falsification"
    else:
        raise ValueError(target)
    return family, horizon


def actuals_for_pair(pair: str) -> DataFrame:
    raw = pd.read_feather(ohlcv_path(pair))
    raw["date"] = pd.to_datetime(raw["date"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
    labelled = build_sieve3_reaction_targets(raw)
    for horizon in (2, 4):
        labelled[f"&-future_reaction_magnitude_{horizon}h_atr"] = pd.concat(
            [
                pd.to_numeric(
                    labelled[f"&-future_upside_{horizon}h_atr"], errors="coerce"
                ),
                pd.to_numeric(
                    labelled[f"&-future_downside_{horizon}h_atr"], errors="coerce"
                ),
            ],
            axis=1,
        ).max(axis=1, skipna=False)
    return labelled[["date", *TARGETS]].rename(
        columns={target: f"{target}_actual" for target in TARGETS}
    )


def scope_dates(scopes_path: Path, pair: str) -> dict[str, set[pd.Timestamp]]:
    frame = pd.read_parquet(scopes_path)
    frame = frame[frame["pair"].eq(pair)].copy()
    frame["decision_time"] = pd.to_datetime(frame["decision_time"], utc=True)
    output: dict[str, set[pd.Timestamp]] = {}
    for keys, group in frame.groupby(
        ["family", "entry_side"], dropna=False, observed=True
    ):
        family, side = keys
        dates = set(group["decision_time"] - pd.Timedelta(hours=1))
        if len(dates) >= 20:
            output[f"family:{family}:{side}"] = dates
    for side, group in frame.groupby("entry_side", observed=True):
        dates = set(group["decision_time"] - pd.Timedelta(hours=1))
        if len(dates) >= 20:
            output[f"side:{side}"] = dates
    return output


def build_manifest(
    output_dir: Path,
    base_config_path: Path,
    python_exe: Path,
    pairs: tuple[str, ...],
    windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    selected_profiles: tuple[str, ...],
    timerange: str,
    cache_dir: Path,
    scopes_path: Path,
) -> dict[str, Any]:
    base = json.loads(base_config_path.read_text(encoding="utf-8"))
    run_id = f"sieve3_event_g1_b3_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    commands: list[dict[str, Any]] = []
    for profile_id in selected_profiles:
        definition = PROFILES[profile_id]
        identifier = f"{run_id}_{profile_id}"
        profile_dir = output_dir / profile_id
        profile_dir.mkdir(parents=True, exist_ok=True)
        config = profile_config(base, identifier, definition, pairs, cache_dir)
        settings = config["sieve3_event_reaction"]
        settings["target_names"] = TARGETS
        if definition.get("feature_columns"):
            settings["event_feature_columns"] = list(definition["feature_columns"])
        config_path = profile_dir / "config.json"
        atomic_json(config_path, config)
        commands.append(
            {
                "profile_id": profile_id,
                "strategy": definition["strategy"],
                "theory": definition["theory"],
                "identifier": identifier,
                "config_path": str(config_path),
                "output_dir": str(profile_dir),
                "event_feature_columns": definition.get("feature_columns") or [],
                "command": [
                    str(python_exe),
                    "-m",
                    "freqtrade",
                    "backtesting",
                    "--userdir",
                    str(USER_DATA_DIR),
                    "--config",
                    str(config_path),
                    "--strategy",
                    definition["strategy"],
                    "--freqaimodel",
                    "LightGBMRegressorMultiTarget",
                    "--timerange",
                    timerange,
                    "--export",
                    "signals",
                    "--export-directory",
                    str(profile_dir),
                ],
                "status": "pending",
                "attempts": 0,
            }
        )
    return {
        "schema_version": 1,
        "batch": "G1-B3",
        "run_id": run_id,
        "created_at": now_iso(),
        "pairs": list(pairs),
        "windows": {
            key: [start.isoformat(), end.isoformat()]
            for key, (start, end) in windows.items()
        },
        "timerange": timerange,
        "targets": TARGETS,
        "include_timeframes": INCLUDE_TIMEFRAMES,
        "event_cache_dir": str(cache_dir),
        "event_scopes": str(scopes_path),
        "worker_python": str(python_exe),
        "comparison_contract": {
            "full_current_state": [
                "fixed_indicator_control",
                "level_components",
                "exact_mixed_equal",
                "stale_full_state",
                "components_no_mixed",
                "components_no_timeframe",
            ],
            "level_components": ["fixed_indicator_control"],
            "exact_mixed_equal": ["fixed_indicator_control"],
            "stale_full_state": ["fixed_indicator_control"],
        },
        "commands": commands,
    }


COMPARATOR_PREFIXES = {
    "fixed_indicator_control": "indicator",
    "level_components": "components",
    "exact_mixed_equal": "exact",
    "stale_full_state": "stale",
    "components_no_mixed": "no_mixed",
    "components_no_timeframe": "no_timeframe",
}


def add_comparisons(scores: DataFrame) -> DataFrame:
    key = ["pair", "window", "scope", "target"]
    scores = scores.copy()
    for metric in METRICS:
        if metric not in scores:
            scores[metric] = np.nan
    output = scores.copy()
    for profile_id, prefix in COMPARATOR_PREFIXES.items():
        control = scores[scores["profile_id"].eq(profile_id)][[*key, *METRICS]].rename(
            columns={metric: f"{prefix}_{metric}" for metric in METRICS}
        )
        output = output.merge(control, on=key, how="left", validate="many_to_one")
        for metric in (
            "prediction_actual_spearman",
            "top_minus_bottom",
            "r_squared",
            "roc_auc",
            "average_precision",
        ):
            output[f"{metric}_delta_vs_{prefix}"] = pd.to_numeric(
                output[metric], errors="coerce"
            ) - pd.to_numeric(output[f"{prefix}_{metric}"], errors="coerce")
        for metric in (
            "mean_absolute_error",
            "root_mean_squared_error",
            "brier_score_clipped",
        ):
            output[f"{metric}_skill_vs_{prefix}"] = pd.to_numeric(
                output[f"{prefix}_{metric}"], errors="coerce"
            ) - pd.to_numeric(output[metric], errors="coerce")
    return output


def score_manifest(manifest: dict[str, Any], run_dir: Path) -> DataFrame:
    pairs = tuple(manifest["pairs"])
    windows = {
        name: (pd.Timestamp(bounds[0]), pd.Timestamp(bounds[1]))
        for name, bounds in manifest["windows"].items()
    }
    actuals = {pair: actuals_for_pair(pair) for pair in pairs}
    scopes = {
        pair: scope_dates(Path(manifest["event_scopes"]), pair) for pair in pairs
    }
    rows: list[dict[str, Any]] = []
    for item in manifest["commands"]:
        predictions = load_predictions(item["identifier"], pairs)
        for pair in pairs:
            merged = predictions[predictions["pair"].eq(pair)].merge(
                actuals[pair], on="date", how="left", validate="one_to_one"
            )
            merged["decision_time"] = merged["date"] + pd.Timedelta(hours=1)
            if "do_predict" in merged:
                merged = merged[
                    pd.to_numeric(merged["do_predict"], errors="coerce").eq(1.0)
                ]
            for window, (start, end) in windows.items():
                window_frame = merged[
                    merged["decision_time"].ge(start)
                    & merged["decision_time"].lt(end)
                ]
                scope_map = {"all": set(window_frame["date"]), **scopes[pair]}
                for scope, dates in scope_map.items():
                    scoped = window_frame[window_frame["date"].isin(dates)]
                    for target in TARGETS:
                        rows.append(
                            {
                                "profile_id": item["profile_id"],
                                "identifier": item["identifier"],
                                "pair": pair,
                                "window": window,
                                "scope": scope,
                                "target": target,
                                **score_values(scoped, target, f"{target}_actual"),
                            }
                        )
    scores = DataFrame(rows)
    metadata = scores["target"].map(target_metadata)
    scores["target_family"] = metadata.map(lambda item: item[0])
    scores["horizon_hours"] = metadata.map(lambda item: item[1])
    scores = add_comparisons(scores)
    scores.to_csv(run_dir / "g1_b3_scores.csv", index=False)
    return scores


def portability_summary(
    scores: DataFrame, manifest: dict[str, Any], run_dir: Path
) -> DataFrame:
    records: list[DataFrame] = []
    for profile_id, comparators in manifest["comparison_contract"].items():
        profile = scores[
            scores["status"].eq("scored")
            & scores["profile_id"].eq(profile_id)
            & ~scores["scope"].eq("all")
        ].copy()
        for comparator_id in comparators:
            prefix = COMPARATOR_PREFIXES[comparator_id]
            work = profile[
                [
                    "pair",
                    "window",
                    "scope",
                    "target",
                    "target_family",
                    "horizon_hours",
                    "rows",
                    f"prediction_actual_spearman_delta_vs_{prefix}",
                    f"mean_absolute_error_skill_vs_{prefix}",
                ]
            ].copy()
            work["profile_id"] = profile_id
            work["comparator_id"] = comparator_id
            work["spearman_delta"] = pd.to_numeric(
                work.pop(f"prediction_actual_spearman_delta_vs_{prefix}"),
                errors="coerce",
            )
            work["mae_skill"] = pd.to_numeric(
                work.pop(f"mean_absolute_error_skill_vs_{prefix}"), errors="coerce"
            )
            work["rank_positive"] = work["spearman_delta"].gt(0.0)
            work["mae_positive"] = work["mae_skill"].gt(0.0)
            work["rank_and_mae_positive"] = (
                work["rank_positive"] & work["mae_positive"]
            )
            records.append(work)
    long = pd.concat(records, ignore_index=True)
    grouping = [
        "profile_id",
        "comparator_id",
        "scope",
        "target",
        "target_family",
        "horizon_hours",
    ]
    summary = (
        long.groupby(
            grouping,
            dropna=False,
            observed=True,
        )
        .agg(
            pair_window_tests=("pair", "size"),
            distinct_pairs=("pair", "nunique"),
            distinct_windows=("window", "nunique"),
            sample_rows=("rows", "sum"),
            rank_positive_share=("rank_positive", "mean"),
            mae_positive_share=("mae_positive", "mean"),
            median_spearman_delta=("spearman_delta", "median"),
            median_mae_skill=("mae_skill", "median"),
        )
        .reset_index()
    )
    summary = add_majority_consistency(
        summary, long, grouping, "rank_and_mae_positive", "rank_mae"
    )
    summary.to_csv(run_dir / "g1_b3_portability_summary.csv", index=False)
    return summary


def direct_state_summary(
    *,
    cache_dir: Path,
    scopes_path: Path,
    windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]],
    pairs: tuple[str, ...],
    output_path: Path,
) -> DataFrame:
    scopes = pd.read_parquet(scopes_path)
    scopes["decision_time"] = pd.to_datetime(scopes["decision_time"], utc=True)
    # The direct level-state question is defined per observable coin/time/side
    # state.  Several entry families can fire on that same state, but repeating
    # the identical cache row once per family would overweight confluence and
    # inflate support.  Family-specific FreqAI scopes remain separate elsewhere.
    scopes = scopes.drop_duplicates(["pair", "decision_time", "entry_side"])
    actuals = {pair: actuals_for_pair(pair) for pair in pairs}
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        pair_scopes = scopes[scopes["pair"].eq(pair)].copy()
        cache = pd.read_parquet(
            cache_dir / f"{pair.split('/', 1)[0].lower()}_sieve3_events_1h.parquet"
        )
        cache["decision_time"] = pd.to_datetime(cache["decision_time"], utc=True)
        frame = pair_scopes.merge(cache, on="decision_time", how="left", validate="many_to_one")
        frame["date"] = frame["decision_time"] - pd.Timedelta(hours=1)
        frame = frame.merge(actuals[pair], on="date", how="left", validate="many_to_one")
        for window, (start, end) in windows.items():
            window_frame = frame[
                frame["decision_time"].ge(start) & frame["decision_time"].lt(end)
            ]
            for side in SIDES:
                side_frame = window_frame[window_frame["entry_side"].eq(side)]
                for band in BANDS:
                    token = band_token(band)
                    prefix = f"level__{side}__{token}"
                    stale_prefix = f"stale_level__{side}__{token}"
                    masks = {
                        "mixed_equal_highest": side_frame[
                            f"{prefix}__mixed_equal_highest"
                        ].eq(1.0),
                        "mixed_same_higher": side_frame[
                            f"{prefix}__mixed_same_higher"
                        ].eq(1.0),
                        "mixed_opposing_higher": side_frame[
                            f"{prefix}__mixed_opposing_higher"
                        ].eq(1.0),
                        "no_nearby_level": side_frame[
                            f"{prefix}__relationship_no_nearby_level"
                        ].eq(1.0),
                        "stale_mixed_equal_highest": side_frame[
                            f"{stale_prefix}__mixed_equal_highest"
                        ].eq(1.0),
                    }
                    for target in TARGETS:
                        values = pd.to_numeric(
                            side_frame[f"{target}_actual"], errors="coerce"
                        )
                        record: dict[str, Any] = {
                            "pair": pair,
                            "window": window,
                            "side": side,
                            "band_pct": band,
                            "target": target,
                            "target_family": target_metadata(target)[0],
                            "horizon_hours": target_metadata(target)[1],
                        }
                        for name, mask in masks.items():
                            selected = values[mask & values.notna()]
                            record[f"{name}_rows"] = int(len(selected))
                            record[f"{name}_mean"] = (
                                float(selected.mean()) if not selected.empty else np.nan
                            )
                        rows.append(record)
    output = DataFrame(rows)
    output.to_csv(output_path, index=False)
    return output


def direct_portability_summary(direct: DataFrame, output_path: Path) -> DataFrame:
    records: list[dict[str, Any]] = []
    for comparator in (
        "no_nearby_level",
        "stale_mixed_equal_highest",
        "mixed_same_higher",
        "mixed_opposing_higher",
    ):
        valid = direct[
            pd.to_numeric(direct["mixed_equal_highest_rows"], errors="coerce").ge(5)
            & pd.to_numeric(direct[f"{comparator}_rows"], errors="coerce").ge(5)
        ].copy()
        valid["delta"] = pd.to_numeric(
            valid["mixed_equal_highest_mean"], errors="coerce"
        ) - pd.to_numeric(valid[f"{comparator}_mean"], errors="coerce")
        for keys, group in valid.groupby(
            ["band_pct", "target", "target_family", "horizon_hours"],
            dropna=False,
            observed=True,
        ):
            band, target, family, horizon = keys
            records.append(
                {
                    "comparator": comparator,
                    "band_pct": float(band),
                    "target": target,
                    "target_family": family,
                    "horizon_hours": int(horizon),
                    "pair_window_side_tests": int(len(group)),
                    "distinct_pairs": int(group["pair"].nunique()),
                    "distinct_windows": int(group["window"].nunique()),
                    "sample_rows": int(group["mixed_equal_highest_rows"].sum()),
                    "comparator_sample_rows": int(
                        group[f"{comparator}_rows"].sum()
                    ),
                    "median_delta": float(group["delta"].median()),
                    "positive_share": float(group["delta"].gt(0.0).mean()),
                    "negative_share": float(group["delta"].lt(0.0).mean()),
                    "nonnegative_share": float(group["delta"].ge(0.0).mean()),
                }
            )
    summary = DataFrame(records)
    summary["classification"] = "mixed_or_insufficient"
    breadth = summary["distinct_pairs"].ge(7) & summary["distinct_windows"].ge(2)
    positive = summary["median_delta"].gt(0.0) & summary["positive_share"].ge(0.65)
    activity = summary["target_family"].isin(
        ["reaction_magnitude", "volume_activity"]
    )
    summary.loc[breadth & positive & activity, "classification"] = (
        "coherent_positive_activity"
    )
    timing_rows = summary["target_family"].eq("censored_reaction_timing")
    for index, row in summary[timing_rows & breadth].iterrows():
        hit = summary[
            summary["comparator"].eq(row["comparator"])
            & np.isclose(summary["band_pct"], row["band_pct"])
            & summary["target_family"].eq("unconditional_one_atr_hit")
            & summary["horizon_hours"].eq(row["horizon_hours"])
        ]
        if hit.empty:
            continue
        hit_row = hit.iloc[0]
        timing_faster = row["median_delta"] < 0.0 and row["negative_share"] >= 0.65
        hit_breadth = (
            hit_row["distinct_pairs"] >= 7
            and hit_row["distinct_windows"] >= 2
        )
        hit_not_worse = (
            hit_row["median_delta"] >= 0.0
            and hit_row["nonnegative_share"] >= 0.65
        )
        if timing_faster and hit_breadth and hit_not_worse:
            summary.loc[index, "classification"] = (
                "coherent_faster_unconditional_reaction"
            )
    summary.to_csv(output_path, index=False)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-exe", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--scopes", type=Path, default=DEFAULT_SCOPES)
    parser.add_argument("--windows-json", type=Path, default=DEFAULT_WINDOWS)
    parser.add_argument("--timerange", default="20240401-20260629")
    parser.add_argument(
        "--pairs",
        default="BTC/USDT:USDT,ETH/USDT:USDT,BNB/USDT:USDT,SOL/USDT:USDT,XRP/USDT:USDT,ADA/USDT:USDT,DOGE/USDT:USDT,TRX/USDT:USDT,AVAX/USDT:USDT,LINK/USDT:USDT",
    )
    parser.add_argument("--profiles", default=",".join(PROFILES))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--direct-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()
    if not args.python_exe.is_file():
        raise FileNotFoundError(args.python_exe)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    selected = tuple(token.strip() for token in args.profiles.split(",") if token.strip())
    unknown = set(selected) - set(PROFILES)
    if unknown:
        raise ValueError(f"Unknown profiles: {sorted(unknown)}")
    pairs = parse_pairs(args.pairs)
    windows = parse_windows(args.windows_json)
    direct = direct_state_summary(
        cache_dir=args.cache_dir.resolve(),
        scopes_path=args.scopes.resolve(),
        windows=windows,
        pairs=pairs,
        output_path=output_dir / "g1_b3_direct_state_summary.csv",
    )
    direct_portability = direct_portability_summary(
        direct, output_dir / "g1_b3_direct_portability_summary.csv"
    )
    if args.direct_only:
        print(
            json.dumps(
                {
                    "direct_rows": len(direct),
                    "direct_portability_rows": len(direct_portability),
                    "classification_counts": direct_portability[
                        "classification"
                    ].value_counts().to_dict(),
                },
                indent=2,
            )
        )
        return 0
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        existing = tuple(item["profile_id"] for item in manifest["commands"])
        if existing != selected:
            raise ValueError(
                f"Existing profile manifest differs: existing={existing} requested={selected}"
            )
    else:
        manifest = build_manifest(
            output_dir,
            args.base_config.resolve(),
            args.python_exe.resolve(),
            pairs,
            windows,
            selected,
            args.timerange,
            args.cache_dir.resolve(),
            args.scopes.resolve(),
        )
        atomic_json(manifest_path, manifest)
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0
    log_path = output_dir.parent / "research_log.md"
    if not args.score_only:
        returncode = run_manifest(manifest, manifest_path, log_path)
        if returncode:
            return returncode
    scores = score_manifest(manifest, output_dir)
    portability = portability_summary(scores, manifest, output_dir)
    summary = {
        "finished_at": now_iso(),
        "batch": "G1-B3",
        "profiles": len(manifest["commands"]),
        "direct_rows": len(direct),
        "direct_portability_rows": len(direct_portability),
        "score_rows": len(scores),
        "portability_rows": len(portability),
        "direct": str(output_dir / "g1_b3_direct_state_summary.csv"),
        "direct_portability": str(
            output_dir / "g1_b3_direct_portability_summary.csv"
        ),
        "scores": str(output_dir / "g1_b3_scores.csv"),
        "portability": str(output_dir / "g1_b3_portability_summary.csv"),
    }
    atomic_json(output_dir / "run_summary.json", summary)
    append_log(
        log_path,
        f"G1-B3 ladder completed with {len(direct)} direct rows, {len(scores)} score rows and {len(portability)} portability rows.",
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
