#!/usr/bin/env python3
"""Validate indicator scores against their documented behaviour and forward outcomes.

This is intentionally separate from Freqtrade hyperopt. Hyperopt can show that a
strategy traded profitably; this runner checks whether each indicator score is
internally honest and whether higher scores sort future market behaviour in the
direction the indicator claims.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


THIS_DIR = Path(__file__).resolve().parent
USER_DATA_DIR = THIS_DIR.parent
PROJECT_ROOT = USER_DATA_DIR.parent
RUNTIME_DIR = USER_DATA_DIR / "Indicator_External_Validator"

from user_data.Indicators.complex_pattern_structure import add_pattern_structure
from user_data.Indicators.complex_pivot_structure import add_pivot_structure
from user_data.Indicators.complex_relative_strength import add_relative_strength
from user_data.Indicators.complex_trendline_projection import TrendlineProjectionConfig, add_trendline_projection
from user_data.Indicators.complex_volatility_cycles import add_volatility_cycles
from user_data.Indicators.complex_volume_indicators import add_complex_volume_indicators
from user_data.Indicators.complex_volume_profile import VolumeProfileConfig, add_volume_profile


INDICATOR_ORDER = ("pa", "tl", "vol", "vp", "pat", "vc", "rs")
STATE_COLUMNS = {"pa_state", "tl_state", "vol_state", "vp_state", "pat_state", "vc_state", "rs_state"}
OHLCV_COLUMNS = ("date", "open", "high", "low", "close", "volume")


@dataclass(frozen=True)
class CandleFile:
    path: Path
    pair_key: str
    pair_label: str
    timeframe: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate indicator score behaviour against OHLCV data.")
    parser.add_argument("--datadir", default=str(USER_DATA_DIR / "data"), help="Freqtrade data directory to scan recursively.")
    parser.add_argument("--pairs", default="", help="Pairs to include. Blank means all discovered pairs up to --max-pairs.")
    parser.add_argument("--timeframes", default="1h", help="Space/comma separated timeframes, e.g. '30m 1h 4h'.")
    parser.add_argument("--timerange", default="", help="Optional Freqtrade-style timerange, e.g. 20230101-20250101.")
    parser.add_argument("--indicators", default="all", help="all or comma/space list from: pa tl vol vp pat vc rs.")
    parser.add_argument("--score-scope", choices=("base", "all"), default="base", help="base validates *_score_long/short/abs only; all includes component score columns.")
    parser.add_argument("--benchmark", default="BTC/USDT:USDT", help="Benchmark pair for relative strength.")
    parser.add_argument("--forward-windows", default="3 6 12 24", help="Forward candle windows for outcome tests.")
    parser.add_argument("--deciles", type=int, default=10, help="Number of score buckets.")
    parser.add_argument("--score-threshold", type=float, default=0.70, help="High-score threshold for behaviour checks.")
    parser.add_argument("--max-pairs", type=int, default=20, help="Maximum pairs per timeframe. 0 means no limit.")
    parser.add_argument("--min-rows", type=int, default=250, help="Minimum rows after timerange filtering.")
    parser.add_argument("--output-dir", default=str(RUNTIME_DIR), help="Directory for JSON/CSV reports.")
    parser.add_argument("--profile-window", type=int, default=96, help="Volume profile rolling window.")
    parser.add_argument("--profile-bins", type=int, default=48, help="Volume profile price bins.")
    parser.add_argument("--profile-chunk-size", type=int, default=512, help="Volume profile chunk size.")
    parser.add_argument("--quiet", action="store_true", help="Only print final report paths.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = normalize_indicators(args.indicators)
    timeframes = token_list(args.timeframes)
    forward_windows = [int(value) for value in token_list(args.forward_windows) if int(value) > 0]
    if not timeframes:
        raise SystemExit("No timeframes selected.")
    if not forward_windows:
        raise SystemExit("No forward windows selected.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    datadir = Path(args.datadir)
    pair_filters = pair_filter_keys(args.pairs)
    files = discover_candle_files(datadir, timeframes, pair_filters, args.max_pairs)
    if not files:
        raise SystemExit(f"No OHLCV files found in {datadir} for timeframes={timeframes} pairs={args.pairs or 'all'}")

    if not args.quiet:
        print("Indicator validation")
        print(f"Data dir: {datadir}")
        print(f"Indicators: {', '.join(selected)}")
        print(f"Forward windows: {', '.join(str(w) for w in forward_windows)}")
        print(f"Files: {len(files)}")

    benchmark_by_timeframe = load_benchmarks(datadir, args.benchmark, timeframes, args.timerange)
    score_rows: list[dict[str, Any]] = []
    decile_rows: list[dict[str, Any]] = []
    behavior_rows: list[dict[str, Any]] = []
    contract_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []

    for file in files:
        try:
            raw = load_candle_frame(file.path)
            raw = apply_timerange(raw, args.timerange)
            if len(raw) < int(args.min_rows):
                failures.append({"file": str(file.path), "reason": f"too few rows after filtering: {len(raw)}"})
                continue
            benchmark = benchmark_by_timeframe.get(file.timeframe)
            frame = add_selected_indicators(raw, selected, benchmark, args)
            if not args.quiet:
                print(f"Validated features: {file.pair_label} {file.timeframe} rows={len(frame)}")
            context = {"pair": file.pair_label, "timeframe": file.timeframe, "file": str(file.path)}
            score_columns = selected_score_columns(frame, selected, args.score_scope)
            contract_rows.extend(contract_checks(frame, score_columns, selected, context))
            behavior_rows.extend(behavior_checks(frame, selected, context, args.score_threshold))
            rows, deciles = outcome_checks(frame, score_columns, forward_windows, args.deciles, context)
            score_rows.extend(rows)
            decile_rows.extend(deciles)
        except Exception as exc:
            failures.append({"file": str(file.path), "reason": repr(exc)})

    summary = summarize_run(score_rows, behavior_rows, contract_rows, failures, args, selected, files)
    summary_path = output_dir / "latest_indicator_validation_summary.json"
    score_path = output_dir / "latest_indicator_score_metrics.csv"
    decile_path = output_dir / "latest_indicator_deciles.csv"
    behavior_path = output_dir / "latest_indicator_behavior.csv"
    contract_path = output_dir / "latest_indicator_contract.csv"

    write_json(summary_path, summary)
    write_csv(score_path, score_rows)
    write_csv(decile_path, decile_rows)
    write_csv(behavior_path, behavior_rows)
    write_csv(contract_path, contract_rows)

    print_report(summary, score_rows, behavior_rows, failures)
    print("")
    print(f"Summary JSON: {summary_path}")
    print(f"Score metrics CSV: {score_path}")
    print(f"Deciles CSV: {decile_path}")
    print(f"Behaviour CSV: {behavior_path}")
    print(f"Contract CSV: {contract_path}")
    return 0 if score_rows else 2


def normalize_indicators(value: str) -> list[str]:
    tokens = [token.lower() for token in token_list(value)]
    if not tokens or "all" in tokens:
        return list(INDICATOR_ORDER)
    selected = [token for token in tokens if token in INDICATOR_ORDER]
    unknown = sorted(set(tokens).difference(INDICATOR_ORDER))
    if unknown:
        raise SystemExit(f"Unknown indicators: {', '.join(unknown)}")
    return selected


def token_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item for item in re.split(r"[\s,;]+", str(value).strip()) if item]


def normalize_pair_key(value: str) -> str:
    text = str(value or "").upper().strip()
    text = text.replace("/", "_").replace(":", "_").replace("-", "_")
    text = re.sub(r"[^A-Z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text


def pair_filter_keys(value: str) -> set[str]:
    keys: set[str] = set()
    for token in token_list(value):
        key = normalize_pair_key(token)
        if not key:
            continue
        keys.add(key)
        if key.endswith("_USDT_USDT"):
            keys.add(key.removesuffix("_USDT"))
        elif key.endswith("_USDT"):
            keys.add(f"{key}_USDT")
    return keys


def discover_candle_files(datadir: Path, timeframes: list[str], pair_filters: set[str], max_pairs: int) -> list[CandleFile]:
    if not datadir.exists():
        return []
    candidates: list[CandleFile] = []
    allowed_suffixes = {".feather", ".parquet", ".json", ".gz"}
    for path in datadir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in allowed_suffixes:
            continue
        name = path.name
        if "funding_rate" in name or "-mark" in name or "_mark" in name:
            continue
        parsed = parse_candle_filename(path, timeframes)
        if parsed is None:
            continue
        pair_key, timeframe = parsed
        if pair_filters and pair_key not in pair_filters:
            continue
        candidates.append(CandleFile(path=path, pair_key=pair_key, pair_label=pair_key.replace("_", "/"), timeframe=timeframe))
    candidates.sort(key=lambda item: (item.timeframe, item.pair_key, str(item.path)))
    if max_pairs <= 0:
        return candidates
    counts: dict[str, int] = {}
    limited: list[CandleFile] = []
    for item in candidates:
        count = counts.get(item.timeframe, 0)
        if count >= max_pairs:
            continue
        limited.append(item)
        counts[item.timeframe] = count + 1
    return limited


def parse_candle_filename(path: Path, timeframes: list[str]) -> tuple[str, str] | None:
    stem = path.name
    for suffix in (".json.gz", ".feather", ".parquet", ".json"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    for timeframe in sorted(timeframes, key=len, reverse=True):
        marker = f"-{timeframe}"
        if marker not in stem:
            continue
        pair_part = stem.split(marker, 1)[0]
        if not pair_part:
            continue
        return normalize_pair_key(pair_part), timeframe
    return None


def load_candle_frame(path: Path) -> DataFrame:
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".feather"):
        frame = pd.read_feather(path)
    elif suffixes.endswith(".parquet"):
        frame = pd.read_parquet(path)
    elif suffixes.endswith(".json") or suffixes.endswith(".json.gz"):
        frame = pd.read_json(path)
    else:
        raise ValueError(f"Unsupported data file: {path}")
    missing = [column for column in OHLCV_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns {missing}")
    frame = frame.loc[:, list(OHLCV_COLUMNS)].copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["open", "high", "low", "close"])
    frame["volume"] = frame["volume"].fillna(0.0)
    return frame.set_index("date", drop=False)


def apply_timerange(frame: DataFrame, timerange: str) -> DataFrame:
    text = str(timerange or "").strip()
    if not text or "-" not in text:
        return frame
    start_text, end_text = text.split("-", 1)
    out = frame
    if start_text:
        out = out.loc[out["date"] >= parse_timerange_date(start_text)]
    if end_text:
        out = out.loc[out["date"] < parse_timerange_date(end_text)]
    return out


def parse_timerange_date(value: str) -> pd.Timestamp:
    text = str(value).strip()
    if len(text) == 8 and text.isdigit():
        return pd.Timestamp(datetime.strptime(text, "%Y%m%d"), tz="UTC")
    return pd.Timestamp(text, tz="UTC")


def load_benchmarks(datadir: Path, benchmark: str, timeframes: list[str], timerange: str) -> dict[str, DataFrame]:
    keys = pair_filter_keys(benchmark)
    if not keys:
        return {}
    files = discover_candle_files(datadir, timeframes, keys, max_pairs=0)
    by_timeframe: dict[str, DataFrame] = {}
    for file in files:
        if file.timeframe in by_timeframe:
            continue
        try:
            by_timeframe[file.timeframe] = apply_timerange(load_candle_frame(file.path), timerange)
        except Exception:
            continue
    return by_timeframe


def add_selected_indicators(frame: DataFrame, selected: list[str], benchmark: DataFrame | None, args: argparse.Namespace) -> DataFrame:
    out = frame.copy()
    needs_pa = any(item in selected for item in ("pa", "tl", "pat"))
    needs_tl = any(item in selected for item in ("tl", "pat", "vc"))
    if needs_pa:
        out = add_pivot_structure(out)
    if needs_tl:
        out = add_trendline_projection(out, TrendlineProjectionConfig(missing_pivot_mode="skip"))
    if "vol" in selected:
        out = add_complex_volume_indicators(out)
    if "vp" in selected:
        out = add_volume_profile(
            out,
            VolumeProfileConfig(
                window=int(args.profile_window),
                bins=int(args.profile_bins),
                chunk_size=int(args.profile_chunk_size),
            ),
        )
    if "pat" in selected:
        out = add_pattern_structure(out)
    if "vc" in selected:
        out = add_volatility_cycles(out)
    if "rs" in selected and benchmark is not None and len(benchmark) > 0:
        out = add_relative_strength(out, benchmark)
    return out


def selected_score_columns(frame: DataFrame, selected: list[str], scope: str) -> list[str]:
    selected_set = set(selected)
    base_names = {
        f"{indicator}_{direction}"
        for indicator in selected
        for direction in ("score_long", "score_short", "score_abs")
    }
    return [
        str(column)
        for column in frame.columns
        if "_score_" in str(column)
        and score_indicator_key(str(column)) in selected_set
        and (scope == "all" or str(column) in base_names)
    ]


def score_indicator_key(column: str) -> str:
    return str(column).split("_", 1)[0]


def contract_checks(frame: DataFrame, score_columns: list[str], selected: list[str], context: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in score_columns:
        score = pd.to_numeric(frame[column], errors="coerce")
        finite = score.replace([np.inf, -np.inf], np.nan).dropna()
        nonzero_ratio = float((finite > 0.0).mean()) if len(finite) else 0.0
        normalized_pass = bool(len(finite) > 0 and finite.min() >= -1e-12 and finite.max() <= 1.0 + 1e-12)
        rows.append(
            {
                **context,
                "score": column,
                "rows": int(len(score)),
                "finite_rows": int(len(finite)),
                "min": safe_float(finite.min()),
                "max": safe_float(finite.max()),
                "mean": safe_float(finite.mean()),
                "nonzero_ratio": nonzero_ratio,
                "has_variation": bool(len(finite) > 0 and float(finite.max() - finite.min()) > 1e-9),
                "contract_pass": normalized_pass,
            }
        )
    selected_states = {f"{indicator}_state" for indicator in selected}
    for state_col in [column for column in frame.columns if str(column) in STATE_COLUMNS and str(column) in selected_states]:
        values = set(pd.to_numeric(frame[state_col], errors="coerce").dropna().unique().tolist())
        rows.append(
            {
                **context,
                "score": state_col,
                "rows": int(len(frame)),
                "finite_rows": int(frame[state_col].notna().sum()),
                "min": safe_float(pd.to_numeric(frame[state_col], errors="coerce").min()),
                "max": safe_float(pd.to_numeric(frame[state_col], errors="coerce").max()),
                "mean": safe_float(pd.to_numeric(frame[state_col], errors="coerce").mean()),
                "nonzero_ratio": safe_float((pd.to_numeric(frame[state_col], errors="coerce") != 0).mean()),
                "has_variation": bool(len(values) > 1),
                "contract_pass": values.issubset({-1.0, 0.0, 1.0}),
            }
        )
    return rows


def behavior_checks(frame: DataFrame, selected: list[str], context: dict[str, Any], threshold: float) -> list[dict[str, Any]]:
    specs = {
        "pa_score_long": ["pa_ms_up_sequence_score", "pa_ms_state", "pa_trend_bias"],
        "pa_score_short": ["pa_ms_down_sequence_score", "pa_ms_state", "pa_trend_bias"],
        "tl_score_long": ["tl_support_quality", "tl_trend_bias", "tl_support_reclaim"],
        "tl_score_short": ["tl_resistance_quality", "tl_trend_bias", "tl_resistance_reject"],
        "vol_score_long": ["vol_cvd_trend_confirm_long", "vol_evr_bull_absorption", "vol_liq_stoprun_long", "vol_avwap_reclaim_long"],
        "vol_score_short": ["vol_cvd_trend_confirm_short", "vol_evr_bear_absorption", "vol_liq_stoprun_short", "vol_avwap_reject_short"],
        "vp_score_long": ["vp_poc_migration_score_long", "vp_vah_breakout_with_pressure", "vp_lower_rejection_with_pressure", "vp_lvn_accept_long"],
        "vp_score_short": ["vp_poc_migration_score_short", "vp_val_breakdown_with_pressure", "vp_upper_rejection_with_pressure", "vp_lvn_accept_short"],
        "pat_score_long": ["pat_impulse_up_score", "pat_range_contraction_score", "pat_flag_long", "pat_breakout_long"],
        "pat_score_short": ["pat_impulse_down_score", "pat_range_contraction_score", "pat_flag_short", "pat_breakout_short"],
        "vc_score_long": ["vc_compression_score", "vc_expansion_long", "vc_exhaustion_down"],
        "vc_score_short": ["vc_compression_score", "vc_expansion_short", "vc_exhaustion_up"],
        "rs_score_long": ["rs_ret_medium", "rs_percentile", "rs_outperforming"],
        "rs_score_short": ["rs_ret_medium", "rs_percentile", "rs_underperforming"],
    }
    rows: list[dict[str, Any]] = []
    selected_set = set(selected)
    for score_col, evidence_cols in specs.items():
        if score_indicator_key(score_col) not in selected_set:
            continue
        if score_col not in frame.columns:
            continue
        score = pd.to_numeric(frame[score_col], errors="coerce").fillna(0.0)
        high_mask = score >= threshold
        if not high_mask.any():
            rows.append({**context, "score": score_col, "evidence": "any", "high_rows": 0, "baseline": 0.0, "high_score": 0.0, "lift": 0.0, "alignment_pass": False})
            continue
        for evidence_col in evidence_cols:
            if evidence_col not in frame.columns:
                continue
            evidence = evidence_series(frame[evidence_col], score_col)
            baseline = safe_float(evidence.mean())
            high_value = safe_float(evidence.loc[high_mask].mean())
            lift = high_value - baseline
            rows.append(
                {
                    **context,
                    "score": score_col,
                    "evidence": evidence_col,
                    "high_rows": int(high_mask.sum()),
                    "baseline": baseline,
                    "high_score": high_value,
                    "lift": safe_float(lift),
                    "alignment_pass": bool(lift > 0.02 or high_value >= 0.55),
                }
            )
    return rows


def evidence_series(series: Series, score_col: str) -> Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype("float64")
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if score_col.endswith("_short") and series.name in {"pa_ms_state", "pa_trend_bias", "tl_trend_bias", "rs_ret_medium", "rs_percentile"}:
        if series.name == "rs_percentile":
            return (1.0 - numeric).clip(0.0, 1.0)
        return (numeric < 0.0).astype("float64")
    if series.name in {"pa_ms_state", "pa_trend_bias", "tl_trend_bias"}:
        return (numeric > 0.0).astype("float64")
    return normalize_observed(numeric)


def outcome_checks(
    frame: DataFrame,
    score_columns: list[str],
    forward_windows: list[int],
    deciles: int,
    context: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    decile_rows: list[dict[str, Any]] = []
    close = pd.to_numeric(frame["close"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    for window in forward_windows:
        future_return = close.shift(-window) / close - 1.0
        future_high = pd.concat([high.shift(-i) for i in range(1, window + 1)], axis=1).max(axis=1)
        future_low = pd.concat([low.shift(-i) for i in range(1, window + 1)], axis=1).min(axis=1)
        long_mfe = future_high / close - 1.0
        long_mae = (future_low / close - 1.0).clip(upper=0.0).abs()
        short_mfe = close / future_low - 1.0
        short_mae = (future_high / close - 1.0).clip(lower=0.0)

        for score_col in score_columns:
            score = pd.to_numeric(frame[score_col], errors="coerce")
            direction = score_direction(score_col)
            if direction == "long":
                expected = future_return
                mfe = long_mfe
                mae = long_mae
            elif direction == "short":
                expected = -future_return
                mfe = short_mfe
                mae = short_mae
            else:
                expected = future_return.abs()
                mfe = pd.concat([long_mfe, short_mfe], axis=1).max(axis=1)
                mae = pd.concat([long_mae, short_mae], axis=1).min(axis=1)
            valid = pd.DataFrame({"score": score, "expected": expected, "mfe": mfe, "mae": mae}).replace([np.inf, -np.inf], np.nan).dropna()
            if len(valid) < max(deciles * 5, 30):
                continue
            valid["bucket"] = score_buckets(valid["score"], deciles)
            grouped = valid.groupby("bucket", observed=True)
            bucket_stats = grouped.agg(
                rows=("score", "size"),
                score_mean=("score", "mean"),
                expected_mean=("expected", "mean"),
                expected_median=("expected", "median"),
                mfe_mean=("mfe", "mean"),
                mae_mean=("mae", "mean"),
            ).reset_index()
            for _, row in bucket_stats.iterrows():
                decile_rows.append(
                    {
                        **context,
                        "score": score_col,
                        "direction": direction,
                        "forward_window": window,
                        "bucket": int(row["bucket"]),
                        "rows": int(row["rows"]),
                        "score_mean": safe_float(row["score_mean"]),
                        "expected_mean": safe_float(row["expected_mean"]),
                        "expected_median": safe_float(row["expected_median"]),
                        "mfe_mean": safe_float(row["mfe_mean"]),
                        "mae_mean": safe_float(row["mae_mean"]),
                        "mfe_mae_ratio": safe_float(row["mfe_mean"] / row["mae_mean"]) if row["mae_mean"] else None,
                    }
                )
            metric_row = score_validity_metrics(bucket_stats, valid, score_col, direction, window, context)
            rows.append(metric_row)
    return rows, decile_rows


def score_direction(score_col: str) -> str:
    if score_col.endswith("_score_abs") or "_score_abs_" in score_col:
        return "abs"
    if score_col.endswith("_score_long") or "_score_long_" in score_col:
        return "long"
    if score_col.endswith("_score_short") or "_score_short_" in score_col:
        return "short"
    lowered = score_col.lower()
    if "_down_" in lowered or "_bear_" in lowered or "_short" in lowered:
        return "short"
    if "_up_" in lowered or "_bull_" in lowered or "_long" in lowered:
        return "long"
    return "abs"


def score_buckets(score: Series, deciles: int) -> Series:
    ranks = score.rank(method="first")
    buckets = pd.qcut(ranks, q=min(deciles, len(score)), labels=False, duplicates="drop")
    return pd.Series(buckets, index=score.index).astype("int64")


def score_validity_metrics(bucket_stats: DataFrame, valid: DataFrame, score_col: str, direction: str, window: int, context: dict[str, Any]) -> dict[str, Any]:
    bucket_stats = bucket_stats.sort_values("bucket")
    bucket_index = pd.to_numeric(bucket_stats["bucket"], errors="coerce")
    expected_mean = pd.to_numeric(bucket_stats["expected_mean"], errors="coerce")
    mfe_mae = pd.to_numeric(bucket_stats["mfe_mean"], errors="coerce") / pd.to_numeric(bucket_stats["mae_mean"], errors="coerce").replace(0.0, np.nan)
    corr = expected_mean.corr(bucket_index) if len(bucket_stats) > 2 else np.nan
    top = bucket_stats.iloc[-1]
    bottom = bucket_stats.iloc[0]
    top_bottom_edge = float(top["expected_mean"] - bottom["expected_mean"])
    top_mfe_mae = float(top["mfe_mean"] / top["mae_mean"]) if float(top["mae_mean"]) else np.nan
    bottom_mfe_mae = float(bottom["mfe_mean"] / bottom["mae_mean"]) if float(bottom["mae_mean"]) else np.nan
    monotonic_score = clip01(((corr if np.isfinite(corr) else 0.0) + 1.0) / 2.0)
    edge_score = 1.0 if top_bottom_edge > 0.0 else 0.0
    mfe_score = 1.0 if np.isfinite(top_mfe_mae) and np.isfinite(bottom_mfe_mae) and top_mfe_mae > bottom_mfe_mae else 0.0
    mean_score_corr = valid["score"].corr(valid["expected"]) if len(valid) > 2 else np.nan
    validity = clip01(0.40 * monotonic_score + 0.30 * edge_score + 0.20 * mfe_score + 0.10 * clip01(((mean_score_corr if np.isfinite(mean_score_corr) else 0.0) + 1.0) / 2.0))
    return {
        **context,
        "score": score_col,
        "direction": direction,
        "forward_window": window,
        "rows": int(len(valid)),
        "score_expected_corr": safe_float(mean_score_corr),
        "decile_expected_corr": safe_float(corr),
        "top_expected_mean": safe_float(top["expected_mean"]),
        "bottom_expected_mean": safe_float(bottom["expected_mean"]),
        "top_bottom_edge": safe_float(top_bottom_edge),
        "top_mfe_mae_ratio": safe_float(top_mfe_mae),
        "bottom_mfe_mae_ratio": safe_float(bottom_mfe_mae),
        "validity_score": safe_float(validity),
        "validity_label": validity_label(validity),
    }


def summarize_run(
    score_rows: list[dict[str, Any]],
    behavior_rows: list[dict[str, Any]],
    contract_rows: list[dict[str, Any]],
    failures: list[dict[str, str]],
    args: argparse.Namespace,
    selected: list[str],
    files: list[CandleFile],
) -> dict[str, Any]:
    score_df = pd.DataFrame(score_rows)
    behavior_df = pd.DataFrame(behavior_rows)
    contract_df = pd.DataFrame(contract_rows)
    top_scores: list[dict[str, Any]] = []
    if not score_df.empty:
        group = score_df.groupby("score", as_index=False).agg(
            avg_validity=("validity_score", "mean"),
            avg_edge=("top_bottom_edge", "mean"),
            tested_windows=("forward_window", "count"),
        )
        group = group.sort_values(["avg_validity", "avg_edge"], ascending=False)
        top_scores = records(group.head(20))
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "datadir": str(args.datadir),
            "pairs": args.pairs,
            "timeframes": args.timeframes,
            "timerange": args.timerange,
            "indicators": selected,
            "score_scope": args.score_scope,
            "benchmark": args.benchmark,
            "forward_windows": args.forward_windows,
            "deciles": args.deciles,
            "score_threshold": args.score_threshold,
            "max_pairs": args.max_pairs,
        },
        "files_discovered": len(files),
        "score_metric_rows": len(score_rows),
        "behavior_rows": len(behavior_rows),
        "contract_rows": len(contract_rows),
        "failures": failures,
        "contract_pass_rate": safe_float(contract_df["contract_pass"].mean()) if "contract_pass" in contract_df else None,
        "behavior_pass_rate": safe_float(behavior_df["alignment_pass"].mean()) if "alignment_pass" in behavior_df else None,
        "top_scores": top_scores,
    }


def print_report(summary: dict[str, Any], score_rows: list[dict[str, Any]], behavior_rows: list[dict[str, Any]], failures: list[dict[str, str]]) -> None:
    print("")
    print("Indicator Validation Summary")
    print(f"Files discovered: {summary['files_discovered']}")
    print(f"Score metric rows: {summary['score_metric_rows']}")
    print(f"Contract pass rate: {format_pct(summary.get('contract_pass_rate'))}")
    print(f"Behaviour pass rate: {format_pct(summary.get('behavior_pass_rate'))}")
    if failures:
        print(f"Failures/skips: {len(failures)}")
        for item in failures[:8]:
            print(f"  {Path(item['file']).name}: {item['reason']}")
    top_scores = summary.get("top_scores") or []
    if top_scores:
        print("")
        print("Top score candidates")
        for row in top_scores[:10]:
            print(
                f"  {row.get('score')}: validity={float(row.get('avg_validity') or 0):.3f} "
                f"edge={float(row.get('avg_edge') or 0):.5f} tests={int(row.get('tested_windows') or 0)}"
            )


def validity_label(value: float) -> str:
    if value >= 0.75:
        return "strong"
    if value >= 0.60:
        return "promising"
    if value >= 0.48:
        return "weak"
    return "poor"


def normalize_observed(series: Series) -> Series:
    numeric = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if numeric.min() >= 0.0 and numeric.max() <= 1.0:
        return numeric
    return clip01((numeric - numeric.rolling(240, min_periods=20).min()) / (numeric.rolling(240, min_periods=20).max() - numeric.rolling(240, min_periods=20).min()).replace(0.0, np.nan))


def clip01(value: Any) -> Any:
    if isinstance(value, Series):
        return pd.to_numeric(value, errors="coerce").replace([np.inf, -np.inf], np.nan).clip(0.0, 1.0).fillna(0.0)
    try:
        return min(1.0, max(0.0, float(value)))
    except Exception:
        return 0.0


def safe_float(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    if not math.isfinite(number):
        return None
    return number


def format_pct(value: Any) -> str:
    try:
        return f"{float(value) * 100.0:.1f}%"
    except Exception:
        return "-"


def records(frame: DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.replace({np.nan: None}).to_json(orient="records"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


if __name__ == "__main__":
    raise SystemExit(main())
