from __future__ import annotations

# Keep numerical libraries inside the user's processor budget even when this
# orchestrator is launched from an environment with broader thread defaults.
# ruff: noqa: E402
import os


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
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation0 import (
    DEFAULT_MANIFEST as GENERATION0_MANIFEST,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation0 import (
    atomic_write_json,
    ohlcv_path,
    pair_file_stem,
    sha256_file,
    validate_cache_metadata,
)
from user_data.strategies.MarketReactionZoneFreqAIResearchStrategy import (
    SELECTED_LEVEL_COLUMNS,
    SELECTED_MTF_LEVEL_TIMEFRAMES,
    TARGET_HORIZONS,
    build_market_reaction_targets,
)


USER_DATA_DIR = REPO_ROOT / "user_data"
DEFAULT_CONFIG = USER_DATA_DIR / "configs" / "config_market_reaction_zone_freqai.example.json"
DEFAULT_PYTHON = (
    REPO_ROOT / "runtime" / "venvs" / "freqtrade-backtest-08" / "Scripts" / "python.exe"
)
DEFAULT_RECORD_ROOT = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "market_reaction_zones"
    / "generation0_freqai"
)
DEFAULT_ARTIFACT_ROOT = (
    Path("D:/FreqTradeStuffLargeData")
    / "research_outputs"
    / "market_reaction_zones"
    / "generation0_freqai"
)
EVENT_ROOT = (
    Path("D:/FreqTradeStuffLargeData")
    / "research_outputs"
    / "market_reaction_zones"
    / "generation0_events"
)
STRATEGY_PATH = USER_DATA_DIR / "strategies"
DATA_DIR = USER_DATA_DIR / "data" / "binance"

DEFAULT_PAIRS = (
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "BNB/USDT:USDT",
    "SOL/USDT:USDT",
    "XRP/USDT:USDT",
    "ADA/USDT:USDT",
    "DOGE/USDT:USDT",
    "TRX/USDT:USDT",
    "AVAX/USDT:USDT",
    "LINK/USDT:USDT",
)

PROFILES: dict[str, dict[str, Any]] = {
    "market_state": {
        "strategy": "MarketReactionZoneMarketStateFreqAIResearchStrategy",
        "level_shift_hours": 0,
        "role": (
            "OHLCV, volume, pressure, volatility, and non-overlapping fixed-indicator "
            "baseline without candidate-level features."
        ),
    },
    "level_only_native": {
        "strategy": "MarketReactionZoneLevelOnlyFreqAIResearchStrategy",
        "level_shift_hours": 0,
        "role": (
            "Native-1h candidate-level identity, proximity, and attributes without "
            "broad market-state features."
        ),
    },
    "combined_native": {
        "strategy": "MarketReactionZoneCombinedFreqAIResearchStrategy",
        "level_shift_hours": 0,
        "role": "Market state plus causally current native-1h candidate-level information.",
    },
    "combined_native_placebo": {
        "strategy": "MarketReactionZoneCombinedFreqAIResearchStrategy",
        "level_shift_hours": 168,
        "role": (
            "The combined native model with level information delayed by 168 hours "
            "while market state and targets stay unchanged."
        ),
    },
    "combined_mtf": {
        "strategy": "MarketReactionZoneCombinedMtfFreqAIResearchStrategy",
        "level_shift_hours": 0,
        "role": (
            "Market state plus causally current 1h, 4h, 8h, and 1d candidate-level "
            "information."
        ),
    },
    "combined_mtf_placebo": {
        "strategy": "MarketReactionZoneCombinedMtfFreqAIResearchStrategy",
        "level_shift_hours": 168,
        "role": "The multi-timeframe combined model with every level block delayed by 168 hours.",
    },
}

TARGETS = tuple(
    target
    for horizon in TARGET_HORIZONS
    for target in (
        f"&-future_abs_excursion_{horizon}h_atr",
        f"&-future_range_ratio_{horizon}h",
        f"&-future_volume_ratio_{horizon}h",
        f"&-future_pressure_change_magnitude_{horizon}h",
    )
)

SCORING_WINDOWS = {
    "development_after_initial_training": (
        pd.Timestamp("2022-06-01", tz="UTC"),
        pd.Timestamp("2024-01-01", tz="UTC"),
    ),
    "validation_early": (
        pd.Timestamp("2024-01-01", tz="UTC"),
        pd.Timestamp("2025-04-01", tz="UTC"),
    ),
    "validation_late": (
        pd.Timestamp("2025-04-01", tz="UTC"),
        pd.Timestamp("2026-04-01", tz="UTC"),
    ),
    "previously_exposed_diagnostic": (
        pd.Timestamp("2026-04-01", tz="UTC"),
        pd.Timestamp("2026-07-20", tz="UTC"),
    ),
}

EXPANSION_LEVELS = {
    "rolling_high_24",
    "rolling_low_24",
    "rolling_high_168",
    "rolling_low_168",
    "rolling_high_720",
    "rolling_low_720",
    "bb20_upper",
    "bb20_lower",
    "round_nearest",
    "lvn_above",
    "lvn_below",
}
ACCEPTANCE_LEVELS = {"hvn_above", "hvn_below", "poc", "prior_poc"}


def utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def target_metadata(target: str) -> tuple[str, int]:
    if "abs_excursion" in target:
        family = "absolute_price_excursion"
    elif "range_ratio" in target:
        family = "range_activity"
    elif "volume_ratio" in target:
        family = "volume_activity"
    elif "pressure_change_magnitude" in target:
        family = "pressure_change_magnitude"
    else:
        raise ValueError(f"Unknown market-reaction target: {target}")
    for horizon in TARGET_HORIZONS:
        if f"_{horizon}h" in target:
            return family, horizon
    raise ValueError(f"Target lacks a declared horizon: {target}")


def profile_config(
    base: dict[str, Any],
    *,
    identifier: str,
    pairs: tuple[str, ...],
    level_shift_hours: int,
    train_days: int,
    backtest_days: int,
    technical_smoke: bool,
) -> dict[str, Any]:
    config = json.loads(json.dumps(base))
    config["exchange"]["pair_whitelist"] = list(pairs)
    freqai = config["freqai"]
    freqai["enabled"] = True
    freqai["identifier"] = identifier
    freqai["train_period_days"] = int(train_days)
    freqai["backtest_period_days"] = int(backtest_days)
    freqai["save_backtest_models"] = False
    freqai["feature_parameters"]["plot_feature_importances"] = 0
    freqai["data_split_parameters"] = {
        "test_size": 0.2,
        "random_state": 42,
        "shuffle": False,
    }
    training = freqai["model_training_parameters"]
    training["n_jobs"] = 1
    if technical_smoke:
        training["n_estimators"] = min(20, int(training.get("n_estimators", 100)))
    config["market_reaction_zone_freqai"]["level_feature_shift_hours"] = int(level_shift_hours)
    return config


def build_manifest(
    *,
    run_id: str,
    record_dir: Path,
    artifact_dir: Path,
    base_config: Path,
    python_exe: Path,
    profiles: tuple[str, ...],
    pairs: tuple[str, ...],
    timerange: str,
    train_days: int,
    backtest_days: int,
    technical_smoke: bool,
) -> dict[str, Any]:
    base = json.loads(base_config.read_text(encoding="utf-8"))
    commands: list[dict[str, Any]] = []
    for profile_id in profiles:
        definition = PROFILES[profile_id]
        profile_dir = artifact_dir / "profiles" / profile_id
        userdir = profile_dir / "user_data"
        export_dir = profile_dir / "export"
        identifier = f"{run_id}_{profile_id}"
        config_path = record_dir / f"config_{profile_id}.json"
        atomic_write_json(
            profile_config(
                base,
                identifier=identifier,
                pairs=pairs,
                level_shift_hours=int(definition["level_shift_hours"]),
                train_days=train_days,
                backtest_days=backtest_days,
                technical_smoke=technical_smoke,
            ),
            config_path,
        )
        command = [
            str(python_exe),
            "-m",
            "freqtrade",
            "backtesting",
            "--userdir",
            str(userdir),
            "--strategy-path",
            str(STRATEGY_PATH),
            "--datadir",
            str(DATA_DIR),
            "--config",
            str(config_path),
            "--strategy",
            str(definition["strategy"]),
            "--freqaimodel",
            "LightGBMRegressorMultiTarget",
            "--timerange",
            timerange,
            "--export",
            "signals",
            "--export-directory",
            str(export_dir),
            "--cache",
            "none",
        ]
        commands.append(
            {
                "profile_id": profile_id,
                "strategy": definition["strategy"],
                "role": definition["role"],
                "level_shift_hours": int(definition["level_shift_hours"]),
                "identifier": identifier,
                "config_path": str(config_path),
                "artifact_dir": str(profile_dir),
                "user_data_dir": str(userdir),
                "model_dir": str(userdir / "models" / identifier),
                "export_dir": str(export_dir),
                "command": command,
                "status": "pending",
                "attempts": 0,
            }
        )
    return {
        "schema_version": 1,
        "run_id": run_id,
        "status": "prepared",
        "created_at_utc": utc_now(),
        "technical_smoke_not_evidence": bool(technical_smoke),
        "objective": (
            "Generation 0C: test whether selected causal level identity, proximity, "
            "and attributes improve unseen non-directional market-reaction estimation "
            "beyond ordinary market state."
        ),
        "hypothesis": (
            "Current level information should improve ranking or error for later "
            "absolute movement, range, volume, or pressure-change magnitude on "
            "identical unseen rows, especially at real contact episodes, and that "
            "improvement should weaken when level features are delayed by 168 hours."
        ),
        "support_interpretation": (
            "Keep as a lead only when current combined features add coherent unseen-data "
            "information beyond market state and the matching delayed-level placebo "
            "across multiple chronological windows and either a rational multi-coin "
            "cohort or repeated BTC evidence."
        ),
        "null_interpretation": (
            "If combined current levels do not beat market state, or do no better than "
            "delayed levels, the tested representation is redundant or uninformative; "
            "this does not prove every possible construction of the underlying level "
            "family is false."
        ),
        "direction_prediction": False,
        "profit_optimization": False,
        "timerange": timerange,
        "train_period_days": int(train_days),
        "backtest_period_days": int(backtest_days),
        "pairs": list(pairs),
        "targets": list(TARGETS),
        "scoring_windows": {
            key: [str(start), str(end)] for key, (start, end) in SCORING_WINDOWS.items()
        },
        "selected_level_columns": list(SELECTED_LEVEL_COLUMNS),
        "market_state_excludes_candidate_level_math": [
            "rolling 24h, 168h, and 720h high/low distances and range positions",
            "Bollinger 20 upper/lower position and width",
        ],
        "controls": [
            "market_state profile",
            "level-only native profile",
            "combined current native levels",
            "combined native levels delayed 168h",
            "combined current 1h/4h/8h/1d levels",
            "combined 1h/4h/8h/1d levels delayed 168h",
            "identical common prediction-key intersection",
            "direct-atlas actual-contact and matched-random evaluation scopes",
        ],
        "source_manifest": str(GENERATION0_MANIFEST),
        "source_manifest_sha256": sha256_file(GENERATION0_MANIFEST),
        "storage": {
            "compact_record_dir": str(record_dir),
            "bulky_artifact_dir": str(artifact_dir),
            "save_backtest_models": False,
            "cleanup": (
                "Technical smoke and failed temporary artifacts become removable only "
                "after the corresponding full replacement is verified."
            ),
        },
        "commands": commands,
    }


def validate_manifest_request(
    manifest: dict[str, Any],
    *,
    profiles: tuple[str, ...],
    pairs: tuple[str, ...],
    timerange: str,
    train_days: int,
    backtest_days: int,
    technical_smoke: bool,
    python_exe: Path,
) -> None:
    """Prevent a resumed run id from silently adopting different CLI settings."""
    actual_profiles = tuple(str(item["profile_id"]) for item in manifest["commands"])
    actual_workers = {str(Path(item["command"][0]).resolve()) for item in manifest["commands"]}
    expected_worker = str(python_exe.resolve())
    mismatches: list[str] = []
    checks = (
        ("profiles", actual_profiles, profiles),
        ("pairs", tuple(manifest["pairs"]), pairs),
        ("timerange", str(manifest["timerange"]), timerange),
        ("train_period_days", int(manifest["train_period_days"]), int(train_days)),
        ("backtest_period_days", int(manifest["backtest_period_days"]), int(backtest_days)),
        (
            "technical_smoke_not_evidence",
            bool(manifest["technical_smoke_not_evidence"]),
            bool(technical_smoke),
        ),
        ("worker_interpreter", actual_workers, {expected_worker}),
    )
    for name, actual, expected in checks:
        if actual != expected:
            mismatches.append(f"{name}: recorded={actual!r}, requested={expected!r}")
    if mismatches:
        detail = "; ".join(mismatches)
        raise ValueError(f"Run id {manifest['run_id']!r} has incompatible settings: {detail}")


def preflight(manifest: dict[str, Any], python_exe: Path) -> dict[str, Any]:
    problems: list[str] = []
    cache_audit: list[dict[str, Any]] = []
    pairs = tuple(str(pair) for pair in manifest["pairs"])
    if not python_exe.is_file():
        problems.append(f"missing worker interpreter: {python_exe}")
    for pair in pairs:
        path = ohlcv_path(pair, "1h")
        if not path.is_file():
            problems.append(f"missing 1h OHLCV: {path}")
        for timeframe in SELECTED_MTF_LEVEL_TIMEFRAMES:
            cache = (
                USER_DATA_DIR
                / "research_news_data"
                / "context_features"
                / "market_reaction_zones"
                / "generation0_cache"
                / f"{pair_file_stem(pair)}-{timeframe}-core-generic.parquet"
            )
            if not cache.is_file():
                problems.append(f"missing causal level cache: {cache}")
                continue
            try:
                metadata = validate_cache_metadata(cache, GENERATION0_MANIFEST)
                actual_rows = pq.ParquetFile(cache).metadata.num_rows
                expected_rows = int(metadata.get("rows", -1))
                if actual_rows != expected_rows:
                    problems.append(
                        f"causal level cache row mismatch: {cache} "
                        f"has {actual_rows}, metadata declares {expected_rows}"
                    )
                cache_audit.append(
                    {
                        "pair": pair,
                        "timeframe": timeframe,
                        "path": str(cache),
                        "rows": actual_rows,
                        "metadata_rows": expected_rows,
                        "source_sha256": metadata.get("source_sha256"),
                        "manifest_sha256": metadata.get("manifest_sha256"),
                    }
                )
            except (FileNotFoundError, ValueError, KeyError) as error:
                problems.append(str(error))
    bulky_root = Path(manifest["storage"]["bulky_artifact_dir"])
    d_free_gib = round(shutil.disk_usage(bulky_root.anchor).free / (1024**3), 3)
    dependency = None
    if python_exe.is_file():
        result = subprocess.run(
            [str(python_exe), "-c", "import lightgbm, freqtrade; print('ok')"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        dependency = {
            "returncode": int(result.returncode),
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
        if result.returncode != 0:
            problems.append("worker interpreter cannot import lightgbm and freqtrade")
    return {
        "created_at_utc": utc_now(),
        "pairs": len(pairs),
        "profiles": len(manifest["commands"]),
        "targets": len(manifest["targets"]),
        "worker_interpreter": str(python_exe),
        "dependency_check": dependency,
        "causal_cache_audit": cache_audit,
        "bulky_storage_free_gib": d_free_gib,
        "thread_limits": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "PYARROW_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "problems": problems,
        "passed": not problems,
    }


def run_manifest(manifest: dict[str, Any], manifest_path: Path) -> int:
    env = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "PYARROW_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        env[name] = "1"
    manifest["status"] = "running"
    manifest["orchestrator_pid"] = os.getpid()
    manifest.setdefault("started_at_utc", utc_now())
    atomic_write_json(manifest, manifest_path)
    total = len(manifest["commands"])
    for index, item in enumerate(manifest["commands"], start=1):
        if item.get("status") == "completed":
            continue
        artifact_dir = Path(item["artifact_dir"])
        artifact_dir.mkdir(parents=True, exist_ok=True)
        Path(item["user_data_dir"]).mkdir(parents=True, exist_ok=True)
        attempt = int(item.get("attempts", 0)) + 1
        stdout_path = artifact_dir / f"freqai_stdout_attempt_{attempt}.log"
        stderr_path = artifact_dir / f"freqai_stderr_attempt_{attempt}.log"
        item.update(
            {
                "status": "running",
                "attempts": attempt,
                "started_at_utc": utc_now(),
                "stdout_log": str(stdout_path),
                "stderr_log": str(stderr_path),
            }
        )
        atomic_write_json(manifest, manifest_path)
        with (
            stdout_path.open("w", encoding="utf-8") as stdout,
            stderr_path.open("w", encoding="utf-8") as stderr,
        ):
            result = subprocess.run(
                [str(part) for part in item["command"]],
                cwd=REPO_ROOT,
                env=env,
                stdout=stdout,
                stderr=stderr,
                text=True,
                check=False,
            )
        item["finished_at_utc"] = utc_now()
        item["returncode"] = int(result.returncode)
        item["status"] = "completed" if result.returncode == 0 else "failed"
        atomic_write_json(manifest, manifest_path)
        print(
            json.dumps(
                {
                    "phase": "freqai_profiles",
                    "processed": index,
                    "total": total,
                    "profile_id": item["profile_id"],
                    "status": item["status"],
                    "stderr_log": str(stderr_path),
                }
            ),
            flush=True,
        )
        if result.returncode != 0:
            manifest["status"] = "failed"
            manifest["failed_profile_id"] = item["profile_id"]
            manifest["finished_at_utc"] = utc_now()
            atomic_write_json(manifest, manifest_path)
            return int(result.returncode)
    return 0


def prediction_pair(path: Path, pairs: tuple[str, ...]) -> str | None:
    """Resolve a FreqAI prediction filename to one frozen pair."""
    stem = path.stem.lower()
    for pair in pairs:
        base = pair.strip().lower().split("/")[0].split(":")[0]
        token = re.escape(base)
        if re.search(rf"(?:^|_)cb_{token}(?:_|$)", stem) or re.search(
            rf"^{token}(?:_|$)", stem
        ):
            return pair
    return None


def score_values(valid: DataFrame, prediction: str, actual: str) -> dict[str, Any]:
    """Score one continuous reaction target without trade or direction metrics."""
    clean = valid[[prediction, actual]].apply(pd.to_numeric, errors="coerce").dropna()
    result: dict[str, Any] = {"rows": len(clean)}
    if len(clean) < 20 or clean[prediction].nunique() < 2 or clean[actual].nunique() < 2:
        return {**result, "status": "unscorable"}
    errors = clean[prediction] - clean[actual]
    residual_sum_squares = float(np.square(errors).sum())
    total_sum_squares = float(np.square(clean[actual] - clean[actual].mean()).sum())
    ranked = clean.sort_values(prediction)
    bucket = max(1, int(len(ranked) * 0.2))
    result.update(
        {
            "status": "scored",
            "actual_mean": float(clean[actual].mean()),
            "prediction_mean": float(clean[prediction].mean()),
            "prediction_bias": float(errors.mean()),
            "mean_absolute_error": float(errors.abs().mean()),
            "root_mean_squared_error": float(np.sqrt(np.square(errors).mean())),
            "r_squared": float(1.0 - (residual_sum_squares / total_sum_squares)),
            "prediction_actual_spearman": float(
                clean[prediction].corr(clean[actual], method="spearman")
            ),
            "top_quintile_actual_mean": float(ranked.tail(bucket)[actual].mean()),
            "bottom_quintile_actual_mean": float(ranked.head(bucket)[actual].mean()),
        }
    )
    result["top_minus_bottom"] = (
        result["top_quintile_actual_mean"] - result["bottom_quintile_actual_mean"]
    )
    return result


def load_predictions(model_dir: Path, pairs: tuple[str, ...]) -> tuple[DataFrame, dict[str, Any]]:
    prediction_dir = model_dir / "backtesting_predictions"
    files = sorted(prediction_dir.glob("*_prediction.feather"))
    frames: list[DataFrame] = []
    unidentified: list[str] = []
    for path in files:
        pair = prediction_pair(path, pairs)
        if pair is None and len(pairs) == 1:
            pair = pairs[0]
        if pair is None:
            unidentified.append(path.name)
            continue
        frame = pd.read_feather(path)
        frame["pair"] = pair
        frame["prediction_file"] = path.name
        frames.append(frame)
    if not frames:
        return DataFrame(), {
            "files": len(files),
            "unidentified_files": unidentified,
            "rows": 0,
        }
    output = pd.concat(frames, ignore_index=True)
    output["date"] = pd.to_datetime(output["date"], utc=True, errors="coerce")
    output = output.dropna(subset=["date"])
    duplicate_rows = int(output.duplicated(["pair", "date"]).sum())
    output = (
        output.drop_duplicates(["pair", "date"], keep="last")
        .sort_values(["pair", "date"])
        .reset_index(drop=True)
    )
    if "do_predict" in output:
        output = output.loc[pd.to_numeric(output["do_predict"], errors="coerce").eq(1.0)].copy()
    return output, {
        "files": len(files),
        "unidentified_files": unidentified,
        "duplicate_pair_date_rows": duplicate_rows,
        "rows": len(output),
    }


def eligibility_digest(frame: DataFrame) -> str:
    if frame.empty:
        return ""
    keys = frame[["pair", "date"]].copy()
    keys["date"] = keys["date"].astype("int64")
    hashed = pd.util.hash_pandas_object(keys, index=False).to_numpy(dtype=np.uint64)
    return hashlib.sha256(hashed.tobytes()).hexdigest()


def common_prediction_keys(
    predictions: dict[str, DataFrame], pairs: tuple[str, ...]
) -> tuple[DataFrame, DataFrame]:
    intersection: set[tuple[str, pd.Timestamp]] | None = None
    rows: list[dict[str, Any]] = []
    for profile_id, frame in predictions.items():
        keys = set(zip(frame["pair"], frame["date"], strict=False))
        intersection = keys if intersection is None else intersection.intersection(keys)
        rows.append(
            {
                "profile_id": profile_id,
                "prediction_rows": len(keys),
                "prediction_key_digest": eligibility_digest(frame),
                "pairs": int(frame["pair"].nunique()) if not frame.empty else 0,
                "first_date": str(frame["date"].min()) if not frame.empty else None,
                "last_date": str(frame["date"].max()) if not frame.empty else None,
            }
        )
    intersection = intersection or set()
    common = DataFrame(sorted(intersection), columns=["pair", "date"])
    for row in rows:
        row["common_rows"] = len(common)
        row["rows_excluded_for_fair_comparison"] = int(row["prediction_rows"] - len(common))
    missing_pairs = sorted(set(pairs).difference(common.get("pair", pd.Series(dtype=str))))
    audit = DataFrame(rows)
    audit.attrs["missing_pairs"] = missing_pairs
    return common, audit


def actual_targets(pair: str) -> DataFrame:
    raw = pd.read_feather(ohlcv_path(pair, "1h"))
    raw["date"] = pd.to_datetime(raw["date"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["date"]).drop_duplicates("date", keep="last")
    labelled = build_market_reaction_targets(raw.sort_values("date"))
    keep = ["date", *TARGETS]
    return labelled[keep].rename(columns={target: f"{target}_actual" for target in TARGETS})


def event_scope_dates(pair: str) -> dict[str, set[pd.Timestamp]]:
    pattern = f"{pair_file_stem(pair)}-*-core-generic-g0b1-g0b2-*.parquet"
    paths = sorted(EVENT_ROOT.glob(pattern))
    frames: list[DataFrame] = []
    for path in paths:
        available = set(pq.ParquetFile(path).schema.names)
        required = {
            "event_time",
            "control",
            "source_timeframe",
            "level_name",
            "zone_method",
        }
        if not required.issubset(available):
            continue
        frames.append(pd.read_parquet(path, columns=sorted(required)))
    if not frames:
        return {}
    events = pd.concat(frames, ignore_index=True)
    events["event_time"] = pd.to_datetime(events["event_time"], utc=True, errors="coerce")
    events = events.loc[
        events["zone_method"].eq("standard_base_atr")
        & events["level_name"].isin(EXPANSION_LEVELS | ACCEPTANCE_LEVELS)
    ].copy()

    def dates(control: str, names: set[str], timeframe: str | None = None) -> set[pd.Timestamp]:
        mask = events["control"].eq(control) & events["level_name"].isin(names)
        if timeframe is not None:
            mask &= events["source_timeframe"].eq(timeframe)
        return set(events.loc[mask, "event_time"].dropna())

    selected = EXPANSION_LEVELS | ACCEPTANCE_LEVELS
    scopes = {
        "all_predicted_rows": set(),
        "actual_selected_contact": dates("actual", selected),
        "actual_expansion_transit_contact": dates("actual", EXPANSION_LEVELS),
        "actual_acceptance_contact": dates("actual", ACCEPTANCE_LEVELS),
        "actual_rolling_168_contact": dates("actual", {"rolling_high_168", "rolling_low_168"}),
        "actual_lvn_contact": dates("actual", {"lvn_above", "lvn_below"}),
        "actual_hvn_poc_contact": dates("actual", {"hvn_above", "hvn_below", "poc", "prior_poc"}),
        "matched_random_expansion": dates("matched_random_time", EXPANSION_LEVELS),
        "matched_random_acceptance": dates("matched_random_time", ACCEPTANCE_LEVELS),
    }
    for timeframe in ("1h", "4h", "8h", "1d"):
        scopes[f"actual_selected_contact_source_{timeframe}"] = dates("actual", selected, timeframe)
    return scopes


def score_profiles(
    manifest: dict[str, Any], record_dir: Path, artifact_dir: Path
) -> dict[str, Any]:
    pairs = tuple(str(pair) for pair in manifest["pairs"])
    predictions: dict[str, DataFrame] = {}
    prediction_audit: list[dict[str, Any]] = []
    for item in manifest["commands"]:
        profile_id = str(item["profile_id"])
        frame, audit = load_predictions(Path(item["model_dir"]), pairs)
        audit["profile_id"] = profile_id
        prediction_audit.append(audit)
        if frame.empty:
            raise ValueError(f"No predictions found for completed profile {profile_id}")
        missing_targets = sorted(set(TARGETS).difference(frame.columns))
        if missing_targets:
            raise ValueError(
                f"Profile {profile_id} is missing prediction targets: {missing_targets}"
            )
        predictions[profile_id] = frame

    common_keys, eligibility = common_prediction_keys(predictions, pairs)
    if common_keys.empty:
        raise ValueError("FreqAI profiles have no common prediction keys")
    missing_pairs = sorted(set(pairs).difference(set(common_keys["pair"])))
    if missing_pairs:
        raise ValueError(f"Common prediction surface is missing pairs: {missing_pairs}")
    eligibility_path = record_dir / "g0c_eligibility_audit.parquet"
    eligibility.to_parquet(eligibility_path, index=False)

    actual_by_pair = {pair: actual_targets(pair) for pair in pairs}
    scopes_by_pair = {pair: event_scope_dates(pair) for pair in pairs}
    score_rows: list[dict[str, Any]] = []
    for profile_id, frame in predictions.items():
        fair = frame.merge(common_keys, on=["pair", "date"], how="inner")
        for pair in pairs:
            merged = fair.loc[fair["pair"].eq(pair)].merge(
                actual_by_pair[pair], on="date", how="left", validate="one_to_one"
            )
            scopes = scopes_by_pair[pair]
            for window, (start, end) in SCORING_WINDOWS.items():
                window_frame = merged.loc[merged["date"].ge(start) & merged["date"].lt(end)].copy()
                for scope, scope_dates in scopes.items():
                    scoped = (
                        window_frame
                        if scope == "all_predicted_rows"
                        else window_frame.loc[window_frame["date"].isin(scope_dates)]
                    )
                    for target in TARGETS:
                        family, horizon = target_metadata(target)
                        score_rows.append(
                            {
                                "profile_id": profile_id,
                                "pair": pair,
                                "window": window,
                                "scope": scope,
                                "target": target,
                                "target_family": family,
                                "horizon_hours": horizon,
                                **score_values(
                                    scoped,
                                    target,
                                    f"{target}_actual",
                                ),
                            }
                        )
    scores = DataFrame(score_rows)
    scores = attach_control_deltas(scores)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    detailed_path = artifact_dir / "g0c_pair_window_scores.parquet"
    scores.to_parquet(detailed_path, index=False)
    summary = summarize_scores(scores, manifest)
    summary_path = record_dir / "g0c_freqai_summary.parquet"
    summary.to_parquet(summary_path, index=False)
    prediction_audit_path = record_dir / "g0c_prediction_file_audit.json"
    atomic_write_json(
        {
            "created_at_utc": utc_now(),
            "profiles": prediction_audit,
            "common_prediction_rows": len(common_keys),
            "common_prediction_key_digest": eligibility_digest(common_keys),
            "missing_pairs": missing_pairs,
        },
        prediction_audit_path,
    )
    result = {
        "created_at_utc": utc_now(),
        "profiles": len(predictions),
        "pairs": len(pairs),
        "targets": len(TARGETS),
        "common_prediction_rows": len(common_keys),
        "detailed_score_rows": len(scores),
        "summary_rows": len(summary),
        "detailed_scores": str(detailed_path),
        "summary": str(summary_path),
        "eligibility_audit": str(eligibility_path),
        "prediction_file_audit": str(prediction_audit_path),
        "direction_prediction": False,
        "profit_optimization": False,
    }
    atomic_write_json(result, record_dir / "g0c_freqai_result_record.json")
    return result


def attach_control_deltas(scores: DataFrame) -> DataFrame:
    keys = ["pair", "window", "scope", "target"]
    metric_columns = [
        "prediction_actual_spearman",
        "top_minus_bottom",
        "mean_absolute_error",
        "root_mean_squared_error",
        "r_squared",
        "prediction_bias",
    ]
    baseline = scores.loc[scores["profile_id"].eq("market_state"), [*keys, *metric_columns]]
    baseline = baseline.rename(
        columns={column: f"market_state_{column}" for column in metric_columns}
    )
    output = scores.merge(baseline, on=keys, how="left", validate="many_to_one")
    for metric in ("prediction_actual_spearman", "top_minus_bottom", "r_squared"):
        output[f"{metric}_delta_vs_market_state"] = (
            output[metric] - output[f"market_state_{metric}"]
        )
    for metric in ("mean_absolute_error", "root_mean_squared_error"):
        output[f"{metric}_skill_vs_market_state"] = (
            output[f"market_state_{metric}"] - output[metric]
        )
    output["absolute_bias_skill_vs_market_state"] = (
        output["market_state_prediction_bias"].abs() - output["prediction_bias"].abs()
    )
    for current, placebo in (
        ("combined_native", "combined_native_placebo"),
        ("combined_mtf", "combined_mtf_placebo"),
    ):
        placebo_frame = scores.loc[
            scores["profile_id"].eq(placebo), [*keys, *metric_columns]
        ].rename(columns={column: f"{current}_placebo_{column}" for column in metric_columns})
        output = output.merge(placebo_frame, on=keys, how="left", validate="many_to_one")
        is_current = output["profile_id"].eq(current)
        for metric in (
            "prediction_actual_spearman",
            "top_minus_bottom",
            "r_squared",
        ):
            output.loc[is_current, f"{metric}_delta_vs_matching_placebo"] = (
                output.loc[is_current, metric]
                - output.loc[is_current, f"{current}_placebo_{metric}"]
            )
        for metric in ("mean_absolute_error", "root_mean_squared_error"):
            output.loc[is_current, f"{metric}_skill_vs_matching_placebo"] = (
                output.loc[is_current, f"{current}_placebo_{metric}"]
                - output.loc[is_current, metric]
            )
    return output


def summarize_scores(scores: DataFrame, manifest: dict[str, Any]) -> DataFrame:
    cohorts = manifest_cohorts(manifest)
    primary = scores.copy()
    primary["cohort"] = primary["pair"].map(cohorts)
    all_pairs = scores.copy()
    all_pairs["cohort"] = "all_pairs"
    non_btc = scores.loc[~scores["pair"].eq("BTC/USDT:USDT")].copy()
    non_btc["cohort"] = "all_non_btc"
    working = pd.concat([primary, all_pairs, non_btc], ignore_index=True)
    working = working.loc[working["status"].eq("scored")].copy()
    for column in (
        "prediction_actual_spearman_delta_vs_market_state",
        "mean_absolute_error_skill_vs_market_state",
        "top_minus_bottom_delta_vs_market_state",
        "prediction_actual_spearman_delta_vs_matching_placebo",
        "mean_absolute_error_skill_vs_matching_placebo",
        "top_minus_bottom_delta_vs_matching_placebo",
    ):
        if column in working:
            working[f"{column}_positive"] = working[column].gt(0.0).astype(float)
    keys = [
        "profile_id",
        "cohort",
        "scope",
        "target_family",
        "horizon_hours",
    ]
    grouped = working.groupby(keys, sort=False, dropna=False, observed=True)
    aggregations: dict[str, tuple[str, str]] = {
        "pair_window_tests": ("pair", "size"),
        "distinct_pairs": ("pair", "nunique"),
        "distinct_windows": ("window", "nunique"),
        "scored_rows": ("rows", "sum"),
        "median_spearman": ("prediction_actual_spearman", "median"),
        "median_top_minus_bottom": ("top_minus_bottom", "median"),
        "median_mae": ("mean_absolute_error", "median"),
        "median_spearman_delta_vs_market_state": (
            "prediction_actual_spearman_delta_vs_market_state",
            "median",
        ),
        "positive_spearman_delta_share_vs_market_state": (
            "prediction_actual_spearman_delta_vs_market_state_positive",
            "mean",
        ),
        "median_mae_skill_vs_market_state": (
            "mean_absolute_error_skill_vs_market_state",
            "median",
        ),
        "positive_mae_skill_share_vs_market_state": (
            "mean_absolute_error_skill_vs_market_state_positive",
            "mean",
        ),
        "median_top_bottom_delta_vs_market_state": (
            "top_minus_bottom_delta_vs_market_state",
            "median",
        ),
        "median_spearman_delta_vs_matching_placebo": (
            "prediction_actual_spearman_delta_vs_matching_placebo",
            "median",
        ),
        "positive_spearman_delta_share_vs_matching_placebo": (
            "prediction_actual_spearman_delta_vs_matching_placebo_positive",
            "mean",
        ),
        "median_mae_skill_vs_matching_placebo": (
            "mean_absolute_error_skill_vs_matching_placebo",
            "median",
        ),
        "positive_mae_skill_share_vs_matching_placebo": (
            "mean_absolute_error_skill_vs_matching_placebo_positive",
            "mean",
        ),
        "median_top_bottom_delta_vs_matching_placebo": (
            "top_minus_bottom_delta_vs_matching_placebo",
            "median",
        ),
    }
    usable = {
        output: source for output, source in aggregations.items() if source[0] in working.columns
    }
    return grouped.agg(**usable).reset_index()


def manifest_cohorts(manifest: dict[str, Any]) -> dict[str, str]:
    generation = json.loads(Path(manifest["source_manifest"]).read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    for cohort, pairs in generation["reporting_groups"]["coin_cohorts"].items():
        for pair in pairs:
            if pair in result:
                raise ValueError(f"Pair appears in more than one primary cohort: {pair}")
            result[pair] = cohort
    missing = sorted(set(manifest["pairs"]).difference(result))
    if missing:
        raise ValueError(f"Pairs lack a frozen reporting cohort: {missing}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the bounded Generation 0C market-reaction FreqAI ladder."
    )
    parser.add_argument("--python-exe", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--record-root", type=Path, default=DEFAULT_RECORD_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--profiles", default=",".join(PROFILES))
    parser.add_argument("--pairs", default=",".join(DEFAULT_PAIRS))
    parser.add_argument("--timerange", default="20220601-20260719")
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--backtest-days", type=int, default=180)
    parser.add_argument("--technical-smoke", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args()

    profiles = parse_csv(args.profiles)
    pairs = parse_csv(args.pairs)
    unknown_profiles = sorted(set(profiles).difference(PROFILES))
    unknown_pairs = sorted(set(pairs).difference(DEFAULT_PAIRS))
    if unknown_profiles:
        raise ValueError(f"Unknown profiles: {unknown_profiles}")
    if unknown_pairs:
        raise ValueError(f"Pairs are outside the frozen Generation 0 universe: {unknown_pairs}")
    if args.train_days <= 0 or args.backtest_days <= 0:
        raise ValueError("train-days and backtest-days must be positive")
    run_id = args.run_id or f"g0c_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    record_dir = args.record_root / run_id
    artifact_dir = args.artifact_root / run_id
    record_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = record_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = build_manifest(
            run_id=run_id,
            record_dir=record_dir,
            artifact_dir=artifact_dir,
            base_config=args.config,
            python_exe=args.python_exe,
            profiles=profiles,
            pairs=pairs,
            timerange=args.timerange,
            train_days=args.train_days,
            backtest_days=args.backtest_days,
            technical_smoke=args.technical_smoke,
        )
        atomic_write_json(manifest, manifest_path)
    validate_manifest_request(
        manifest,
        profiles=profiles,
        pairs=pairs,
        timerange=args.timerange,
        train_days=args.train_days,
        backtest_days=args.backtest_days,
        technical_smoke=args.technical_smoke,
        python_exe=args.python_exe,
    )
    audit = preflight(manifest, args.python_exe)
    atomic_write_json(audit, record_dir / "preflight.json")
    if not audit["passed"]:
        print(json.dumps(audit, indent=2))
        return 2
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return 0
    if not args.score_only:
        returncode = run_manifest(manifest, manifest_path)
        if returncode != 0:
            return returncode
    else:
        manifest["status"] = "scoring"
        atomic_write_json(manifest, manifest_path)
    result = score_profiles(manifest, record_dir, artifact_dir)
    manifest["status"] = "completed"
    manifest["finished_at_utc"] = utc_now()
    manifest["result"] = result
    atomic_write_json(manifest, manifest_path)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
