from __future__ import annotations

# Bound numerical libraries before importing pandas/FreqAI helpers.
# ruff: noqa: E402, E501
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
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation6 as g6f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation7 as g7f,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation8_cache as g8c,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation8_preflight as g8p,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_freqai_generation0 import (
    load_predictions,
)


USER_DATA_DIR = REPO_ROOT / "user_data"
DEFAULT_CONFIG = USER_DATA_DIR / "configs" / "config_market_reaction_zone_freqai.example.json"
DEFAULT_PYTHON = (
    REPO_ROOT / "runtime" / "venvs" / "freqtrade-backtest-08" / "Scripts" / "python.exe"
)
STRATEGY_PATH = USER_DATA_DIR / "strategies"
STRATEGY_FILE = STRATEGY_PATH / "MarketReactionZoneFreqAIGeneration8Strategy.py"
STRATEGY_CLASS = "MarketReactionZoneG8ConfigurableFreqAIResearchStrategy"
DATA_DIR = USER_DATA_DIR / "data" / "binance"
OUTPUT_ROOT = g8p.OUTPUT_ROOT
RECORD_ROOT = OUTPUT_ROOT / "generation8_branches" / "g8_freqai_attribution"
ARTIFACT_ROOT = (
    Path("D:/FreqTradeStuffLargeData")
    / "research_outputs"
    / "market_reaction_zones"
    / "generation8_branches"
    / "g8_freqai_attribution"
)
MAX_WORKERS = 4
MIN_TRAINING_ROWS = 30
MIN_PAIR_SCORABLE_ROWS = g7f.MIN_PAIR_SCORABLE_ROWS
INITIAL_SEED = 42
CONFIRMATION_SEEDS = (17, 73)


@dataclass(frozen=True)
class QuestionSpec:
    cell: g8p.CellSpec
    base_blocks: tuple[str, ...]
    dimensions: tuple[tuple[str, tuple[str, ...]], ...]
    targets: tuple[str, ...]
    plain_name: str

    @property
    def question_id(self) -> str:
        return self.cell.cell_id

    @property
    def ready_block(self) -> str:
        return g8c.cell_ready_name(self.cell)


def participation_blocks() -> tuple[str, ...]:
    return (
        "g8_participation_relative_volume",
        "g8_participation_volume_acceleration",
        "g8_participation_absolute_pressure",
        "g8_participation_pressure_persistence",
    )


def question_spec(cell: g8p.CellSpec, frozen: dict[str, Any]) -> QuestionSpec:
    branch = cell.branch_id.split("_", maxsplit=1)[0]
    level = ("g8_level_base",)
    participation = participation_blocks()
    if branch == "g8a":
        base = (*level, *participation, "g8_volatility_absolute", "g8_volatility_compression")
        dimensions = tuple(
            (name, (block,))
            for name, block in (
                ("identity", "g8_level_identity"),
                ("prominence", "g8_level_prominence"),
                ("width", "g8_level_width"),
                ("proximity", "g8_level_proximity"),
                ("timeframe_composition", "g8_level_timeframe"),
                ("age_persistence_touch_history", "g8_level_history"),
            )
        )
    elif branch == "g8b":
        base = (*level, "g8_volatility_absolute", "g8_volatility_compression")
        dimensions = (
            ("relative_volume", ("g8_participation_relative_volume",)),
            ("volume_acceleration", ("g8_participation_volume_acceleration",)),
            ("absolute_pressure", ("g8_participation_absolute_pressure",)),
            ("pressure_persistence", ("g8_participation_pressure_persistence",)),
            ("participation_duration", ("g8_participation_duration",)),
        )
    elif branch == "g8c":
        base = (*level, *participation)
        dimensions = (
            ("absolute_volatility", ("g8_volatility_absolute",)),
            ("compression", ("g8_volatility_compression",)),
            ("volatility_state_duration", ("g8_volatility_duration",)),
        )
    elif branch == "g8d":
        base = (*level, *participation)
        dimensions = (
            ("persistent_trend_strength", ("g8_trend_persistent",)),
            ("return_acceleration", ("g8_trend_acceleration",)),
            ("oscillator_displacement", ("g8_momentum_oscillator",)),
            ("trend_state_duration", ("g8_trend_duration",)),
        )
    elif branch == "g8e":
        base = (*level, *participation)
        dimensions = (
            ("btc_activity_1h", ("g8_btc_horizon_1h",)),
            ("btc_activity_4h", ("g8_btc_horizon_4h",)),
            ("btc_activity_24h", ("g8_btc_horizon_24h",)),
            ("btc_relative_volume", ("g8_btc_relative_volume",)),
            ("btc_activity_duration", ("g8_btc_duration",)),
        )
    elif branch == "g8f":
        base = (*level, *participation)
        dimensions = (
            ("four_hour_location", ("g8_timeframe_4h",)),
            ("eight_hour_location", ("g8_timeframe_8h",)),
            ("daily_location", ("g8_timeframe_1d",)),
            ("room_and_obstacles", ("g8_room_geometry",)),
        )
    elif branch == "g8g":
        base = (
            *level,
            "g8_btc_horizon_1h",
            "g8_btc_horizon_4h",
            "g8_btc_horizon_24h",
            "g8_btc_relative_volume",
        )
        dimensions = (
            ("displayed_pressure", ("g8_orderbook_pressure",)),
            ("coverage_and_freshness", ("g8_orderbook_coverage",)),
            ("pressure_intensity_regime", ("g8_orderbook_regime",)),
            ("pressure_persistence", ("g8_orderbook_persistence",)),
        )
    elif branch == "g8h":
        parent = {
            "participation_volatility": (
                "g8_volatility_absolute",
                "g8_volatility_compression",
            ),
            "participation_trend": (
                "g8_trend_persistent",
                "g8_trend_acceleration",
                "g8_momentum_oscillator",
            ),
            "participation_btc": (
                "g8_btc_horizon_1h",
                "g8_btc_horizon_4h",
                "g8_btc_horizon_24h",
                "g8_btc_relative_volume",
            ),
        }[cell.surface]
        base = (*level, *participation, *parent)
        dimensions = (
            ("first_contact", ("g8_touch_first",)),
            ("continued_occupancy", ("g8_touch_occupancy",)),
            ("retest_state", ("g8_touch_retest",)),
            ("age_persistence_prior_contacts", ("g8_level_history",)),
        )
    else:
        raise ValueError(f"Unknown Generation 8 branch {branch!r}")
    return QuestionSpec(
        cell=cell,
        base_blocks=tuple(dict.fromkeys(base)),
        dimensions=dimensions,
        targets=tuple(str(target) for target in frozen["targets"]),
        plain_name=str(frozen["plain_name"]),
    )


def load_frozen() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    return g8p.load_frozen_batch()


FROZEN, FROZEN_BRANCHES = load_frozen()
QUESTIONS = tuple(
    question_spec(cell, FROZEN_BRANCHES[cell.branch_id]) for cell in g8p.CELLS
)


def placebo_blocks(blocks: Sequence[str], variant: str) -> tuple[str, ...]:
    if variant not in {"stale", "shuffled"}:
        raise ValueError(f"Unknown Generation 8 placebo {variant!r}")
    return tuple(f"{block}_{variant}" for block in blocks)


def profile_id(question: QuestionSpec, role: str, seed: int) -> str:
    return f"{question.question_id}__{role}__seed{seed}"


def build_registry(
    seeds: Sequence[int] = (INITIAL_SEED,),
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    profiles: dict[str, dict[str, Any]] = {}
    comparisons: list[dict[str, Any]] = []
    for question in QUESTIONS:
        dimension_map = dict(question.dimensions)
        all_dimension_blocks = tuple(
            dict.fromkeys(
                block for _, blocks in question.dimensions for block in blocks
            )
        )
        for seed in seeds:
            roles: dict[str, tuple[str, ...]] = {
                "baseline": question.base_blocks,
                "complete": (*question.base_blocks, *all_dimension_blocks),
                "complete_all_stale": (
                    *question.base_blocks,
                    *placebo_blocks(all_dimension_blocks, "stale"),
                ),
                "complete_all_shuffled": (
                    *question.base_blocks,
                    *placebo_blocks(all_dimension_blocks, "shuffled"),
                ),
            }
            for name, blocks in question.dimensions:
                roles[f"add_{name}"] = (*question.base_blocks, *blocks)
                roles[f"add_{name}_stale"] = (
                    *question.base_blocks,
                    *placebo_blocks(blocks, "stale"),
                )
                roles[f"add_{name}_shuffled"] = (
                    *question.base_blocks,
                    *placebo_blocks(blocks, "shuffled"),
                )
                others = tuple(
                    block
                    for other, other_blocks in question.dimensions
                    if other != name
                    for block in other_blocks
                )
                roles[f"complete_leave_{name}"] = (*question.base_blocks, *others)
            for role, blocks in roles.items():
                identifier = profile_id(question, role, seed)
                profiles[identifier] = {
                    "profile_id": identifier,
                    "question_id": question.question_id,
                    "branch_id": question.cell.branch_id,
                    "cohort": question.cell.cohort,
                    "surface": question.cell.surface,
                    "role": role,
                    "seed": int(seed),
                    "blocks": list(dict.fromkeys(blocks)),
                    "required_ready_blocks": [question.ready_block],
                    "targets": list(question.targets),
                    "plain_name": question.plain_name,
                    "pair_scope": question.cell.pair_scope,
                }
            for name in dimension_map:
                candidate = profile_id(question, f"add_{name}", seed)
                for baseline_role, control_type in (
                    ("baseline", "immediately_simpler"),
                    (f"add_{name}_stale", "causal_stale"),
                    (f"add_{name}_shuffled", "within_period_nonself_shuffle"),
                ):
                    comparisons.append(
                        {
                            "comparison_id": (
                                f"{question.question_id}__{name}__vs_{baseline_role}__seed{seed}"
                            ),
                            "question_id": question.question_id,
                            "branch_id": question.cell.branch_id,
                            "cohort": question.cell.cohort,
                            "surface": question.cell.surface,
                            "route_id": name,
                            "route_type": "add_one_dimension",
                            "control_type": control_type,
                            "candidate": candidate,
                            "baseline": profile_id(question, baseline_role, seed),
                            "baseline_role": baseline_role,
                            "seed": int(seed),
                            "targets": list(question.targets),
                            "plain_name": question.plain_name,
                            "expected_controls_for_route": 3,
                        }
                    )
            complete_candidate = profile_id(question, "complete", seed)
            complete_controls = [
                ("baseline", "all_dimensions_vs_base"),
                ("complete_all_stale", "all_dimensions_causal_stale"),
                ("complete_all_shuffled", "all_dimensions_nonself_shuffle"),
                *(
                    (f"complete_leave_{name}", f"leave_one_out_{name}")
                    for name in dimension_map
                ),
            ]
            expected = len(complete_controls)
            for baseline_role, control_type in complete_controls:
                comparisons.append(
                    {
                        "comparison_id": (
                            f"{question.question_id}__complete__vs_{baseline_role}__seed{seed}"
                        ),
                        "question_id": question.question_id,
                        "branch_id": question.cell.branch_id,
                        "cohort": question.cell.cohort,
                        "surface": question.cell.surface,
                        "route_id": "complete_attribution",
                        "route_type": "complete_leave_one_out_ladder",
                        "control_type": control_type,
                        "candidate": complete_candidate,
                        "baseline": profile_id(question, baseline_role, seed),
                        "baseline_role": baseline_role,
                        "seed": int(seed),
                        "targets": list(question.targets),
                        "plain_name": question.plain_name,
                        "expected_controls_for_route": expected,
                    }
                )
    return profiles, comparisons


PROFILES, COMPARISONS = build_registry()


def cache_manifest(cohort: str) -> dict[str, Any]:
    path = g8c.RECORD_ROOT / f"{cohort}_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed_generation8_attribution_shared_cache":
        raise ValueError(f"Generation 8 {cohort} cache is not terminal.")
    integrity = manifest.get("integrity", {})
    for key in (
        "causal_level_timestamp_violations",
        "stale_timestamp_violations",
        "shuffled_self_matches",
        "profit_features",
        "future_signed_direction_features",
    ):
        if integrity.get(key) != 0:
            raise ValueError(f"Generation 8 cache integrity failure: {key}")
    return manifest


def profiles_for_cohort(cohort: str) -> tuple[str, ...]:
    return tuple(
        profile_id
        for profile_id, profile in PROFILES.items()
        if profile["cohort"] == cohort
    )


def smoke_profiles(cohort: str) -> tuple[str, ...]:
    selected: list[str] = []
    for question in QUESTIONS:
        if question.cell.cohort != cohort:
            continue
        for role in ("baseline", "complete"):
            selected.append(profile_id(question, role, INITIAL_SEED))
    return tuple(selected)


def allowed_pairs(cohort: str) -> tuple[str, ...]:
    return tuple(str(pair) for pair in cache_manifest(cohort)["pairs"])


def select_pairs(allowed: Sequence[str], requested: str) -> tuple[str, ...]:
    if requested.strip().lower() == "all":
        return tuple(allowed)
    requested_parts = set(g0.split_csv(requested))
    selected = tuple(
        pair
        for pair in allowed
        if pair in requested_parts or g0.pair_file_stem(pair) in requested_parts
    )
    if not selected:
        raise ValueError("No requested Generation 8 pairs are in the frozen cohort.")
    return selected


def profile_pairs(profile: dict[str, Any], pairs: Sequence[str]) -> tuple[str, ...]:
    if profile["pair_scope"] == "all":
        return tuple(pairs)
    if profile["pair_scope"] == "smart_contract_platforms":
        return tuple(pair for pair in pairs if pair in g8p.SMART_CONTRACT_PLATFORMS)
    raise ValueError(f"Unknown pair scope {profile['pair_scope']!r}")


def profile_config(
    base: dict[str, Any],
    *,
    identifier: str,
    pairs: Sequence[str],
    feature_dir: Path,
    event_dir: Path,
    profile: dict[str, Any],
    train_days: int,
    backtest_days: int,
    technical_smoke: bool,
) -> dict[str, Any]:
    config = g6f.profile_config(
        base,
        identifier=identifier,
        pairs=pairs,
        feature_dir=feature_dir,
        event_dir=event_dir,
        profile=profile,
        train_days=train_days,
        backtest_days=backtest_days,
        technical_smoke=technical_smoke,
    )
    research = config.pop("market_reaction_zone_g6")
    research["target_columns"] = list(profile["targets"])
    research["frozen_branch_id"] = profile["branch_id"]
    research["question_id"] = profile["question_id"]
    research["profile_role"] = profile["role"]
    research["model_seed"] = int(profile["seed"])
    config["market_reaction_zone_g8"] = research
    config["freqai"]["data_split_parameters"]["random_state"] = int(profile["seed"])
    config["freqai"]["model_training_parameters"]["random_state"] = int(
        profile["seed"]
    )
    return config


def active_comparisons(
    profile_ids: Sequence[str],
    *,
    registry_comparisons: Sequence[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    selected = set(profile_ids)
    available = COMPARISONS if registry_comparisons is None else registry_comparisons
    return [
        comparison
        for comparison in available
        if comparison["candidate"] in selected and comparison["baseline"] in selected
    ]


def build_manifest(
    *,
    run_id: str,
    cohort: str,
    pairs: Sequence[str],
    profile_ids: Sequence[str],
    base_config: Path,
    python_exe: Path,
    profile_workers: int,
    technical_smoke: bool,
    registry_profiles: dict[str, dict[str, Any]] | None = None,
    registry_comparisons: Sequence[dict[str, Any]] | None = None,
    run_stage: str = "initial_seed_attribution",
    frozen_selection_path: Path | None = None,
) -> tuple[dict[str, Any], Path, Path]:
    record_dir = RECORD_ROOT / run_id
    artifact_dir = ARTIFACT_ROOT / run_id
    manifest_path = record_dir / "g8_freqai_run_manifest.json"
    if manifest_path.is_file():
        return (
            json.loads(manifest_path.read_text(encoding="utf-8")),
            manifest_path,
            artifact_dir,
        )
    record_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    shared = cache_manifest(cohort)
    inventory = [item for item in shared["inventory"] if item["pair"] in pairs]
    feature_dir = Path(inventory[0]["feature_path"]).parent
    event_dir = Path(inventory[0]["event_path"]).parent
    settings = dict(g7f.cohort_settings(cohort))
    timerange = str(settings["timerange"])
    if technical_smoke:
        if cohort == "normal":
            timerange = "20250401-20250701"
            settings = {**settings, "train_days": 365, "backtest_days": 90}
        else:
            timerange = "20260401-20260516"
            settings = {**settings, "train_days": 160, "backtest_days": 45}
    base = json.loads(base_config.read_text(encoding="utf-8"))
    available_profiles = PROFILES if registry_profiles is None else registry_profiles
    available_comparisons = (
        COMPARISONS if registry_comparisons is None else registry_comparisons
    )
    commands: list[dict[str, Any]] = []
    active: list[str] = []
    for requested_id in profile_ids:
        profile = available_profiles[requested_id]
        subset = profile_pairs(profile, pairs)
        if not subset:
            continue
        active.append(requested_id)
        short_id = f"p{len(active):03d}_{g6f.stable_digest(requested_id, 6)}"
        profile_dir = artifact_dir / "profiles" / short_id
        userdir = profile_dir / "user_data"
        export_dir = profile_dir / "export"
        identifier = f"g8-{g6f.stable_digest(f'{run_id}|{requested_id}', 16)}"
        config_path = record_dir / f"config_{short_id}.json"
        g0.atomic_write_json(
            profile_config(
                base,
                identifier=identifier,
                pairs=subset,
                feature_dir=feature_dir,
                event_dir=event_dir,
                profile=profile,
                train_days=int(settings["train_days"]),
                backtest_days=int(settings["backtest_days"]),
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
            STRATEGY_CLASS,
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
                **profile,
                "short_id": short_id,
                "pairs": list(subset),
                "strategy": STRATEGY_CLASS,
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
    comparisons = active_comparisons(
        active, registry_comparisons=available_comparisons
    )
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "prepared",
        "created_at_utc": g0.utc_now(),
        "generation": 8,
        "run_stage": run_stage,
        "technical_smoke_not_evidence": bool(technical_smoke),
        "cohort": cohort,
        "pairs": list(pairs),
        "timerange": timerange,
        "train_period_days": int(settings["train_days"]),
        "backtest_period_days": int(settings["backtest_days"]),
        "validation_periods": list(settings["validation_periods"]),
        "profile_workers": int(profile_workers),
        "model_threads_per_profile": 1,
        "model_class": "LightGBMRegressorMultiTarget",
        "profiles": active,
        "profile_count": len(active),
        "comparisons": comparisons,
        "comparison_count": len(comparisons),
        "questions": sorted(
            {available_profiles[item]["question_id"] for item in active}
        ),
        "seeds": sorted({int(available_profiles[item]["seed"]) for item in active}),
        "profile_registry_frozen_before_model_outcomes": True,
        "profile_registry": {
            "profiles": {item: available_profiles[item] for item in active},
            "comparisons": comparisons,
        },
        "research_boundary": {
            "profit_used": False,
            "future_signed_direction": False,
            "entry_exit_construction": False,
            "strategy_promotion": False,
        },
        "source_contracts": {
            "base_config": str(base_config.resolve()),
            "base_config_sha256": g0.sha256_file(base_config),
            "python_executable": str(python_exe.resolve()),
            "strategy": str(STRATEGY_FILE.resolve()),
            "strategy_sha256": g0.sha256_file(STRATEGY_FILE),
            "cache_manifest": str(
                (g8c.RECORD_ROOT / f"{cohort}_manifest.json").resolve()
            ),
            "cache_manifest_sha256": g0.sha256_file(
                g8c.RECORD_ROOT / f"{cohort}_manifest.json"
            ),
            "cache_inventory": inventory,
            "frozen_selection": (
                None
                if frozen_selection_path is None
                else {
                    "path": str(frozen_selection_path.resolve()),
                    "sha256": g0.sha256_file(frozen_selection_path),
                }
            ),
        },
        "storage": {
            "record_dir": str(record_dir),
            "bulky_artifact_dir": str(artifact_dir),
            "feature_cache_dir": str(feature_dir),
            "event_cache_dir": str(event_dir),
            "save_backtest_models": False,
        },
        "runtime_preparation_snapshot": g7f.runtime_snapshot(),
        "commands": commands,
    }
    g0.atomic_write_json(manifest, manifest_path)
    return manifest, manifest_path, artifact_dir


def preflight_run(  # noqa: C901 - runtime and per-profile gates remain explicit
    manifest: dict[str, Any], *, python_exe: Path
) -> dict[str, Any]:
    problems: list[str] = []
    if not python_exe.is_file():
        problems.append(f"Missing worker interpreter: {python_exe}")
    if not STRATEGY_FILE.is_file():
        problems.append(f"Missing Generation 8 strategy: {STRATEGY_FILE}")
    dependency: dict[str, Any] = {}
    if python_exe.is_file():
        check = subprocess.run(
            [
                str(python_exe),
                "-c",
                "import freqtrade, lightgbm, sklearn; print(freqtrade.__version__)",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        dependency = {
            "returncode": check.returncode,
            "stdout": check.stdout.strip(),
            "stderr": check.stderr.strip(),
        }
        if check.returncode:
            problems.append("The worker cannot import Freqtrade and LightGBM.")
    for item in manifest["source_contracts"]["cache_inventory"]:
        for key in ("feature", "event", "evaluation"):
            path = Path(item[f"{key}_path"])
            if not path.is_file() or g0.sha256_file(path) != item[f"{key}_sha256"]:
                problems.append(f"Cache hash mismatch for {item['pair']} and {key}")
    frozen_selection = manifest["source_contracts"].get("frozen_selection")
    if frozen_selection:
        selection_path = Path(frozen_selection["path"])
        if (
            not selection_path.is_file()
            or g0.sha256_file(selection_path) != frozen_selection["sha256"]
        ):
            problems.append("Frozen confirmation selection hash mismatch.")
    start_raw, end_raw = str(manifest["timerange"]).split("-", maxsplit=1)
    prediction_start = pd.Timestamp(start_raw, tz="UTC")
    prediction_end = pd.Timestamp(end_raw, tz="UTC")
    training_start = prediction_start - pd.Timedelta(
        days=int(manifest["train_period_days"])
    )
    event_dir = Path(manifest["storage"]["event_cache_dir"])
    readiness: list[dict[str, Any]] = []
    requirements = {
        (
            tuple(item["pairs"]),
            tuple(item["required_ready_blocks"]),
            tuple(item["targets"]),
        )
        for item in manifest["commands"]
    }
    for pair_subset, ready_blocks, targets in sorted(requirements):
        for pair in pair_subset:
            ready_columns = [f"ready__{block}" for block in ready_blocks]
            frame = pd.read_parquet(
                event_dir / f"{g0.pair_file_stem(pair)}.parquet",
                columns=["date", "period", *targets, *ready_columns],
            )
            frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
            eligible = frame[ready_columns].fillna(False).astype(bool).all(axis=1)
            eligible &= frame[list(targets)].notna().all(axis=1)
            training_rows = int(
                (
                    eligible
                    & frame["date"].ge(training_start)
                    & frame["date"].lt(prediction_start)
                ).sum()
            )
            prediction_rows = int(
                (
                    eligible
                    & frame["date"].ge(prediction_start)
                    & frame["date"].lt(prediction_end)
                ).sum()
            )
            readiness.append(
                {
                    "pair": pair,
                    "ready_blocks": ",".join(ready_blocks),
                    "targets": ",".join(targets),
                    "training_rows": training_rows,
                    "prediction_rows": prediction_rows,
                }
            )
            if training_rows < MIN_TRAINING_ROWS:
                problems.append(
                    f"{pair} has only {training_rows} training rows for {ready_blocks}."
                )
            if prediction_rows < MIN_PAIR_SCORABLE_ROWS:
                problems.append(
                    f"{pair} has only {prediction_rows} prediction rows for {ready_blocks}."
                )
    free_gib = shutil.disk_usage(Path(manifest["storage"]["bulky_artifact_dir"])).free / (
        1024**3
    )
    if free_gib < 20.0:
        problems.append(f"Only {free_gib:.2f} GiB free on the artifact drive.")
    return {
        "created_at_utc": g0.utc_now(),
        "passed": not problems,
        "problems": problems,
        "dependency_check": dependency,
        "readiness_audit": readiness,
        "profile_workers": manifest["profile_workers"],
        "model_threads_per_profile": 1,
        "runtime_snapshot": g7f.runtime_snapshot(),
        "bulky_storage_free_gib": round(free_gib, 3),
    }


def load_event_targets(manifest: dict[str, Any]) -> DataFrame:
    frames = []
    event_dir = Path(manifest["storage"]["event_cache_dir"])
    targets = sorted(
        {target for item in manifest["commands"] for target in item["targets"]}
    )
    for pair in manifest["pairs"]:
        frame = pd.read_parquet(
            event_dir / f"{g0.pair_file_stem(pair)}.parquet",
            columns=["date", "period", *targets],
        )
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        frame["pair"] = pair
        frames.append(frame)
    output = pd.concat(frames, ignore_index=True)
    if output.duplicated(["pair", "date"]).any():
        raise ValueError("Duplicate pair/date keys in Generation 8 targets.")
    return output


def score_comparisons(
    *,
    manifest: dict[str, Any],
    predictions: dict[str, DataFrame],
    actual: DataFrame,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    pair_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    eligibility_rows: list[dict[str, Any]] = []
    for definition in manifest["comparisons"]:
        candidate_id = str(definition["candidate"])
        baseline_id = str(definition["baseline"])
        targets = tuple(str(target) for target in definition["targets"])
        fair, audit = g7f.fair_comparison_frame(
            candidate=predictions[candidate_id],
            baseline=predictions[baseline_id],
            actual=actual,
            targets=targets,
        )
        eligibility_rows.append({**definition, **audit})
        for (pair, period), frame in fair.groupby(
            ["pair", "period"], sort=False, observed=True
        ):
            for target in targets:
                pair_rows.append(
                    {
                        **definition,
                        "pair": pair,
                        "period": period,
                        "target": target,
                        **g7f.regression_metrics(
                            frame,
                            target=target,
                            candidate_column=f"{target}__candidate",
                            baseline_column=f"{target}__baseline",
                        ),
                    }
                )
        comparison_pairs = sorted(fair["pair"].unique())
        for group in g7f.group_definitions(str(manifest["cohort"]), comparison_pairs):
            for period in manifest["validation_periods"]:
                for target in targets:
                    group_rows.append(
                        g7f.group_comparison_row(
                            fair,
                            manifest=manifest,
                            definition=definition,
                            target=target,
                            period=period,
                            group=group,
                        )
                    )
    return (
        DataFrame.from_records(pair_rows),
        DataFrame.from_records(group_rows),
        DataFrame.from_records(eligibility_rows),
    )


def route_decisions(manifest: dict[str, Any], scores: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    keys = [
        "question_id",
        "branch_id",
        "surface",
        "route_id",
        "route_type",
        "target",
        "group_id",
        "group_members",
        "seed",
    ]
    expected_periods = set(manifest["validation_periods"])
    for key, frame in scores.groupby(keys, dropna=False):
        values = dict(zip(keys, key, strict=True))
        expected_controls = int(frame["expected_controls_for_route"].iloc[0])
        controls = int(frame["comparison_id"].nunique())
        periods_complete = all(
            set(group["period"]) == expected_periods
            for _, group in frame.groupby("comparison_id", observed=True)
        )
        strict = (
            controls == expected_controls
            and periods_complete
            and frame["strict_period_pass"].fillna(False).astype(bool).all()
        )
        provisional = (
            controls == expected_controls
            and periods_complete
            and frame["provisional_period_pass"].fillna(False).astype(bool).all()
        )
        confirmation = manifest.get("run_stage") == "seed_confirmation"
        if strict and confirmation:
            status = "strict_confirmation_seed_pass"
        elif provisional and confirmation:
            status = "provisional_confirmation_seed_pass"
        elif confirmation:
            status = "not_retained_at_confirmation_seed"
        elif strict:
            status = "strict_seed42_candidate_for_seed_confirmation"
        elif provisional:
            status = "provisional_seed42_candidate_for_seed_confirmation"
        else:
            status = "not_retained_at_initial_seed"
        failed = frame.loc[
            ~frame["strict_period_pass"].fillna(False).astype(bool),
            [
                "comparison_id",
                "control_type",
                "period",
                "rows",
                "positive_coins",
                "equal_coin_paired_mae_gain",
                "bootstrap_lower",
                "not_dominated_by_one_coin",
            ],
        ]
        rows.append(
            {
                **values,
                "plain_name": frame["plain_name"].iloc[0],
                "status": status,
                "expected_controls": expected_controls,
                "controls_present": controls,
                "both_validation_periods_present": periods_complete,
                "all_controls_strict": strict,
                "all_controls_point_positive": provisional,
                "failed_strict_checks": json.dumps(
                    failed.to_dict("records"),
                    sort_keys=True,
                    default=g0.json_default,
                ),
                "requires_confirmation_seeds": (
                    not confirmation and status != "not_retained_at_initial_seed"
                ),
            }
        )
    return DataFrame.from_records(rows)


def score_run(
    manifest: dict[str, Any], *, record_dir: Path, artifact_dir: Path
) -> dict[str, Any]:
    predictions: dict[str, DataFrame] = {}
    audits: list[dict[str, Any]] = []
    for item in manifest["commands"]:
        frame, audit = load_predictions(
            Path(item["model_dir"]), tuple(str(pair) for pair in item["pairs"])
        )
        audit["profile_id"] = item["profile_id"]
        audits.append(audit)
        if frame.empty:
            raise ValueError(f"No predictions for Generation 8 profile {item['profile_id']}")
        missing = sorted(set(item["targets"]).difference(frame.columns))
        if missing:
            raise ValueError(f"Profile {item['profile_id']} lacks targets: {missing}")
        predictions[str(item["profile_id"])] = frame
    actual = load_event_targets(manifest)
    pair_scores, group_scores, eligibility = score_comparisons(
        manifest=manifest,
        predictions=predictions,
        actual=actual,
    )
    decisions = route_decisions(manifest, group_scores)
    score_paths = {
        "prediction_audit": record_dir / "g8_freqai_prediction_audit.csv",
        "comparison_eligibility": record_dir / "g8_freqai_comparison_eligibility.csv",
        "pair_scores": record_dir / "g8_freqai_pair_scores.csv",
        "group_scores": record_dir / "g8_freqai_group_scores.csv",
        "route_decisions": record_dir / "g8_freqai_route_decisions.csv",
    }
    g0.atomic_write_csv(DataFrame.from_records(audits), score_paths["prediction_audit"])
    g0.atomic_write_csv(eligibility, score_paths["comparison_eligibility"])
    g0.atomic_write_csv(pair_scores, score_paths["pair_scores"])
    g0.atomic_write_csv(group_scores, score_paths["group_scores"])
    g0.atomic_write_csv(decisions, score_paths["route_decisions"])
    unequal = int((~eligibility["identical_prediction_keys"]).sum()) if len(eligibility) else 0
    retained = decisions.loc[decisions["requires_confirmation_seeds"].astype(bool)]
    run_stage = str(manifest.get("run_stage", "initial_seed_attribution"))
    strict_count = int(
        decisions["all_controls_strict"].fillna(False).astype(bool).sum()
    )
    provisional_count = int(
        (
            decisions["all_controls_point_positive"].fillna(False).astype(bool)
            & ~decisions["all_controls_strict"].fillna(False).astype(bool)
        ).sum()
    )
    result = {
        "schema_version": 1,
        "run_id": manifest["run_id"],
        "status": (
            "completed_generation8_technical_smoke_not_evidence"
            if manifest["technical_smoke_not_evidence"]
            else (
                "completed_generation8_seed_confirmation"
                if run_stage == "seed_confirmation"
                else "completed_generation8_initial_seed_attribution"
            )
        ),
        "created_at_utc": g0.utc_now(),
        "technical_smoke_not_evidence": manifest["technical_smoke_not_evidence"],
        "cohort": manifest["cohort"],
        "run_stage": run_stage,
        "seeds": manifest["seeds"],
        "profiles_completed": len(predictions),
        "questions_completed": len(manifest["questions"]),
        "comparisons_completed": len(manifest["comparisons"]),
        "routes_scored": len(decisions),
        "strict_route_decisions": strict_count,
        "provisional_route_decisions": provisional_count,
        "strict_routes_pending_seed_confirmation": (
            strict_count if run_stage == "initial_seed_attribution" else 0
        ),
        "provisional_routes_pending_seed_confirmation": (
            provisional_count if run_stage == "initial_seed_attribution" else 0
        ),
        "retained_question_routes": sorted(
            {
                f"{row.question_id}|{row.route_id}|{row.target}|{row.group_id}"
                for row in retained.itertuples(index=False)
            }
        ),
        "integrity": {
            "unequal_prediction_key_comparisons": unequal,
            "duplicate_prediction_rows_removed": int(
                sum(item.get("duplicate_pair_date_rows", 0) for item in audits)
            ),
            "profit_used": False,
            "future_signed_direction": False,
            "all_profile_commands_terminal": all(
                item.get("status") == "completed" for item in manifest["commands"]
            ),
            "profile_registry_frozen_before_model_outcomes": manifest[
                "profile_registry_frozen_before_model_outcomes"
            ],
        },
        "confirmation_rule": (
            "Only seed-42 routes that pass every predeclared control in both validation "
            "periods may be rerun with seeds 17 and 73. No failed route is tuned."
            if run_stage == "initial_seed_attribution"
            else (
                "The exact frozen seed-42 candidates are repeated without changing "
                "features, controls, groups, targets, periods, or thresholds."
            )
        ),
        "artifacts": {
            key: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": g0.sha256_file(path),
            }
            for key, path in score_paths.items()
        },
        "interpretation_boundary": (
            "Direction-neutral reaction-magnitude attribution only; not causation, future "
            "direction, profit, an entry, an exit, or a trading rule."
        ),
    }
    result_path = record_dir / "g8_freqai_result.json"
    g0.atomic_write_json(result, result_path)
    return {**result, "result_path": str(result_path)}


def parse_profile_selection(cohort: str, raw: str, technical_smoke: bool) -> tuple[str, ...]:
    available = profiles_for_cohort(cohort)
    if technical_smoke and raw.strip().lower() == "all":
        return smoke_profiles(cohort)
    if raw.strip().lower() == "all":
        return available
    requested = set(g0.split_csv(raw))
    selected = tuple(
        item
        for item in available
        if item in requested
        or PROFILES[item]["branch_id"] in requested
        or PROFILES[item]["question_id"] in requested
    )
    if not selected:
        raise ValueError("No Generation 8 profiles matched the request.")
    return selected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Generation 8 FreqAI attribution ladders against simpler, "
            "causal stale, and deterministic nonself-shuffled controls."
        )
    )
    parser.add_argument("--cohort", choices=("normal", "meme"), required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--pairs", default="all")
    parser.add_argument("--profiles", default="all")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--python-exe", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--profile-workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--technical-smoke", action="store_true")
    args = parser.parse_args(argv)
    if not args.base_config.is_file():
        raise FileNotFoundError(args.base_config)
    if not args.python_exe.is_file():
        raise FileNotFoundError(args.python_exe)
    workers = max(1, min(int(args.profile_workers), MAX_WORKERS))
    pairs = select_pairs(allowed_pairs(args.cohort), args.pairs)
    profiles = parse_profile_selection(
        args.cohort, args.profiles, args.technical_smoke
    )
    manifest, manifest_path, artifact_dir = build_manifest(
        run_id=args.run_id,
        cohort=args.cohort,
        pairs=pairs,
        profile_ids=profiles,
        base_config=args.base_config,
        python_exe=args.python_exe,
        profile_workers=workers,
        technical_smoke=args.technical_smoke,
    )
    audit = preflight_run(manifest, python_exe=args.python_exe)
    audit_path = manifest_path.parent / "g8_freqai_preflight.json"
    g0.atomic_write_json(audit, audit_path)
    if not audit["passed"]:
        manifest["status"] = "blocked_preflight"
        manifest["preflight"] = audit
        g0.atomic_write_json(manifest, manifest_path)
        print(json.dumps(audit, indent=2, sort_keys=True))
        return 2
    returncode = g7f.run_manifest(manifest, manifest_path)
    if returncode:
        return returncode
    manifest["status"] = "profiles_completed"
    manifest["finished_at_utc"] = g0.utc_now()
    g0.atomic_write_json(manifest, manifest_path)
    result = score_run(
        manifest,
        record_dir=manifest_path.parent,
        artifact_dir=artifact_dir,
    )
    manifest["status"] = result["status"]
    manifest["result"] = result
    g0.atomic_write_json(manifest, manifest_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
