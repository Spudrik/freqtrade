from __future__ import annotations

# Keep numerical libraries single-threaded. Parallelism belongs to the FreqAI runner.
# ruff: noqa: E402
import os


for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "PYARROW_NUM_THREADS",
):
    os.environ[_name] = "1"

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_freqai_generation8 as g8,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation3_one_minute_replay as g3m,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation8_cache as g8c,
)


OUTPUT_ROOT = (
    REPO_ROOT
    / "user_data"
    / "research_news_data"
    / "context_features"
    / "market_reaction_zones"
)
DEFAULT_RUN_ID = "g9_frozen_siblings_20260821a"
RECORD_ROOT = OUTPUT_ROOT / "generation9_preflight"
FREEZE_PATH = (
    OUTPUT_ROOT
    / "generation9_review"
    / "g9_frozen_limited_multisource_and_one_minute_batch_20260821a.json"
)
G8_REVIEW_PATH = (
    OUTPUT_ROOT
    / "generation8_review"
    / "g8_three_seed_joint_review_20260821a"
    / "g8_three_seed_joint_review.json"
)
SEEDS = (42, 17, 73)
MODEL_CONTROL_VARIANTS = ("leave", "stale", "shuffled")
MODEL_INPUT_LIMIT = 3
EXPECTED_CONTROLS_PER_QUESTION = 9
SELECTION_SEED = "g9_evidence_triggered_one_minute_v1"
VOLUME_REACTION_THRESHOLD = 2.0
MINIMUM_HIGH_VOLUME_DURATION_HOURS = 2.0
INDEPENDENCE_HOURS = 7 * 24
ONE_MINUTE_TIMEFRAMES = ("1h", "4h")
FEATURE_WARMUP_HOURS = 24
PERIODS = {
    "normal": ("development", "validation_early", "validation_late"),
    "meme": (
        "meme_development",
        "meme_validation_early",
        "meme_validation_late",
    ),
}
VOLUME_TARGET = "&-g6_future_volume_ratio_h1"
RANGE_TARGET = "&-g6_future_range_ratio_h1"
EPISODE_FEATURE_COLUMNS = (
    "date",
    "g8_participation_relative_volume__relative_volume",
    "g8_participation_duration__relative_volume_band",
    "g8_participation_duration__relative_volume_band_duration_hours",
    "g8_touch_first__first_count",
)
RAW_ANCHOR_COLUMNS = (
    "pair",
    "source_timeframe",
    "level_family",
    "level_name",
    "level_column",
    "representation",
    "control",
    "zone_method",
    "base_index",
    "event_time",
    "period",
    "source_available_at",
    "source_open",
    "level_price",
    "zone_half_width",
    "zone_half_width_atr",
    "base_atr",
    "approach_state",
    "pre_distance_atr",
    "level_score",
    "level_identity",
    "contact_close_distance_atr",
)


@dataclass(frozen=True)
class ModelQuestion:
    question_id: str
    cohort: str
    plain_name: str
    mechanism: str
    blocks: tuple[str, str, str]
    target: str
    ready_block: str
    pair_scope: str = "all"
    result_group: str = "all_declared_groups"


MODEL_QUESTIONS = (
    ModelQuestion(
        question_id="g9a_level_relative_volume_participation_duration__normal",
        cohort="normal",
        plain_name=(
            "Does a calculated level, current relative volume, and the duration of the "
            "current participation state jointly improve next-hour volume estimates?"
        ),
        mechanism=(
            "A level locates the reaction area, current volume measures immediate "
            "participation, and duration distinguishes a sustained state from a one-hour spike."
        ),
        blocks=(
            "g8_level_base",
            "g8_participation_relative_volume",
            "g8_participation_duration",
        ),
        target=VOLUME_TARGET,
        ready_block="g8b_normal",
    ),
    ModelQuestion(
        question_id="g9a_level_relative_volume_participation_duration__meme",
        cohort="meme",
        plain_name=(
            "Does the same level, volume, and participation-duration relationship repeat "
            "across the frozen top-ten meme cohort?"
        ),
        mechanism=(
            "The normal-coin mechanism is repeated without pooling meme coins into the "
            "normal cohort."
        ),
        blocks=(
            "g8_level_base",
            "g8_participation_relative_volume",
            "g8_participation_duration",
        ),
        target=VOLUME_TARGET,
        ready_block="g8b_meme",
    ),
    ModelQuestion(
        question_id="g9b_level_relative_volume_first_contact__normal",
        cohort="normal",
        plain_name=(
            "Does knowing that a level contact is fresh add to the combination of a "
            "calculated level and current relative volume?"
        ),
        mechanism=(
            "A fresh first contact may concentrate resting interest that repeated occupancy "
            "has already consumed."
        ),
        blocks=(
            "g8_level_base",
            "g8_participation_relative_volume",
            "g8_touch_first",
        ),
        target=VOLUME_TARGET,
        ready_block="g8h_normal_participation_volatility",
    ),
    ModelQuestion(
        question_id="g9b_level_relative_volume_first_contact__meme",
        cohort="meme",
        plain_name=(
            "Does fresh contact add to level and relative-volume information for meme coins?"
        ),
        mechanism=(
            "The fresh-contact mechanism is tested separately in the higher-turnover meme "
            "cohort."
        ),
        blocks=(
            "g8_level_base",
            "g8_participation_relative_volume",
            "g8_touch_first",
        ),
        target=VOLUME_TARGET,
        ready_block="g8h_meme_participation_volatility",
    ),
    ModelQuestion(
        question_id="g9c_level_relative_volume_compression__normal",
        cohort="normal",
        plain_name=(
            "Does local compression help a calculated level and current relative volume "
            "estimate the next hour's price range?"
        ),
        mechanism=(
            "A compressed market may release more range when participation meets a known area."
        ),
        blocks=(
            "g8_level_base",
            "g8_participation_relative_volume",
            "g8_volatility_compression",
        ),
        target=RANGE_TARGET,
        ready_block="g8c_normal",
    ),
    ModelQuestion(
        question_id="g9d_level_relative_volume_trend_duration__normal",
        cohort="normal",
        plain_name=(
            "Does the duration of the current trend state help a level and relative volume "
            "estimate the next hour's price range?"
        ),
        mechanism=(
            "A mature persistent move and a newly forming move may react differently at the "
            "same calculated area."
        ),
        blocks=(
            "g8_level_base",
            "g8_participation_relative_volume",
            "g8_trend_duration",
        ),
        target=RANGE_TARGET,
        ready_block="g8d_normal",
    ),
    ModelQuestion(
        question_id="g9e_level_relative_volume_oscillator_displacement__normal",
        cohort="normal",
        plain_name=(
            "For smart-contract platforms, does momentum displacement add to a calculated "
            "level and current relative volume when estimating next-hour range?"
        ),
        mechanism=(
            "The retained smart-contract lead may reflect how stretched momentum is when "
            "active participation reaches a calculated area."
        ),
        blocks=(
            "g8_level_base",
            "g8_participation_relative_volume",
            "g8_momentum_oscillator",
        ),
        target=RANGE_TARGET,
        ready_block="g8d_normal",
        pair_scope="smart_contract_platforms",
        result_group="smart_contract_platforms",
    ),
)


def profile_id(question: ModelQuestion, role: str, seed: int) -> str:
    return f"{question.question_id}__{role}__seed{seed}"


def build_model_registry(
    seeds: Sequence[int] = SEEDS,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    profiles: dict[str, dict[str, Any]] = {}
    comparisons: list[dict[str, Any]] = []
    for question in MODEL_QUESTIONS:
        for seed in seeds:
            complete_id = profile_id(question, "complete", int(seed))
            roles: dict[str, tuple[str, ...]] = {"complete": question.blocks}
            for block_index, block in enumerate(question.blocks):
                short_name = block.removeprefix("g8_")
                roles[f"leave_{short_name}"] = tuple(
                    item for item in question.blocks if item != block
                )
                for variant in ("stale", "shuffled"):
                    replacement = f"{block}_{variant}"
                    roles[f"{variant}_{short_name}"] = tuple(
                        replacement if index == block_index else item
                        for index, item in enumerate(question.blocks)
                    )
            for role, blocks in roles.items():
                identifier = profile_id(question, role, int(seed))
                profiles[identifier] = {
                    "profile_id": identifier,
                    "question_id": question.question_id,
                    "branch_id": question.question_id.split("__", maxsplit=1)[0],
                    "cohort": question.cohort,
                    "surface": "limited_three_source",
                    "role": role,
                    "seed": int(seed),
                    "blocks": list(blocks),
                    "required_ready_blocks": [question.ready_block],
                    "targets": [question.target],
                    "plain_name": question.plain_name,
                    "mechanism": question.mechanism,
                    "pair_scope": question.pair_scope,
                    "result_group": question.result_group,
                }
            for role in roles:
                if role == "complete":
                    continue
                if role.startswith("leave_"):
                    control_type = "immediately_simpler_leave_one_input_out"
                elif role.startswith("stale_"):
                    control_type = "single_input_causal_stale"
                else:
                    control_type = "single_input_within_period_nonself_shuffle"
                comparisons.append(
                    {
                        "comparison_id": (
                            f"{question.question_id}__complete__vs_{role}__seed{seed}"
                        ),
                        "question_id": question.question_id,
                        "branch_id": question.question_id.split("__", maxsplit=1)[0],
                        "cohort": question.cohort,
                        "surface": "limited_three_source",
                        "route_id": "complete_three_source_model",
                        "route_type": "limited_three_source_attribution",
                        "control_type": control_type,
                        "candidate": complete_id,
                        "baseline": profile_id(question, role, int(seed)),
                        "baseline_role": role,
                        "seed": int(seed),
                        "targets": [question.target],
                        "plain_name": question.plain_name,
                        "mechanism": question.mechanism,
                        "result_group": question.result_group,
                        "expected_controls_for_route": EXPECTED_CONTROLS_PER_QUESTION,
                    }
                )
    return profiles, comparisons


def stable_hash(*parts: object) -> str:
    value = "|".join(str(part) for part in parts)
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def cache_inventory(cohort: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest = g8.cache_manifest(cohort)
    return manifest, [dict(item) for item in manifest["inventory"]]


def qualifying_event_rows(item: dict[str, Any], cohort: str) -> DataFrame:
    features = pd.read_parquet(item["feature_path"], columns=list(EPISODE_FEATURE_COLUMNS))
    targets = pd.read_parquet(
        item["event_path"], columns=["date", "period", VOLUME_TARGET]
    )
    features["date"] = pd.to_datetime(features["date"], utc=True, errors="raise")
    targets["date"] = pd.to_datetime(targets["date"], utc=True, errors="raise")
    merged = features.merge(targets, on="date", how="inner", validate="one_to_one")
    duration = pd.to_numeric(
        merged["g8_participation_duration__relative_volume_band_duration_hours"],
        errors="coerce",
    )
    volume_band = pd.to_numeric(
        merged["g8_participation_duration__relative_volume_band"], errors="coerce"
    )
    first_count = pd.to_numeric(
        merged["g8_touch_first__first_count"], errors="coerce"
    )
    future_volume = pd.to_numeric(merged[VOLUME_TARGET], errors="coerce")
    eligible = merged["period"].isin(PERIODS[cohort])
    eligible &= volume_band.eq(1.0)
    eligible &= duration.ge(MINIMUM_HIGH_VOLUME_DURATION_HOURS)
    eligible &= first_count.ge(1.0)
    eligible &= future_volume.ge(VOLUME_REACTION_THRESHOLD)
    selected = merged.loc[eligible].copy()
    selected["pair"] = str(item["pair"])
    selected["cohort"] = cohort
    return selected


def anchor_freshness(
    anchors: DataFrame, *, pair: str
) -> tuple[DataFrame, dict[str, int]]:
    if anchors.empty:
        return anchors.copy(), {
            "candidate_rows": 0,
            "availability_violations": 0,
            "missing_base_candles": 0,
            "current_contact_failures": 0,
            "prior_four_contact_failures": 0,
        }
    base = g0.load_ohlcv(g0.ohlcv_path(pair, "1h"))
    base = base.sort_values("date").drop_duplicates("date").set_index("date")
    rows: list[dict[str, Any]] = []
    availability_violations = 0
    missing_base = 0
    current_failures = 0
    prior_failures = 0
    for _, source in anchors.iterrows():
        # Series.to_dict preserves non-identifier research column names such as
        # ``&-g6_future_volume_ratio_h1``; namedtuples silently rename them.
        row = source.to_dict()
        event_time = pd.Timestamp(row["event_time"])
        available_at = pd.Timestamp(row["source_available_at"])
        availability_ok = bool(available_at <= event_time)
        availability_violations += int(not availability_ok)
        window_start = event_time - pd.Timedelta(hours=4)
        window = base.loc[(base.index >= window_start) & (base.index <= event_time)]
        complete_window = len(window) == 5 and event_time in window.index
        missing_base += int(not complete_window)
        price = float(row["level_price"])
        half_width = float(row["zone_half_width"])
        lower = price - half_width
        upper = price + half_width
        if complete_window:
            current = window.loc[event_time]
            current_contact = bool(
                float(current["high"]) >= lower and float(current["low"]) <= upper
            )
            prior = window.loc[window.index < event_time]
            prior_contact = bool(
                ((prior["high"] >= lower) & (prior["low"] <= upper)).any()
            )
        else:
            current_contact = False
            prior_contact = True
        current_failures += int(not current_contact)
        prior_failures += int(prior_contact)
        row.update(
            {
                "source_available_before_contact": availability_ok,
                "complete_five_hour_base_window": complete_window,
                "current_hour_contacts_frozen_zone": current_contact,
                "preceding_four_hours_contact_frozen_zone": prior_contact,
                "fresh_contact_verified": bool(
                    availability_ok
                    and complete_window
                    and current_contact
                    and not prior_contact
                ),
            }
        )
        rows.append(row)
    return DataFrame.from_records(rows), {
        "candidate_rows": len(anchors),
        "availability_violations": availability_violations,
        "missing_base_candles": missing_base,
        "current_contact_failures": current_failures,
        "prior_four_contact_failures": prior_failures,
    }


def build_anchor_pool() -> tuple[DataFrame, DataFrame, dict[str, Any]]:
    accepted: list[DataFrame] = []
    audit_parts: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    for cohort in ("normal", "meme"):
        _, inventory = cache_inventory(cohort)
        for item in inventory:
            events = qualifying_event_rows(item, cohort)
            source_counts[f"{cohort}|{item['pair']}"] = len(events)
            if events.empty:
                continue
            raw = pd.read_parquet(
                item["raw_provenance_path"], columns=list(RAW_ANCHOR_COLUMNS)
            )
            raw["event_time"] = pd.to_datetime(raw["event_time"], utc=True, errors="raise")
            raw["source_available_at"] = pd.to_datetime(
                raw["source_available_at"], utc=True, errors="raise"
            )
            candidate_dates = set(events["date"])
            raw = raw.loc[
                raw["control"].eq("actual")
                & raw["source_timeframe"].isin(ONE_MINUTE_TIMEFRAMES)
                & raw["event_time"].isin(candidate_dates)
            ].copy()
            raw = raw.merge(
                events.drop(columns=["pair", "cohort"]),
                left_on=["event_time", "period"],
                right_on=["date", "period"],
                how="inner",
                validate="many_to_one",
            )
            raw.drop(columns=["date"], inplace=True)
            raw["pair"] = str(item["pair"])
            raw["cohort"] = cohort
            checked, audit = anchor_freshness(raw, pair=str(item["pair"]))
            audit_parts.append({"cohort": cohort, "pair": item["pair"], **audit})
            accepted.append(checked)
    all_anchors = pd.concat(accepted, ignore_index=True)
    fresh = all_anchors.loc[all_anchors["fresh_contact_verified"]].copy()
    fresh["anchor_hash"] = fresh.apply(
        lambda row: stable_hash(
            SELECTION_SEED,
            row["cohort"],
            row["pair"],
            row["event_time"],
            row["source_timeframe"],
            row["level_family"],
            row["level_name"],
            row["level_price"],
            row["zone_half_width"],
        ),
        axis=1,
    )
    fresh["absolute_contact_distance_atr"] = pd.to_numeric(
        fresh["contact_close_distance_atr"], errors="coerce"
    ).abs()
    fresh["sortable_zone_width_atr"] = pd.to_numeric(
        fresh["zone_half_width_atr"], errors="coerce"
    )
    fresh.sort_values(
        [
            "cohort",
            "pair",
            "event_time",
            "source_timeframe",
            "absolute_contact_distance_atr",
            "sortable_zone_width_atr",
            "anchor_hash",
        ],
        na_position="last",
        inplace=True,
    )
    anchors = fresh.drop_duplicates(
        ["cohort", "pair", "event_time", "source_timeframe"], keep="first"
    ).copy()
    anchors["episode_id"] = anchors.apply(
        lambda row: "g9m_"
        + stable_hash(
            row["cohort"],
            row["pair"],
            row["event_time"],
            row["source_timeframe"],
            row["anchor_hash"],
        )[:20],
        axis=1,
    )
    anchors["selection_hash"] = anchors.apply(
        lambda row: stable_hash(SELECTION_SEED, row["episode_id"]), axis=1
    )
    anchors["cluster_causal_anchor_timeframe"] = anchors["source_timeframe"]
    independent = g3m.select_time_independent(anchors, hours=INDEPENDENCE_HOURS)
    independent.sort_values(
        ["cohort", "period", "source_timeframe", "selection_hash"], inplace=True
    )
    independent.reset_index(drop=True, inplace=True)
    audit_frame = DataFrame.from_records(audit_parts)
    audit_summary = {
        "qualifying_feature_rows_by_pair": source_counts,
        "raw_anchor_rows": len(all_anchors),
        "fresh_anchor_rows_before_one_per_event_timeframe": len(fresh),
        "one_anchor_per_event_timeframe_rows": len(anchors),
        "independent_rows": len(independent),
        "independence_hours": INDEPENDENCE_HOURS,
        "availability_violations": int(audit_frame["availability_violations"].sum()),
        "missing_base_candles": int(audit_frame["missing_base_candles"].sum()),
        "current_contact_failures": int(audit_frame["current_contact_failures"].sum()),
        "prior_four_contact_failures": int(
            audit_frame["prior_four_contact_failures"].sum()
        ),
    }
    return independent, audit_frame, audit_summary


def select_diagnostic_sample(pool: DataFrame) -> DataFrame:
    chosen: list[str] = []
    order: dict[str, int] = {}
    rank = 0
    for cohort in ("normal", "meme"):
        used_pairs: set[str] = set()
        cohort_pool = pool.loc[pool["cohort"].eq(cohort)]
        for period in PERIODS[cohort]:
            for timeframe in ONE_MINUTE_TIMEFRAMES:
                cell = cohort_pool.loc[
                    cohort_pool["period"].eq(period)
                    & cohort_pool["source_timeframe"].eq(timeframe)
                ].sort_values("selection_hash")
                if cell.empty:
                    raise ValueError(
                        f"No independent one-minute episode for {cohort}|{period}|{timeframe}."
                    )
                unused = cell.loc[~cell["pair"].isin(used_pairs)]
                selected = unused.iloc[0] if not unused.empty else cell.iloc[0]
                episode_id = str(selected["episode_id"])
                chosen.append(episode_id)
                order[episode_id] = rank
                used_pairs.add(str(selected["pair"]))
                rank += 1
        if len(used_pairs) != 6:
            raise ValueError(
                f"The {cohort} diagnostic replay did not retain six distinct pairs."
            )
    sample = pool.loc[pool["episode_id"].isin(chosen)].copy()
    sample["sample_selection_order"] = sample["episode_id"].map(order).astype(int)
    sample["selected_for_one_minute_replay"] = True
    sample["sample_stratum"] = sample.apply(
        lambda row: (
            f"{row['cohort']}|{row['period']}|{row['source_timeframe']}"
        ),
        axis=1,
    )
    return sample.sort_values("sample_selection_order").reset_index(drop=True)


def source_contracts() -> dict[str, Any]:
    if not G8_REVIEW_PATH.is_file():
        raise FileNotFoundError(G8_REVIEW_PATH)
    manifests = {}
    for cohort in ("normal", "meme"):
        path = g8c.RECORD_ROOT / f"{cohort}_manifest.json"
        manifests[cohort] = {
            "path": str(path.resolve()),
            "sha256": g0.sha256_file(path),
        }
    return {
        "generation8_terminal_joint_review": {
            "path": str(G8_REVIEW_PATH.resolve()),
            "sha256": g0.sha256_file(G8_REVIEW_PATH),
        },
        "generation8_cache_manifests": manifests,
    }


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


def validate_existing_freeze(path: Path) -> dict[str, Any]:
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if frozen.get("status") != "frozen_before_generation9_sibling_outcomes":
        raise ValueError(f"Generation 9 freeze is not terminal: {path}")
    for item in frozen["artifacts"].values():
        artifact_path = Path(item["path"])
        if not artifact_path.is_file() or g0.sha256_file(artifact_path) != item["sha256"]:
            raise ValueError(f"Frozen Generation 9 artifact changed: {artifact_path}")
    return frozen


def freeze_generation9(run_id: str = DEFAULT_RUN_ID) -> dict[str, Any]:
    if FREEZE_PATH.is_file():
        return validate_existing_freeze(FREEZE_PATH)
    run_dir = RECORD_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    profiles, comparisons = build_model_registry()
    if max(len(item["blocks"]) for item in profiles.values()) > MODEL_INPUT_LIMIT:
        raise ValueError("A Generation 9 model exceeds the three-source limit.")
    pool, anchor_audit, audit_summary = build_anchor_pool()
    sample = select_diagnostic_sample(pool)
    intervals = g3m.acquisition_intervals(sample)
    pool_path = run_dir / "g9_one_minute_qualifying_episode_pool.csv"
    sample_path = run_dir / "g9_one_minute_diagnostic_sample.csv"
    interval_path = run_dir / "g9_one_minute_acquisition_intervals.csv"
    anchor_audit_path = run_dir / "g9_one_minute_anchor_audit.csv"
    g0.atomic_write_csv(pool, pool_path)
    g0.atomic_write_csv(sample, sample_path)
    g0.atomic_write_csv(intervals, interval_path)
    g0.atomic_write_csv(anchor_audit, anchor_audit_path)
    request = {
        "run_id": run_id,
        "model_questions": [asdict(question) for question in MODEL_QUESTIONS],
        "model_seeds": list(SEEDS),
        "model_controls": list(MODEL_CONTROL_VARIANTS),
        "one_minute_selection": {
            "selection_seed": SELECTION_SEED,
            "cohorts": ["normal", "meme"],
            "periods": PERIODS,
            "anchor_timeframes": list(ONE_MINUTE_TIMEFRAMES),
            "relative_volume_band": "frozen within-period upper band (+1)",
            "minimum_band_duration_hours": MINIMUM_HIGH_VOLUME_DURATION_HOURS,
            "minimum_next_hour_volume_ratio": VOLUME_REACTION_THRESHOLD,
            "fresh_contact_rule": (
                "Current 1h candle intersects the exact frozen zone and the preceding "
                "four complete 1h candles do not."
            ),
            "independence_hours_per_pair": INDEPENDENCE_HOURS,
            "diagnostic_sample": (
                "One deterministic episode for every cohort x chronological period x "
                "causal anchor timeframe cell, preferring a different pair in each cell."
            ),
            "future_signed_direction_used_for_selection": False,
        },
    }
    request_sha256 = g3m.sha256_text(
        json.dumps(request, sort_keys=True, default=g0.json_default)
    )
    one_minute_record_path = run_dir / "g9_one_minute_freeze_record.json"
    one_minute_record = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_one_minute_paths",
        "request_sha256": request_sha256,
        "selection_rule": request["one_minute_selection"],
        "episode_pool_rows": len(pool),
        "diagnostic_sample_rows": len(sample),
        "acquisition_intervals": len(intervals),
        "artifacts": {
            "episode_pool_csv": artifact(pool_path),
            "diagnostic_sample_csv": artifact(sample_path),
            "acquisition_intervals_csv": artifact(interval_path),
            "anchor_audit_csv": artifact(anchor_audit_path),
        },
        "directional_path_read": False,
    }
    g0.atomic_write_json(one_minute_record, one_minute_record_path)
    frozen = {
        "schema_version": 1,
        "objective": "objective_02b_market_reaction_zone_discovery",
        "generation": 9,
        "run_id": run_id,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_generation9_sibling_outcomes",
        "request": request,
        "request_sha256": request_sha256,
        "source_contracts": source_contracts(),
        "limited_three_source_family": {
            "questions": [asdict(question) for question in MODEL_QUESTIONS],
            "question_count": len(MODEL_QUESTIONS),
            "seeds": list(SEEDS),
            "profiles": profiles,
            "profile_count": len(profiles),
            "comparisons": comparisons,
            "comparison_count": len(comparisons),
            "controls_per_question_seed": EXPECTED_CONTROLS_PER_QUESTION,
            "pass_rule": (
                "For its declared market group, the complete three-source model must beat "
                "all three leave-one-out controls, all three single-input stale controls, "
                "and all three single-input nonself-shuffled controls in both validation "
                "periods and all three seeds."
            ),
        },
        "one_minute_family": {
            "freeze_record": artifact(one_minute_record_path),
            "selection_rule": request["one_minute_selection"],
            "pool_rows": len(pool),
            "sample_rows": len(sample),
            "sample_strata": sorted(sample["sample_stratum"].unique()),
            "sample_pairs": sorted(sample["pair"].unique()),
            "acquisition_intervals": len(intervals),
            "anchor_audit": audit_summary,
            "analysis_contract": {
                "activity_and_direction_kept_separate": True,
                "minute_horizons": [1, 3, 5, 10, 15, 30, 60, 120],
                "hour_horizons": [4, 8, 12, 24, 48],
                "path_outcomes": [
                    "signed return",
                    "maximum favourable and adverse excursion",
                    "first passage away from or through the zone",
                    "rejection versus breakout",
                    "dwell and recross count",
                ],
                "controls": [
                    "matched no-level timestamp",
                    "causal stale level",
                    "deterministically shifted level",
                    "majority direction comparator",
                    "simple pre-contact trend comparator",
                ],
                "freqai_on_twelve_episode_sample": False,
            },
        },
        "artifacts": {
            "episode_pool_csv": artifact(pool_path),
            "diagnostic_sample_csv": artifact(sample_path),
            "acquisition_intervals_csv": artifact(interval_path),
            "anchor_audit_csv": artifact(anchor_audit_path),
            "one_minute_freeze_record": artifact(one_minute_record_path),
        },
        "research_boundary": {
            "ordinary_timestamp_direction_search": False,
            "profit_used": False,
            "entry_exit_construction": False,
            "strategy_promotion": False,
            "one_minute_direction_only_inside_frozen_episode_queue": True,
        },
        "sequencing": (
            "Both Generation 9 sibling families were frozen together. Both must be "
            "completed or honestly parked before their joint review."
        ),
    }
    g0.atomic_write_json(frozen, FREEZE_PATH)
    return frozen


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze both Generation 9 sibling families before opening their model or "
            "one-minute path outcomes."
        )
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    args = parser.parse_args(argv)
    frozen = freeze_generation9(args.run_id)
    print(
        json.dumps(
            {
                "status": frozen["status"],
                "freeze_path": str(FREEZE_PATH),
                "model_questions": frozen["limited_three_source_family"][
                    "question_count"
                ],
                "model_profiles": frozen["limited_three_source_family"][
                    "profile_count"
                ],
                "model_comparisons": frozen["limited_three_source_family"][
                    "comparison_count"
                ],
                "one_minute_pool_rows": frozen["one_minute_family"]["pool_rows"],
                "one_minute_sample_rows": frozen["one_minute_family"]["sample_rows"],
                "one_minute_intervals": frozen["one_minute_family"][
                    "acquisition_intervals"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
