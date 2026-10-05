from __future__ import annotations

# Fix numerical-library thread counts before importing numpy/pandas/FreqAI helpers.
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
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features.market_reaction_zone_freqai_generation0 import (  # noqa: E501
    common_prediction_keys,
    eligibility_digest,
    load_predictions,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_freqai_generation1 import (  # noqa: E501
    run_profile as run_freqai_profile,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation0 import (
    OUTPUT_ROOT,
    atomic_write_json,
    atomic_write_parquet,
    load_manifest,
    normalize_dates,
    prepare_base_market_frame,
    sha256_file,
    utc_now,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation1_localization import (  # noqa: E501
    causal_market_context,
)
from user_data.strategies.MarketReactionZoneFreqAIGeneration4Strategy import (
    MAX_TARGET_HORIZON_HOURS,
    TARGET_COLUMNS,
)


USER_DATA_DIR = REPO_ROOT / "user_data"
DEFAULT_CONFIG = USER_DATA_DIR / "configs" / "config_market_reaction_zone_freqai.example.json"
DEFAULT_PYTHON = (
    REPO_ROOT / "runtime" / "venvs" / "freqtrade-backtest-08" / "Scripts" / "python.exe"
)
STRATEGY_PATH = USER_DATA_DIR / "strategies"
STRATEGY_FILE = STRATEGY_PATH / "MarketReactionZoneFreqAIGeneration4Strategy.py"
DATA_DIR = USER_DATA_DIR / "data" / "binance"
FROZEN_BATCH = OUTPUT_ROOT / "generation3_review" / "g4_frozen_branch_batch.json"
RECORD_ROOT = (
    OUTPUT_ROOT / "generation4_branches" / "g4d_freqai_reaction_ablation"
)
ARTIFACT_ROOT = (
    Path(r"D:\FreqTradeStuffLargeData")
    / "research_outputs"
    / "market_reaction_zones"
    / "generation4_branches"
    / "g4d_freqai_reaction_ablation"
)
G3A_RECORD_ROOT = OUTPUT_ROOT / "generation3_branches" / "g3a_thin_lvn_attribution"
G3A_ARTIFACT_ROOT = (
    Path(r"D:\FreqTradeStuffLargeData")
    / "research_outputs"
    / "market_reaction_zones"
    / "generation3_branches"
    / "g3a_thin_lvn_attribution"
)
G3D_RECORD_ROOT = OUTPUT_ROOT / "generation3_branches" / "g3d_density_arrival_states"
G3D_ARTIFACT_ROOT = (
    Path(r"D:\FreqTradeStuffLargeData")
    / "research_outputs"
    / "market_reaction_zones"
    / "generation3_branches"
    / "g3d_density_arrival_states"
)
NORMAL_MANIFEST = OUTPUT_ROOT / "generation0_manifest.json"
MEME_MANIFEST = OUTPUT_ROOT / "generation2_shared" / "g2_meme_reaction_manifest.json"

OUTPUT_SCHEMA_VERSION = 1
RUN_SCHEMA_VERSION = 1
MAX_WORKERS = 4
MIN_SCORABLE_ROWS = 20
MIN_RETAIN_COINS = 5
TARGET_SOURCE_COLUMNS = {
    TARGET_COLUMNS[0]: "contact_volume_ratio",
    TARGET_COLUMNS[1]: "abs_excursion_atr_h1",
    TARGET_COLUMNS[2]: "dwell_fraction_h4",
}
WIDE_SOURCE_COLUMNS = (
    "state_btc_return_1h",
    "state_btc_return_24h",
    "state_btc_atr_pct",
    "state_top10_breadth",
    "state_top10_mean_abs_return",
    "state_top10_return_dispersion",
)
APPROACH_STATES = (
    "from_above",
    "from_below",
    "already_inside_or_unclear",
    "unknown",
)
ARRIVAL_STATES = (
    "first_arrival_after_outside_interval",
    "near_miss",
    "already_inside",
    "repeat_contact",
)

PROFILES: dict[str, dict[str, Any]] = {
    "unconditional": {
        "frozen_profile": "unconditional_or_matched_state_baseline",
        "strategy": "MarketReactionZoneG4DUnconditionalFreqAIResearchStrategy",
        "role": (
            "Rolling per-pair time-only reaction baseline with no price, volume, level, "
            "or outcome inputs. The causal time coordinate prevents FreqAI's variance "
            "filter from rejecting a literal constant."
        ),
        "baseline": None,
    },
    "market_state": {
        "frozen_profile": "ohlcv_and_broad_market_state_only",
        "strategy": "MarketReactionZoneG4DMarketStateFreqAIResearchStrategy",
        "role": "Prior-candle local OHLCV/indicators plus causal BTC and cohort state.",
        "baseline": "unconditional",
    },
    "level_only": {
        "frozen_profile": "candidate_level_identity_and_attributes_only",
        "strategy": "MarketReactionZoneG4DLevelOnlyFreqAIResearchStrategy",
        "role": "Current candidate-level identity and causal attributes without market state.",
        "baseline": "unconditional",
    },
    "combined_current": {
        "frozen_profile": "ohlcv_state_plus_current_level_and_geometry",
        "strategy": "MarketReactionZoneG4DCombinedCurrentFreqAIResearchStrategy",
        "role": "Market state plus the current level and local geometry.",
        "baseline": "market_state",
    },
    "combined_shuffled": {
        "frozen_profile": "ohlcv_state_plus_shifted_shuffled_or_stale_level",
        "strategy": "MarketReactionZoneG4DCombinedShuffledFreqAIResearchStrategy",
        "role": "Market state plus the same level/geometry columns shuffled within pair-period.",
        "baseline": "market_state",
    },
    "selected_mtf": {
        "frozen_profile": "selected_multi_timeframe_geometry",
        "strategy": "MarketReactionZoneG4DSelectedMTFFreqAIResearchStrategy",
        "role": "Combined current profile plus predeclared multi-timeframe geometry counts.",
        "baseline": "combined_current",
    },
    "limited_interactions": {
        "frozen_profile": "combined_with_only_predeclared_low_order_interactions",
        "strategy": "MarketReactionZoneG4DLimitedInteractionsFreqAIResearchStrategy",
        "role": "Selected-MTF profile plus two predeclared low-order interactions.",
        "baseline": "selected_mtf",
    },
}

SURFACES: dict[str, dict[str, Any]] = {
    "g3a_normal_thin_lvn_cluster": {
        "source_family": "g3a",
        "cohort": "normal",
        "source_run_id": "g3a_lvn_cluster_attribution_normal10_full_20260814b",
        "manifest": NORMAL_MANIFEST,
        "timerange": "20240101-20260401",
        "train_days": 730,
        "backtest_days": 90,
        "validation_periods": ("validation_early", "validation_late"),
    },
    "g3a_meme_thin_lvn_cluster": {
        "source_family": "g3a",
        "cohort": "meme",
        "source_run_id": "g3a_lvn_cluster_attribution_meme_full_20260814b",
        "manifest": MEME_MANIFEST,
        "timerange": "20260101-20260714",
        "train_days": 160,
        "backtest_days": 30,
        "validation_periods": ("meme_validation_early", "meme_validation_late"),
    },
    "g3d_normal_isolated_density": {
        "source_family": "g3d",
        "cohort": "normal",
        "source_run_id": "g3d_arrival_states_normal10_full_20260814a",
        "manifest": NORMAL_MANIFEST,
        "actual_scope": "single_density_zone",
        "timerange": "20240101-20260401",
        "train_days": 730,
        "backtest_days": 90,
        "validation_periods": ("validation_early", "validation_late"),
    },
    "g3d_normal_density_cluster": {
        "source_family": "g3d",
        "cohort": "normal",
        "source_run_id": "g3d_arrival_states_normal10_full_20260814a",
        "manifest": NORMAL_MANIFEST,
        "actual_scope": "density_cluster",
        "timerange": "20240101-20260401",
        "train_days": 730,
        "backtest_days": 90,
        "validation_periods": ("validation_early", "validation_late"),
    },
    "g3d_meme_density_cluster": {
        "source_family": "g3d",
        "cohort": "meme",
        "source_run_id": "g3d_arrival_states_meme_full_20260814a",
        "manifest": MEME_MANIFEST,
        "actual_scope": "density_cluster",
        "timerange": "20260101-20260714",
        "train_days": 160,
        "backtest_days": 30,
        "validation_periods": ("meme_validation_early", "meme_validation_late"),
    },
}


def parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def pair_stem(pair: str) -> str:
    return pair.strip().upper().replace("/", "_").replace(":", "_")


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def period_contract(manifest: dict[str, Any]) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    return {
        str(item["id"]): (
            pd.Timestamp(item["start_utc"]),
            pd.Timestamp(item["end_utc_exclusive"]),
        )
        for item in manifest["data"]["chronological_periods"]
        if str(item.get("role")) != "diagnostic_only_not_confirmation"
    }


def deterministic_shift(size: int, key: str) -> int:
    if size < 2:
        raise ValueError(f"A shuffled control needs at least two rows: {key}")
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return 1 + int.from_bytes(digest[:8], "big") % (size - 1)


def independently_spaced_events(events: DataFrame, hours: int = 4) -> DataFrame:
    """Keep a deterministic chronological event mask whose future paths do not overlap."""
    if events.empty:
        return events.copy()
    ordered = events.sort_values(
        ["date", "event_state", "level_identity"], kind="mergesort"
    ).reset_index(drop=True)
    retained: list[int] = []
    last: pd.Timestamp | None = None
    minimum = pd.Timedelta(hours=hours)
    for index, date in ordered["date"].items():
        timestamp = pd.Timestamp(date)
        if last is None or timestamp - last > minimum:
            retained.append(index)
            last = timestamp
    return ordered.iloc[retained].reset_index(drop=True)


def one_hot(value: Any, allowed: Sequence[str], prefix: str) -> dict[str, float]:
    normalized = str(value) if pd.notna(value) else "unknown"
    if normalized not in allowed:
        normalized = "unknown" if "unknown" in allowed else normalized
    return {f"{prefix}_{item}": float(normalized == item) for item in allowed}


def source_run_record(surface: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    run_id = str(surface["source_run_id"])
    if surface["source_family"] == "g3a":
        path = G3A_RECORD_ROOT / run_id / "g3a_run_record.json"
    else:
        path = G3D_RECORD_ROOT / run_id / "g3d_state_run_record.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    record = json.loads(path.read_text(encoding="utf-8"))
    if record.get("status") != "completed":
        raise ValueError(f"Source run is not complete: {path}")
    return path, record


def g3a_event_path_by_pair(record: dict[str, Any]) -> dict[str, Path]:
    contracts = record["request_contract"]["source_contracts"]
    output: dict[str, Path] = {}
    for item in contracts:
        if item.get("timeframe") != "1h" or not item.get("event_path"):
            continue
        output[str(item["pair"])] = Path(str(item["event_path"]))
    return output


def g3a_candidate_events(
    pair: str,
    *,
    surface: dict[str, Any],
    source_record: dict[str, Any],
) -> DataFrame:
    run_id = str(surface["source_run_id"])
    match_path = G3A_ARTIFACT_ROOT / run_id / "pair_event_matches" / f"{pair_stem(pair)}.parquet"
    if not match_path.is_file():
        raise FileNotFoundError(match_path)
    matches = pd.read_parquet(match_path)
    matches = matches.loc[
        matches["question_id"].eq("g3a_lvn_thinness_after_surviving_cluster")
        & matches["attribute_assignment"].eq("actual")
        & matches["source_timeframe"].eq("1h")
        & matches["zone_method"].eq("wide_base_atr")
    ].copy()
    if matches.empty:
        raise ValueError(f"No corrected actual G3A matches for {pair}.")

    rows: list[dict[str, Any]] = []
    for side in ("high", "low"):
        for row in matches.to_dict("records"):
            event: dict[str, Any] = {
                "pair": pair,
                "date": row[f"{side}_event_time"],
                "base_index": int(row[f"{side}_base_index"]),
                "period": str(row["period"]),
                "event_state": "thin_lvn_mixed_cluster_contact",
                "level_identity": str(row["level_name"]),
                "approach_state": str(row["approach_state"]),
                "level__pre_distance_atr": row[f"{side}_pre_distance_atr"],
                "level__thinness": row[f"{side}_attribute_value"],
                "level__vp_value_area_width_pct": row[
                    f"{side}_state__state_vp_value_area_width_pct"
                ],
                "level__vp_persistence_bars": row[
                    f"{side}_state__state_vp_level_persistence_bars"
                ],
                "geometry__peer_level_count": row[
                    f"{side}_state__state_peer_level_count"
                ],
                "geometry__peer_contact_count": row[
                    f"{side}_state__state_peer_contact_count"
                ],
                "geometry__peer_dependency_group_count": row[
                    f"{side}_state__state_peer_dependency_group_count"
                ],
                "geometry__peer_contact_connected_dependency_group_count": row[
                    f"{side}_state__state_peer_contact_connected_dependency_group_count"
                ],
                "mtf__peer_higher_tf_count": row[
                    f"{side}_state__state_peer_higher_tf_count"
                ],
                "mtf__peer_daily_count": row[f"{side}_state__state_peer_daily_count"],
            }
            event.update(one_hot(row["level_name"], ("lvn_above", "lvn_below"), "level__id"))
            event.update(one_hot(row["approach_state"], APPROACH_STATES, "level__approach"))
            rows.append(event)
    events = DataFrame(rows)
    key = ["pair", "date", "base_index", "level_identity"]
    value_columns = [column for column in events if column not in key]
    conflicts = events.groupby(key, dropna=False)[value_columns].nunique(dropna=False).max(axis=1)
    if conflicts.gt(1).any():
        raise ValueError(f"Conflicting duplicate G3A event attributes for {pair}.")
    events = events.drop_duplicates(key, keep="first")

    raw_path = g3a_event_path_by_pair(source_record).get(pair)
    if raw_path is None or not raw_path.is_file():
        raise FileNotFoundError(raw_path or f"G3A raw event path for {pair}")
    raw_columns = [
        "base_index",
        "event_time",
        "period",
        "level_name",
        "source_available_at",
        *TARGET_SOURCE_COLUMNS.values(),
    ]
    raw = pd.read_parquet(
        raw_path,
        filters=[
            ("control", "==", "actual"),
            ("level_name", "in", ["lvn_above", "lvn_below"]),
            ("zone_method", "==", "wide_base_atr"),
            ("source_timeframe", "==", "1h"),
        ],
        columns=raw_columns,
    )
    raw = raw.rename(columns={"event_time": "date", "level_name": "level_identity"})
    raw["date"] = normalize_dates(raw["date"])
    events["date"] = normalize_dates(events["date"])
    joined = events.merge(
        raw,
        on=["base_index", "date", "period", "level_identity"],
        how="left",
        validate="one_to_one",
    )
    for target, source in TARGET_SOURCE_COLUMNS.items():
        joined[target] = pd.to_numeric(joined[source], errors="coerce")
    joined["source_available_at"] = normalize_dates(joined["source_available_at"])
    joined["interaction__thinness_x_contact_dependency_groups"] = (
        pd.to_numeric(joined["level__thinness"], errors="coerce")
        * pd.to_numeric(
            joined["geometry__peer_contact_connected_dependency_group_count"],
            errors="coerce",
        )
    )
    joined["interaction__thinness_x_higher_tf_count"] = (
        pd.to_numeric(joined["level__thinness"], errors="coerce")
        * pd.to_numeric(joined["mtf__peer_higher_tf_count"], errors="coerce")
    )
    return joined.drop(columns=list(TARGET_SOURCE_COLUMNS.values()))


def g3d_candidate_events(pair: str, *, surface: dict[str, Any]) -> DataFrame:
    run_id = str(surface["source_run_id"])
    source_path = G3D_ARTIFACT_ROOT / run_id / "g3d_independent_pairs.parquet"
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    frame = pd.read_parquet(
        source_path,
        filters=[
            ("pair", "==", pair),
            ("density_family", "==", "confirmed_swing_price_density"),
            ("history_hours", "==", 168),
            ("actual_scope", "==", str(surface["actual_scope"])),
            ("response_window", "in", ["h1", "h4"]),
        ],
    )
    allowed_controls = {
        "event_state__near_miss",
        "event_state__already_inside",
        "event_state__repeat_contact",
    }
    frame = frame.loc[frame["control"].isin(allowed_controls)].copy()
    h1 = frame.loc[frame["response_window"].eq("h1")].copy()
    frame = frame.loc[frame["response_window"].eq("h4")].copy()
    if frame.empty or h1.empty:
        raise ValueError(f"No fixed G3D arrival rows for {pair} and {surface['actual_scope']}.")

    one_hour_excursion: dict[tuple[str, int, pd.Timestamp, str], float] = {}
    for side in ("actual", "control"):
        for row in h1.to_dict("records"):
            key = (
                side,
                int(row[f"{side}_base_index"]),
                pd.Timestamp(row[f"{side}_event_time"]),
                str(row[f"{side}_event_state"]),
            )
            value = float(row[f"{side}__abs_excursion_atr_h1"])
            previous = one_hour_excursion.get(key)
            if previous is not None and not np.isclose(previous, value, rtol=1e-12, atol=1e-12):
                raise ValueError(f"Conflicting one-hour G3D target for {pair} and {key}.")
            one_hour_excursion[key] = value

    rows: list[dict[str, Any]] = []
    for side in ("actual", "control"):
        for row in frame.to_dict("records"):
            state = str(row[f"{side}_event_state"])
            event: dict[str, Any] = {
                "pair": pair,
                "date": row[f"{side}_event_time"],
                "base_index": int(row[f"{side}_base_index"]),
                "period": str(row["period"]),
                "event_state": state,
                "level_identity": "confirmed_swing_price_density",
                "approach_state": "not_available_as_an_event_specific_g3d_field",
                "source_available_at": row[f"{side}_source_available_at"],
                "level__pre_distance_atr": row[f"{side}_raw_pre_distance_atr"],
                "level__zone_support_fraction": row[f"{side}_zone_support_fraction"],
                "level__zone_half_width_atr": row[
                    f"{side}_state__state_g3d_zone_half_width_atr"
                ],
                "geometry__other_density_zone_count": row[
                    f"{side}_state__state_g3d_other_density_zone_count"
                ],
                "mtf__reference_level_count": row[
                    f"{side}_state__state_g3d_reference_level_count"
                ],
                "mtf__overlap_reference_level_count": row[
                    f"{side}_overlap_reference_level_count"
                ],
            }
            event.update(one_hot(state, ARRIVAL_STATES, "level__arrival"))
            event.update(
                one_hot(
                    surface["actual_scope"],
                    ("single_density_zone", "density_cluster"),
                    "level__scope",
                )
            )
            for target, source in TARGET_SOURCE_COLUMNS.items():
                if source == "abs_excursion_atr_h1":
                    key = (
                        side,
                        int(row[f"{side}_base_index"]),
                        pd.Timestamp(row[f"{side}_event_time"]),
                        state,
                    )
                    event[target] = one_hour_excursion.get(key, np.nan)
                else:
                    event[target] = row[f"{side}__{source}"]
            rows.append(event)
    events = DataFrame(rows)
    events["date"] = normalize_dates(events["date"])
    events["source_available_at"] = normalize_dates(events["source_available_at"])
    key = ["pair", "date", "base_index", "event_state", "level_identity"]
    value_columns = [column for column in events if column not in key]
    conflicts = events.groupby(key, dropna=False)[value_columns].nunique(dropna=False).max(axis=1)
    if conflicts.gt(1).any():
        raise ValueError(f"Conflicting duplicate G3D event attributes for {pair}.")
    events = events.drop_duplicates(key, keep="first")
    fresh = pd.to_numeric(
        events["level__arrival_first_arrival_after_outside_interval"], errors="coerce"
    )
    support = pd.to_numeric(events["level__zone_support_fraction"], errors="coerce")
    reference_count = pd.to_numeric(events["mtf__reference_level_count"], errors="coerce")
    events["interaction__fresh_arrival_x_support_fraction"] = fresh * support
    events["interaction__support_fraction_x_reference_count"] = support * reference_count
    return events


def validate_frozen_branch() -> dict[str, Any]:
    if not FROZEN_BATCH.is_file():
        raise FileNotFoundError(FROZEN_BATCH)
    frozen = json.loads(FROZEN_BATCH.read_text(encoding="utf-8"))
    matches = [
        branch
        for branch in frozen.get("branches", [])
        if branch.get("id") == "g4d_freqai_reaction_ablation"
    ]
    if len(matches) != 1:
        raise ValueError("The frozen Generation 4 batch lacks exactly one G4D branch.")
    branch = matches[0]
    expected_targets = {
        "contact_volume_ratio",
        "one_hour_absolute_excursion_in_prior_atr",
        "four_hour_dwell_fraction",
    }
    if set(branch.get("fixed_targets", [])) != expected_targets:
        raise ValueError("The frozen G4D targets changed.")
    expected_profiles = {item["frozen_profile"] for item in PROFILES.values()}
    if set(branch.get("fixed_profiles", [])) != expected_profiles:
        raise ValueError("The implemented G4D ladder does not match the frozen profiles.")
    expected_surfaces = {
        "corrected_g3a_thin_lvn_mixed_cluster_pairs",
        "g3d_supported_fresh_arrival_density_scopes",
    }
    if set(branch.get("fixed_source_surfaces", [])) != expected_surfaces:
        raise ValueError("The frozen G4D source surfaces changed.")
    if int(branch.get("iteration_cap", -1)) != 4:
        raise ValueError("The frozen G4D iteration cap changed.")
    return branch


def select_pairs(manifest: dict[str, Any], requested: str) -> tuple[str, ...]:
    allowed = tuple(str(pair) for pair in manifest["data"]["pairs"])
    if requested.strip().lower() == "all":
        return allowed
    selected = parse_csv(requested)
    unknown = sorted(set(selected).difference(allowed))
    if not selected or unknown:
        raise ValueError(f"Invalid pair selection: selected={selected}, unknown={unknown}")
    return selected


def event_feature_columns(events: DataFrame) -> tuple[str, ...]:
    return tuple(
        column
        for column in events.columns
        if column.startswith(("level__", "geometry__", "mtf__", "interaction__"))
    )


def current_placebo_columns(events: DataFrame) -> tuple[str, ...]:
    return tuple(
        column for column in events if column.startswith(("level__", "geometry__"))
    )


def attach_shuffled_control(
    events: DataFrame,
    *,
    pair: str,
    surface_id: str,
) -> tuple[DataFrame, list[dict[str, Any]]]:
    output = events.copy()
    current_columns = current_placebo_columns(output)
    if not current_columns:
        raise ValueError("The G4D current level/geometry block is empty.")
    placebo_columns = [
        f"placebo__{column.replace('__', '_', 1)}" for column in current_columns
    ]
    for column in placebo_columns:
        output[column] = np.nan
    audit: list[dict[str, Any]] = []
    for period, index in output.groupby("period", sort=False).groups.items():
        positions = np.asarray(list(index), dtype=np.int64)
        size = len(positions)
        shift = deterministic_shift(size, f"{surface_id}|{pair}|{period}")
        source_positions = np.roll(positions, shift)
        current = output.loc[source_positions, current_columns].to_numpy(dtype=float)
        output.loc[positions, placebo_columns] = current
        source_dates = output.loc[source_positions, "date"].to_numpy()
        destination_dates = output.loc[positions, "date"].to_numpy()
        self_assignments = int(np.sum(source_dates == destination_dates))
        if self_assignments:
            raise AssertionError(f"Shuffled control kept {self_assignments} self rows.")
        audit.append(
            {
                "pair": pair,
                "period": str(period),
                "rows": size,
                "circular_shift": shift,
                "self_assignments": self_assignments,
                "source_date_digest": stable_json_sha256([str(value) for value in source_dates]),
            }
        )
    return output, audit


def causal_wide_frame(manifest: dict[str, Any]) -> DataFrame:
    wide = causal_market_context(manifest).copy()
    wide["date"] = normalize_dates(wide["date"])
    missing = sorted(set(WIDE_SOURCE_COLUMNS).difference(wide.columns))
    if missing:
        raise ValueError(f"Causal market context lacks G4D columns: {missing}")
    wide = wide[["date", *WIDE_SOURCE_COLUMNS]].rename(
        columns={column: f"wide__{column.removeprefix('state_')}" for column in WIDE_SOURCE_COLUMNS}
    )
    if wide["date"].duplicated().any():
        raise ValueError("Causal wide-market context has duplicate dates.")
    return wide


def apply_event_eligibility(
    events: DataFrame,
    *,
    manifest: dict[str, Any],
    wide: DataFrame,
    pair: str,
) -> tuple[DataFrame, dict[str, Any]]:
    periods = period_contract(manifest)
    output = events.copy()
    output["date"] = normalize_dates(output["date"])
    output["source_available_at"] = normalize_dates(output["source_available_at"])
    raw_rows = len(output)
    source_missing = int(output["source_available_at"].isna().sum())
    future_sources = int(
        (
            output["source_available_at"].notna()
            & output["source_available_at"].gt(output["date"])
        ).sum()
    )
    if source_missing or future_sources:
        raise ValueError(
            f"Causal source boundary failed for {pair}: missing={source_missing}, "
            f"future={future_sources}."
        )

    output = output.loc[output["period"].isin(periods)].copy()
    period_rows = len(output)
    inside = pd.Series(False, index=output.index)
    for period, (start, end) in periods.items():
        selected = output["period"].eq(period)
        inside.loc[selected] = output.loc[selected, "date"].ge(start) & output.loc[
            selected, "date"
        ].lt(end - pd.Timedelta(hours=MAX_TARGET_HORIZON_HOURS))
    period_boundary_drops = int((~inside).sum())
    output = output.loc[inside].copy()

    wide_columns = [column for column in wide if column.startswith("wide__")]
    output = output.merge(wide, on="date", how="left", validate="many_to_one")
    feature_columns = list(event_feature_columns(output))
    numeric_columns = [*feature_columns, *wide_columns, *TARGET_COLUMNS]
    for column in numeric_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    missing_by_column = {
        column: int(output[column].isna().sum())
        for column in numeric_columns
        if output[column].isna().any()
    }
    complete = output[numeric_columns].notna().all(axis=1)
    incomplete_drops = int((~complete).sum())
    output = output.loc[complete].copy()
    duplicate_dates_before = int(output.duplicated("date").sum())
    if duplicate_dates_before:
        key = ["date", "event_state", "level_identity"]
        output = output.sort_values(key, kind="mergesort").drop_duplicates("date", keep="first")
    before_independence = len(output)
    output = independently_spaced_events(output, MAX_TARGET_HORIZON_HOURS)
    output, shuffle_audit = attach_shuffled_control(
        output,
        pair=pair,
        surface_id=str(events.attrs.get("surface_id", "g4d")),
    )
    all_features = [
        column
        for column in output
        if column.startswith(
            ("wide__", "level__", "geometry__", "mtf__", "interaction__", "placebo__")
        )
    ]
    if output[all_features].isna().any().any():
        raise ValueError(f"Eligible G4D event rows contain missing feature values for {pair}.")
    current_count = len(current_placebo_columns(output))
    placebo_count = sum(column.startswith("placebo__") for column in output)
    if current_count != placebo_count:
        raise ValueError(
            f"Current/placebo feature counts differ for {pair}: {current_count} != {placebo_count}."
        )
    audit = {
        "pair": pair,
        "raw_candidate_rows": raw_rows,
        "known_period_rows": period_rows,
        "period_boundary_drops": period_boundary_drops,
        "incomplete_rows_dropped": incomplete_drops,
        "missing_by_column_before_complete_case": missing_by_column,
        "duplicate_dates_resolved_outcome_blind": duplicate_dates_before,
        "complete_rows_before_independence": before_independence,
        "retained_independent_events": len(output),
        "minimum_event_gap_hours": (
            float(output["date"].sort_values().diff().dropna().dt.total_seconds().min() / 3600.0)
            if len(output) > 1
            else None
        ),
        "current_level_geometry_features": current_count,
        "shuffled_level_geometry_features": placebo_count,
        "shuffle": shuffle_audit,
    }
    return output.reset_index(drop=True), audit


def prediction_training_windows(
    events: DataFrame,
    *,
    surface: dict[str, Any],
    pair: str,
) -> list[dict[str, Any]]:
    manifest = load_manifest(Path(surface["manifest"]))
    periods = period_contract(manifest)
    train = pd.Timedelta(days=int(surface["train_days"]))
    step = pd.Timedelta(days=int(surface["backtest_days"]))
    rows: list[dict[str, Any]] = []
    for period in surface["validation_periods"]:
        start, end = periods[str(period)]
        cursor = start
        segment = 0
        while cursor < end:
            prediction_end = min(cursor + step, end)
            train_start = cursor - train
            training_rows = int(events["date"].ge(train_start).mul(events["date"].lt(cursor)).sum())
            prediction_rows = int(
                events["date"].ge(cursor).mul(events["date"].lt(prediction_end)).sum()
            )
            rows.append(
                {
                    "pair": pair,
                    "period": str(period),
                    "segment": segment,
                    "train_start": train_start,
                    "prediction_start": cursor,
                    "prediction_end": prediction_end,
                    "training_event_rows": training_rows,
                    "prediction_event_rows": prediction_rows,
                    "training_support_at_least_20": training_rows >= MIN_SCORABLE_ROWS,
                }
            )
            cursor = prediction_end
            segment += 1
    return rows


def build_pair_caches(
    pair: str,
    *,
    surface_id: str,
    surface: dict[str, Any],
    manifest: dict[str, Any],
    source_record: dict[str, Any],
    wide: DataFrame,
    feature_dir: Path,
    event_dir: Path,
    overwrite: bool,
) -> dict[str, Any]:
    feature_path = feature_dir / f"{pair_stem(pair)}.parquet"
    event_path = event_dir / f"{pair_stem(pair)}.parquet"
    if feature_path.exists() or event_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"G4D cache already exists; use a new run id or --overwrite: {feature_path}"
            )
    if surface["source_family"] == "g3a":
        candidates = g3a_candidate_events(
            pair,
            surface=surface,
            source_record=source_record,
        )
    else:
        candidates = g3d_candidate_events(pair, surface=surface)
    candidates.attrs["surface_id"] = surface_id
    events, audit = apply_event_eligibility(
        candidates,
        manifest=manifest,
        wide=wide,
        pair=pair,
    )
    if events.empty:
        raise ValueError(f"No eligible G4D events remain for {pair}.")

    base = prepare_base_market_frame(pair, manifest)[["date"]].copy()
    base["date"] = normalize_dates(base["date"])
    feature_columns = [
        column
        for column in events
        if column.startswith(("level__", "geometry__", "mtf__", "interaction__", "placebo__"))
    ]
    feature_cache = base.merge(wide, on="date", how="left", validate="one_to_one").merge(
        events[["date", *feature_columns]],
        on="date",
        how="left",
        validate="one_to_one",
    )
    event_metadata = [
        "date",
        "period",
        "event_state",
        "level_identity",
        "approach_state",
        "source_available_at",
        *TARGET_COLUMNS,
    ]
    event_cache = events[event_metadata].copy()
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(feature_cache, feature_path)
    atomic_write_parquet(event_cache, event_path)

    period_counts = (
        event_cache.groupby(["period", "event_state"], dropna=False)
        .size()
        .reset_index(name="rows")
        .to_dict("records")
    )
    training_windows = prediction_training_windows(events, surface=surface, pair=pair)
    key_frame = events[["pair", "date"]].copy()
    audit.update(
        {
            "surface_id": surface_id,
            "feature_path": str(feature_path),
            "feature_sha256": sha256_file(feature_path),
            "event_path": str(event_path),
            "event_sha256": sha256_file(event_path),
            "event_key_digest": eligibility_digest(key_frame),
            "event_period_state_counts": period_counts,
            "training_windows": training_windows,
            "feature_columns": feature_columns,
            "wide_feature_columns": [column for column in wide if column.startswith("wide__")],
        }
    )
    return audit


def source_contract_audit(
    *,
    surface: dict[str, Any],
    pairs: Sequence[str],
    source_record_path: Path,
    source_record: dict[str, Any],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if surface["source_family"] == "g3a":
        raw_paths = g3a_event_path_by_pair(source_record)
        contracts = {
            (str(item["pair"]), str(item["timeframe"])): item
            for item in source_record["request_contract"]["source_contracts"]
        }
        for pair in pairs:
            match_path = (
                G3A_ARTIFACT_ROOT
                / str(surface["source_run_id"])
                / "pair_event_matches"
                / f"{pair_stem(pair)}.parquet"
            )
            raw_path = raw_paths.get(pair)
            if raw_path is None or not raw_path.is_file() or not match_path.is_file():
                raise FileNotFoundError(raw_path or match_path)
            contract = contracts[(pair, "1h")]
            stat = raw_path.stat()
            if int(contract["event_bytes"]) != stat.st_size:
                raise ValueError(f"G3A raw event byte size changed: {raw_path}")
            if int(contract["event_modified_ns"]) != stat.st_mtime_ns:
                raise ValueError(f"G3A raw event timestamp changed: {raw_path}")
            rows.append(
                {
                    "pair": pair,
                    "pair_match_path": str(match_path),
                    "pair_match_sha256": sha256_file(match_path),
                    "raw_event_path": str(raw_path),
                    "raw_event_bytes": stat.st_size,
                    "raw_event_modified_ns": stat.st_mtime_ns,
                    "raw_event_metadata_sha256": contract["event_metadata_sha256"],
                }
            )
    else:
        source_path = (
            G3D_ARTIFACT_ROOT
            / str(surface["source_run_id"])
            / "g3d_independent_pairs.parquet"
        )
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        rows.append(
            {
                "combined_source_path": str(source_path),
                "combined_source_sha256": sha256_file(source_path),
                "pairs": list(pairs),
            }
        )
    return {
        "source_run_record": str(source_record_path),
        "source_run_record_sha256": sha256_file(source_record_path),
        "source_rows": rows,
    }


def profile_config(
    base: dict[str, Any],
    *,
    identifier: str,
    pairs: Sequence[str],
    feature_dir: Path,
    event_dir: Path,
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
    freqai["multitarget_parallel_training"] = False
    freqai["feature_parameters"]["plot_feature_importances"] = 0
    freqai["feature_parameters"]["include_corr_pairlist"] = []
    freqai["feature_parameters"]["include_timeframes"] = ["1h"]
    freqai["feature_parameters"]["include_shifted_candles"] = 0
    freqai["feature_parameters"]["label_period_candles"] = MAX_TARGET_HORIZON_HOURS
    freqai["data_split_parameters"] = {"test_size": 0, "shuffle": False}
    training = freqai["model_training_parameters"]
    training["n_jobs"] = 1
    training["min_child_samples"] = 10
    if technical_smoke:
        training["n_estimators"] = min(20, int(training.get("n_estimators", 100)))
    config["market_reaction_zone_g4d"] = {
        "feature_cache_dir": str(feature_dir.resolve()),
        "event_cache_dir": str(event_dir.resolve()),
        "maximum_target_horizon_hours": MAX_TARGET_HORIZON_HOURS,
        "train_prediction_embargo_hours": MAX_TARGET_HORIZON_HOURS,
        "missing_source_policy": "retain_nan_and_exclude_from_common_event_mask",
    }
    return config


def runtime_snapshot() -> dict[str, Any]:
    snapshot: dict[str, Any] = {
        "logical_processors": os.cpu_count(),
        "visible_project_jobs": [],
        "load_note": "Exact process commands are inspected before increasing profile workers.",
    }
    try:
        import psutil

        memory = psutil.virtual_memory()
        snapshot.update(
            {
                "cpu_percent_sample": psutil.cpu_percent(interval=0.25),
                "memory_available_gib": round(memory.available / (1024**3), 3),
            }
        )
        for process in psutil.process_iter(["pid", "name", "cmdline", "num_threads"]):
            try:
                command = " ".join(process.info.get("cmdline") or [])
                lowered = command.lower()
                if any(token in lowered for token in ("freqtrade", "hyperopt", "sieve")):
                    snapshot["visible_project_jobs"].append(
                        {
                            "pid": process.info["pid"],
                            "name": process.info.get("name"),
                            "num_threads": process.info.get("num_threads"),
                            "command": command[:1000],
                        }
                    )
            except (psutil.AccessDenied, psutil.NoSuchProcess, TypeError):
                continue
    except ImportError:
        snapshot["load_note"] = "psutil unavailable; inspect live load outside this preflight."
    return snapshot


def build_manifest(
    *,
    run_id: str,
    surface_id: str,
    surface: dict[str, Any],
    record_dir: Path,
    artifact_dir: Path,
    base_config: Path,
    python_exe: Path,
    profiles: Sequence[str],
    pairs: Sequence[str],
    profile_workers: int,
    technical_smoke: bool,
    cache_inventory: Sequence[dict[str, Any]],
    source_audit: dict[str, Any],
    frozen_branch: dict[str, Any],
) -> dict[str, Any]:
    base = json.loads(base_config.read_text(encoding="utf-8"))
    feature_dir = artifact_dir / "feature_cache"
    event_dir = artifact_dir / "event_cache"
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
                feature_dir=feature_dir,
                event_dir=event_dir,
                train_days=int(surface["train_days"]),
                backtest_days=int(surface["backtest_days"]),
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
            str(surface["timerange"]),
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
                "frozen_profile": definition["frozen_profile"],
                "strategy": definition["strategy"],
                "role": definition["role"],
                "baseline": definition["baseline"],
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
    cache_contract = [
        {
            "pair": item["pair"],
            "feature_path": item["feature_path"],
            "feature_sha256": item["feature_sha256"],
            "event_path": item["event_path"],
            "event_sha256": item["event_sha256"],
            "event_key_digest": item["event_key_digest"],
            "retained_independent_events": item["retained_independent_events"],
        }
        for item in cache_inventory
    ]
    return {
        "schema_version": RUN_SCHEMA_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "prepared",
        "created_at_utc": utc_now(),
        "technical_smoke_not_evidence": bool(technical_smoke),
        "frozen_branch_id": "g4d_freqai_reaction_ablation",
        "surface_id": surface_id,
        "source_family": surface["source_family"],
        "cohort": surface["cohort"],
        "actual_scope": surface.get("actual_scope"),
        "objective": frozen_branch["hypothesis"],
        "strongest_alternative": frozen_branch["strongest_alternative"],
        "retain_interpretation": frozen_branch["retain_interpretation"],
        "park_interpretation": frozen_branch["park_interpretation"],
        "iteration_cap": int(frozen_branch["iteration_cap"]),
        "direction_prediction": False,
        "profit_optimization": False,
        "timerange": surface["timerange"],
        "development_role": "rolling training only; no in-sample score is used as evidence",
        "validation_periods": list(surface["validation_periods"]),
        "train_period_days": int(surface["train_days"]),
        "backtest_period_days": int(surface["backtest_days"]),
        "profile_workers": int(profile_workers),
        "model_threads_per_profile": 1,
        "model_class": "LightGBMRegressorMultiTarget",
        "pairs": list(pairs),
        "profiles": list(profiles),
        "targets": list(TARGET_COLUMNS),
        "target_meanings": {
            TARGET_COLUMNS[0]: "Contact-candle volume divided by its prior typical volume.",
            TARGET_COLUMNS[1]: (
                "Largest absolute one-hour move around the level, measured in the ATR "
                "known before contact. It is magnitude, not direction."
            ),
            TARGET_COLUMNS[2]: (
                "Fraction of the next four hourly observations that remain inside the "
                "tested level or zone."
            ),
        },
        "profile_comparisons": {
            "market_state": "unconditional",
            "level_only": "unconditional",
            "combined_current": "market_state",
            "combined_shuffled": "market_state",
            "selected_mtf": "combined_current",
            "limited_interactions": "selected_mtf",
            "current_versus_placebo": "combined_shuffled",
        },
        "retention_gate": {
            "unit": "surface and target",
            "required_validation_periods": list(surface["validation_periods"]),
            "aggregate_mae_improvement_over_market_state": "positive in both periods",
            "aggregate_spearman_change": "not negative in both periods",
            "coins_with_positive_mae_improvement": MIN_RETAIN_COINS,
            "current_mae_better_than_shuffled_control": "both periods",
            "calibration": "reported, not used as an arbitrary promotion cutoff",
            "meaning": "research lead only; never a strategy or trading promotion",
        },
        "event_surface": {
            "source_run_id": surface["source_run_id"],
            "maximum_target_horizon_hours": MAX_TARGET_HORIZON_HOURS,
            "minimum_gap_between_retained_events_hours": (
                f"strictly greater than {MAX_TARGET_HORIZON_HOURS}"
            ),
            "same_event_rows_for_every_profile": True,
            "duplicate_timestamp_resolution": (
                "stable chronological/state/identity order before any target values are scored"
            ),
            "shuffled_control": (
                "deterministic non-self circular assignment within each pair and period"
            ),
        },
        "leakage_controls": {
            "contact_candle_direct_ohlcv_used_as_feature": False,
            "contact_candle_event_classification_used": True,
            "contact_candle_timing_interpretation": (
                "Arrival/contact-geometry fields are known by completion of the event candle. "
                "They can describe or nowcast contact-candle volume; only targets beginning "
                "with the next candle are causal forward estimates from that decision point."
            ),
            "future_outcomes_used_for_event_matching": False,
            "development_outcomes_used_to_change_features": False,
            "internal_train_test_split": False,
            "train_prediction_embargo_hours": MAX_TARGET_HORIZON_HOURS,
            "higher_timeframe_source_available_at_not_after_event": True,
            "missing_source_policy": (
                "missing stays NaN and is excluded before the common event mask; never zero-filled"
            ),
            "same_model_class_and_parameters": True,
        },
        "source_contracts": {
            "manifest": str(Path(surface["manifest"]).resolve()),
            "manifest_sha256": sha256_file(Path(surface["manifest"])),
            "frozen_batch": str(FROZEN_BATCH.resolve()),
            "frozen_batch_sha256": sha256_file(FROZEN_BATCH),
            "analysis_script_sha256": sha256_file(Path(__file__)),
            "strategy_sha256": sha256_file(STRATEGY_FILE),
            "upstream_source": source_audit,
            "cache_inventory": cache_contract,
        },
        "storage": {
            "compact_record_dir": str(record_dir),
            "bulky_artifact_dir": str(artifact_dir),
            "feature_cache_dir": str(feature_dir),
            "event_cache_dir": str(event_dir),
            "save_backtest_models": False,
        },
        "runtime_preparation_snapshot": runtime_snapshot(),
        "commands": commands,
    }


def flatten_cache_audits(
    cache_inventory: Sequence[dict[str, Any]],
) -> tuple[DataFrame, DataFrame, DataFrame]:
    event_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    shuffle_rows: list[dict[str, Any]] = []
    for audit in cache_inventory:
        base = {
            "surface_id": audit["surface_id"],
            "pair": audit["pair"],
            "raw_candidate_rows": audit["raw_candidate_rows"],
            "complete_rows_before_independence": audit["complete_rows_before_independence"],
            "retained_independent_events": audit["retained_independent_events"],
            "minimum_event_gap_hours": audit["minimum_event_gap_hours"],
        }
        for count in audit["event_period_state_counts"]:
            event_rows.append({**base, **count})
        window_rows.extend(audit["training_windows"])
        for item in audit["shuffle"]:
            shuffle_rows.append({"surface_id": audit["surface_id"], **item})
    return DataFrame(event_rows), DataFrame(window_rows), DataFrame(shuffle_rows)


def prepare_run(
    *,
    run_id: str,
    surface_id: str,
    pairs: Sequence[str],
    profiles: Sequence[str],
    base_config: Path,
    python_exe: Path,
    profile_workers: int,
    cache_workers: int,
    technical_smoke: bool,
    overwrite: bool,
) -> tuple[dict[str, Any], Path, Path]:
    frozen_branch = validate_frozen_branch()
    surface = SURFACES[surface_id]
    manifest = load_manifest(Path(surface["manifest"]))
    record_dir = RECORD_ROOT / run_id
    artifact_dir = ARTIFACT_ROOT / run_id
    manifest_path = record_dir / "manifest.json"
    if manifest_path.is_file() and not overwrite:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "surface_id": surface_id,
            "pairs": list(pairs),
            "profiles": list(profiles),
            "profile_workers": int(profile_workers),
            "technical_smoke_not_evidence": bool(technical_smoke),
        }
        mismatches = {
            key: (existing.get(key), value)
            for key, value in expected.items()
            if existing.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Existing G4D manifest request differs: {mismatches}")
        return existing, manifest_path, artifact_dir
    if manifest_path.exists() and overwrite:
        raise ValueError(
            "Do not overwrite an existing G4D run record. Use a new run id for a repair."
        )

    source_record_path, source_record = source_run_record(surface)
    source_audit = source_contract_audit(
        surface=surface,
        pairs=pairs,
        source_record_path=source_record_path,
        source_record=source_record,
    )
    wide = causal_wide_frame(manifest)
    feature_dir = artifact_dir / "feature_cache"
    event_dir = artifact_dir / "event_cache"
    cache_inventory: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=cache_workers) as pool:
        futures = {
            pool.submit(
                build_pair_caches,
                pair,
                surface_id=surface_id,
                surface=surface,
                manifest=manifest,
                source_record=source_record,
                wide=wide,
                feature_dir=feature_dir,
                event_dir=event_dir,
                overwrite=overwrite,
            ): pair
            for pair in pairs
        }
        for future in as_completed(futures):
            audit = future.result()
            cache_inventory.append(audit)
            print(
                json.dumps(
                    {
                        "phase": "g4d_cache_preflight",
                        "pair": audit["pair"],
                        "events": audit["retained_independent_events"],
                    }
                ),
                flush=True,
            )
    cache_inventory.sort(key=lambda item: pairs.index(item["pair"]))
    record_dir.mkdir(parents=True, exist_ok=True)
    event_inventory, training_windows, shuffle_audit = flatten_cache_audits(cache_inventory)
    atomic_write_parquet(event_inventory, record_dir / "g4d_event_inventory.parquet")
    atomic_write_parquet(training_windows, record_dir / "g4d_training_window_support.parquet")
    atomic_write_parquet(shuffle_audit, record_dir / "g4d_shuffle_audit.parquet")
    atomic_write_json(
        {
            "created_at_utc": utc_now(),
            "surface_id": surface_id,
            "outcomes_opened_for_comparison": False,
            "direction_prediction": False,
            "profit_optimization": False,
            "source_contract": source_audit,
            "pair_audits": cache_inventory,
        },
        record_dir / "g4d_outcome_blind_preflight.json",
    )
    run_manifest = build_manifest(
        run_id=run_id,
        surface_id=surface_id,
        surface=surface,
        record_dir=record_dir,
        artifact_dir=artifact_dir,
        base_config=base_config,
        python_exe=python_exe,
        profiles=profiles,
        pairs=pairs,
        profile_workers=profile_workers,
        technical_smoke=technical_smoke,
        cache_inventory=cache_inventory,
        source_audit=source_audit,
        frozen_branch=frozen_branch,
    )
    atomic_write_json(run_manifest, manifest_path)
    return run_manifest, manifest_path, artifact_dir


def preflight_run(  # noqa: C901 - one auditable gate intentionally checks every launch contract
    manifest: dict[str, Any],
    *,
    python_exe: Path,
) -> dict[str, Any]:
    problems: list[str] = []
    warnings: list[str] = []
    sources = manifest["source_contracts"]
    for label, path, expected in (
        ("analysis script", Path(__file__), sources["analysis_script_sha256"]),
        ("research strategy", STRATEGY_FILE, sources["strategy_sha256"]),
        ("cohort manifest", Path(sources["manifest"]), sources["manifest_sha256"]),
        ("frozen G4 batch", Path(sources["frozen_batch"]), sources["frozen_batch_sha256"]),
    ):
        if not path.is_file():
            problems.append(f"missing {label}: {path}")
        elif sha256_file(path) != expected:
            problems.append(f"changed {label}: {path}")
    if not python_exe.is_file():
        problems.append(f"missing worker interpreter: {python_exe}")

    feature_dir = Path(manifest["storage"]["feature_cache_dir"])
    event_dir = Path(manifest["storage"]["event_cache_dir"])
    inventory = {item["pair"]: item for item in sources["cache_inventory"]}
    cache_rows: list[dict[str, Any]] = []
    for pair in manifest["pairs"]:
        recorded = inventory.get(pair)
        if recorded is None:
            problems.append(f"cache inventory missing pair: {pair}")
            continue
        feature_path = feature_dir / f"{pair_stem(pair)}.parquet"
        event_path = event_dir / f"{pair_stem(pair)}.parquet"
        for label, path, digest_key in (
            ("feature", feature_path, "feature_sha256"),
            ("event", event_path, "event_sha256"),
        ):
            if not path.is_file():
                problems.append(f"missing {label} cache: {path}")
            elif sha256_file(path) != recorded[digest_key]:
                problems.append(f"changed {label} cache: {path}")
        if not feature_path.is_file() or not event_path.is_file():
            continue
        events = pd.read_parquet(event_path)
        features = pd.read_parquet(feature_path)
        events["date"] = normalize_dates(events["date"])
        events["source_available_at"] = normalize_dates(events["source_available_at"])
        features["date"] = normalize_dates(features["date"])
        if events["date"].duplicated().any():
            problems.append(f"duplicate event dates: {event_path}")
        if features["date"].duplicated().any():
            problems.append(f"duplicate feature dates: {feature_path}")
        if events[list(TARGET_COLUMNS)].isna().any().any():
            problems.append(f"missing G4D target values: {event_path}")
        if (events["source_available_at"] > events["date"]).any():
            problems.append(f"future source timestamps: {event_path}")
        ordered = events["date"].sort_values()
        minimum_gap = (
            float(ordered.diff().dropna().dt.total_seconds().min() / 3600.0)
            if len(ordered) > 1
            else np.nan
        )
        if np.isfinite(minimum_gap) and minimum_gap <= MAX_TARGET_HORIZON_HOURS:
            problems.append(f"overlapping event futures for {pair}: {minimum_gap}h")
        feature_blocks = {
            prefix: [column for column in features if column.startswith(f"{prefix}__")]
            for prefix in ("wide", "level", "geometry", "placebo", "mtf", "interaction")
        }
        for prefix, columns in feature_blocks.items():
            if not columns:
                problems.append(f"empty {prefix} feature block for {pair}")
        if len(feature_blocks["placebo"]) != (
            len(feature_blocks["level"]) + len(feature_blocks["geometry"])
        ):
            problems.append(f"current/placebo feature count mismatch for {pair}")
        selected_dates = events[["date"]]
        event_features = selected_dates.merge(
            features,
            on="date",
            how="left",
            validate="one_to_one",
        )
        used = [
            column
            for columns in feature_blocks.values()
            for column in columns
        ]
        if event_features[used].replace([np.inf, -np.inf], np.nan).isna().any().any():
            problems.append(f"missing or infinite feature values on event rows for {pair}")
        key_frame = DataFrame({"pair": pair, "date": events["date"]})
        digest = eligibility_digest(key_frame)
        if digest != recorded["event_key_digest"]:
            problems.append(f"event-key digest changed for {pair}")
        cache_rows.append(
            {
                "pair": pair,
                "event_rows": len(events),
                "periods": int(events["period"].nunique()),
                "minimum_event_gap_hours": minimum_gap,
                "event_key_digest": digest,
                "feature_block_counts": {
                    key: len(value) for key, value in feature_blocks.items()
                },
            }
        )

    support_path = Path(manifest["storage"]["compact_record_dir"]) / (
        "g4d_training_window_support.parquet"
    )
    if not support_path.is_file():
        problems.append(f"missing training-window support audit: {support_path}")
        weak_windows = DataFrame()
    else:
        support = pd.read_parquet(support_path)
        weak_windows = support.loc[~support["training_support_at_least_20"]].copy()
        if not weak_windows.empty:
            weak_pairs = sorted(weak_windows["pair"].unique())
            warnings.append(
                "Some rolling pair windows contain fewer than 20 source events; "
                f"they remain visible but cannot support a strong inference: {weak_pairs}"
            )
        zero_prediction = support.loc[support["prediction_event_rows"].eq(0)]
        if not zero_prediction.empty:
            warnings.append(
                f"{len(zero_prediction)} pair-period segments contain no scoreable source events."
            )
    return {
        "created_at_utc": utc_now(),
        "surface_id": manifest["surface_id"],
        "profiles": len(manifest["profiles"]),
        "pairs": len(manifest["pairs"]),
        "targets": len(TARGET_COLUMNS),
        "outcomes_opened_for_comparison": False,
        "direction_prediction": False,
        "profit_optimization": False,
        "runtime_snapshot": runtime_snapshot(),
        "cache_audit": cache_rows,
        "weak_training_window_rows": len(weak_windows),
        "warnings": warnings,
        "problems": problems,
        "passed": not problems,
    }


def run_manifest(manifest: dict[str, Any], manifest_path: Path) -> int:
    manifest["status"] = "running"
    manifest["orchestrator_pid"] = os.getpid()
    manifest.setdefault("started_at_utc", utc_now())
    atomic_write_json(manifest, manifest_path)
    items_by_id = {item["profile_id"]: item for item in manifest["commands"]}
    pending = [item for item in manifest["commands"] if item.get("status") != "completed"]
    workers = int(manifest["profile_workers"])
    processed = len(manifest["commands"]) - len(pending)
    for start in range(0, len(pending), workers):
        batch = pending[start : start + workers]
        for item in batch:
            item["status"] = "running"
            item["started_at_utc"] = utc_now()
        atomic_write_json(manifest, manifest_path)
        with ThreadPoolExecutor(max_workers=len(batch)) as pool:
            futures = {pool.submit(run_freqai_profile, item): item for item in batch}
            for future in as_completed(futures):
                result = future.result()
                item = items_by_id[str(result["profile_id"])]
                item.update(result)
                item["status"] = "completed" if result["returncode"] == 0 else "failed"
                processed += 1
                atomic_write_json(manifest, manifest_path)
                print(
                    json.dumps(
                        {
                            "phase": "g4d_freqai_profiles",
                            "processed": processed,
                            "total": len(manifest["commands"]),
                            "profile_id": item["profile_id"],
                            "status": item["status"],
                            "stderr_log": item["stderr_log"],
                        }
                    ),
                    flush=True,
                )
        failures = [item for item in batch if item["status"] == "failed"]
        if failures:
            manifest["status"] = "failed"
            manifest["failed_profile_ids"] = [item["profile_id"] for item in failures]
            manifest["finished_at_utc"] = utc_now()
            atomic_write_json(manifest, manifest_path)
            return int(failures[0]["returncode"])
    return 0


def load_event_cache(event_dir: Path, pair: str) -> DataFrame:
    path = event_dir / f"{pair_stem(pair)}.parquet"
    frame = pd.read_parquet(path)
    frame["date"] = normalize_dates(frame["date"])
    return frame


def development_period(manifest: dict[str, Any]) -> str:
    source = load_manifest(Path(manifest["source_contracts"]["manifest"]))
    matches = [
        str(item["id"])
        for item in source["data"]["chronological_periods"]
        if item.get("role") == "development"
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one development period, found {matches}.")
    return matches[0]


def target_band_thresholds(
    events_by_pair: dict[str, DataFrame],
    *,
    development: str,
) -> tuple[dict[tuple[str, str], tuple[float, float]], DataFrame]:
    thresholds: dict[tuple[str, str], tuple[float, float]] = {}
    rows: list[dict[str, Any]] = []
    for pair, events in events_by_pair.items():
        dev = events.loc[events["period"].eq(development)]
        for target in TARGET_COLUMNS:
            values = pd.to_numeric(dev[target], errors="coerce").dropna()
            q1 = float(values.quantile(1.0 / 3.0)) if len(values) >= MIN_SCORABLE_ROWS else np.nan
            q2 = float(values.quantile(2.0 / 3.0)) if len(values) >= MIN_SCORABLE_ROWS else np.nan
            if np.isfinite(q1) and np.isfinite(q2) and q2 > q1:
                thresholds[(pair, target)] = (q1, q2)
                status = "supported"
            else:
                status = "unscorable"
            rows.append(
                {
                    "pair": pair,
                    "target": target,
                    "development_rows": len(values),
                    "lower_to_middle_boundary": q1,
                    "middle_to_upper_boundary": q2,
                    "status": status,
                }
            )
    return thresholds, DataFrame(rows)


def assign_bands(
    frame: DataFrame,
    *,
    thresholds: dict[tuple[str, str], tuple[float, float]],
) -> DataFrame:
    output = frame.copy()
    output["actual_band"] = np.nan
    output["prediction_band"] = np.nan
    for (pair, target), (lower, upper) in thresholds.items():
        selected = output["pair"].eq(pair) & output["target"].eq(target)
        if not selected.any():
            continue
        for source, destination in (
            ("actual", "actual_band"),
            ("prediction", "prediction_band"),
        ):
            values = pd.to_numeric(output.loc[selected, source], errors="coerce")
            output.loc[selected, destination] = np.select(
                (values.le(lower), values.le(upper)),
                (0.0, 1.0),
                default=2.0,
            )
    return output


def regression_metrics(frame: DataFrame) -> dict[str, Any]:
    columns = ["prediction", "actual", "prediction_band", "actual_band"]
    clean = frame[columns].apply(pd.to_numeric, errors="coerce")
    numeric = clean[["prediction", "actual"]].dropna()
    result: dict[str, Any] = {"rows": len(numeric)}
    if (
        len(numeric) < MIN_SCORABLE_ROWS
        or numeric["prediction"].nunique() < 2
        or numeric["actual"].nunique() < 2
    ):
        return {**result, "status": "unscorable"}
    errors = numeric["prediction"] - numeric["actual"]
    residual_sum_squares = float(np.square(errors).sum())
    total_sum_squares = float(
        np.square(numeric["actual"] - numeric["actual"].mean()).sum()
    )
    ranked = numeric.sort_values("prediction")
    bucket = max(1, int(len(ranked) * 0.2))
    slope, intercept = np.polyfit(
        numeric["prediction"].to_numpy(dtype=float),
        numeric["actual"].to_numpy(dtype=float),
        1,
    )
    bands = clean[["prediction_band", "actual_band"]].dropna()
    result.update(
        {
            "status": "scored",
            "actual_mean": float(numeric["actual"].mean()),
            "prediction_mean": float(numeric["prediction"].mean()),
            "prediction_bias": float(errors.mean()),
            "mean_absolute_error": float(errors.abs().mean()),
            "root_mean_squared_error": float(np.sqrt(np.square(errors).mean())),
            "r_squared": float(1.0 - residual_sum_squares / total_sum_squares),
            "prediction_actual_spearman": float(
                numeric["prediction"].corr(numeric["actual"], method="spearman")
            ),
            "calibration_slope": float(slope),
            "calibration_intercept": float(intercept),
            "top_quintile_actual_mean": float(ranked.tail(bucket)["actual"].mean()),
            "bottom_quintile_actual_mean": float(ranked.head(bucket)["actual"].mean()),
            "top_minus_bottom": float(
                ranked.tail(bucket)["actual"].mean()
                - ranked.head(bucket)["actual"].mean()
            ),
            "band_rows": len(bands),
            "mean_absolute_band_error": (
                float((bands["prediction_band"] - bands["actual_band"]).abs().mean())
                if not bands.empty
                else np.nan
            ),
            "exact_band_accuracy": (
                float(bands["prediction_band"].eq(bands["actual_band"]).mean())
                if not bands.empty
                else np.nan
            ),
        }
    )
    return result


def prediction_event_surface(
    manifest: dict[str, Any],
    *,
    record_dir: Path,
) -> tuple[DataFrame, dict[str, Any], DataFrame]:
    pairs = tuple(str(pair) for pair in manifest["pairs"])
    predictions: dict[str, DataFrame] = {}
    file_audits: list[dict[str, Any]] = []
    for item in manifest["commands"]:
        if item.get("status") != "completed":
            continue
        profile_id = str(item["profile_id"])
        frame, audit = load_predictions(Path(item["model_dir"]), pairs)
        audit["profile_id"] = profile_id
        file_audits.append(audit)
        if frame.empty:
            raise ValueError(f"No FreqAI predictions found for completed profile {profile_id}.")
        missing = sorted(set(TARGET_COLUMNS).difference(frame.columns))
        if missing:
            raise ValueError(f"Profile {profile_id} is missing targets: {missing}")
        predictions[profile_id] = frame
    if not predictions:
        raise ValueError("No completed G4D profile predictions are available.")

    common_keys, eligibility = common_prediction_keys(predictions, pairs)
    if common_keys.empty:
        raise ValueError("G4D profiles have no common prediction keys.")
    atomic_write_parquet(eligibility, record_dir / "g4d_prediction_eligibility.parquet")
    event_dir = Path(manifest["storage"]["event_cache_dir"])
    events_by_pair = {pair: load_event_cache(event_dir, pair) for pair in pairs}
    development = development_period(manifest)
    thresholds, threshold_frame = target_band_thresholds(
        events_by_pair,
        development=development,
    )
    validation_periods = set(str(value) for value in manifest["validation_periods"])

    rows: list[DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []
    for profile_id, prediction_frame in predictions.items():
        fair = prediction_frame.merge(
            common_keys,
            on=["pair", "date"],
            how="inner",
            validate="one_to_one",
        )
        for pair in pairs:
            events = events_by_pair[pair]
            events = events.loc[events["period"].isin(validation_periods)].copy()
            pair_predictions = fair.loc[fair["pair"].eq(pair)]
            merged = events.merge(
                pair_predictions[["date", *TARGET_COLUMNS]],
                on="date",
                how="inner",
                suffixes=("_actual", "_prediction"),
                validate="one_to_one",
            )
            coverage_rows.append(
                {
                    "profile_id": profile_id,
                    "pair": pair,
                    "eligible_validation_events": len(events),
                    "common_prediction_events": len(merged),
                    "excluded_validation_events": len(events) - len(merged),
                }
            )
            for target in TARGET_COLUMNS:
                block = merged[
                    ["date", "period", f"{target}_actual", f"{target}_prediction"]
                ].copy()
                block = block.rename(
                    columns={
                        f"{target}_actual": "actual",
                        f"{target}_prediction": "prediction",
                    }
                )
                block["profile_id"] = profile_id
                block["pair"] = pair
                block["target"] = target
                rows.append(block)
    output = pd.concat(rows, ignore_index=True) if rows else DataFrame()
    output = assign_bands(output, thresholds=thresholds)
    if output[["prediction", "actual"]].replace([np.inf, -np.inf], np.nan).isna().any().any():
        raise ValueError("Common G4D event surface contains missing predictions or targets.")
    duplicates = int(output.duplicated(["profile_id", "pair", "date", "target"]).sum())
    if duplicates:
        raise ValueError(f"G4D scoring surface has {duplicates} duplicate prediction keys.")
    missing_pairs = sorted(set(pairs).difference(common_keys["pair"].unique()))
    audit = {
        "created_at_utc": utc_now(),
        "profiles": file_audits,
        "common_prediction_rows": len(common_keys),
        "common_prediction_key_digest": eligibility_digest(common_keys),
        "common_event_prediction_rows": len(output),
        "missing_pairs": missing_pairs,
        "event_coverage": coverage_rows,
        "development_band_thresholds_used_only_for_calibration": True,
    }
    return output, audit, threshold_frame


def score_scopes(manifest: dict[str, Any], predictions: DataFrame) -> DataFrame:
    pairs = tuple(str(pair) for pair in manifest["pairs"])
    scope_members: dict[str, tuple[str, ...]] = {
        "all_frozen_pairs": pairs,
        **{f"pair::{pair}": (pair,) for pair in pairs},
    }
    if manifest["cohort"] == "normal":
        non_btc = tuple(pair for pair in pairs if not pair.startswith("BTC/"))
        btc = tuple(pair for pair in pairs if pair.startswith("BTC/"))
        scope_members["normal_ex_btc"] = non_btc
        scope_members["btc_only"] = btc
        for omitted in non_btc:
            scope_members[f"normal_ex_btc_leave_out::{omitted}"] = tuple(
                pair for pair in non_btc if pair != omitted
            )
    else:
        scope_members["meme_all"] = pairs
        for omitted in pairs:
            scope_members[f"meme_leave_out::{omitted}"] = tuple(
                pair for pair in pairs if pair != omitted
            )

    rows: list[dict[str, Any]] = []
    for scope, members in scope_members.items():
        selected = predictions.loc[predictions["pair"].isin(members)]
        if scope.startswith("pair::"):
            scope_type = "pair"
        elif "leave_out::" in scope:
            scope_type = "leave_one_coin_out"
        elif scope == "btc_only":
            scope_type = "btc"
        else:
            scope_type = "cohort"
        for (profile_id, period, target), frame in selected.groupby(
            ["profile_id", "period", "target"],
            sort=False,
            observed=True,
        ):
            rows.append(
                {
                    "scope": scope,
                    "scope_type": scope_type,
                    "member_pairs": json.dumps(list(members)),
                    "member_pair_count": len(members),
                    "profile_id": str(profile_id),
                    "period": str(period),
                    "target": str(target),
                    **regression_metrics(frame),
                }
            )
    return DataFrame(rows)


def profile_comparisons(scores: DataFrame) -> DataFrame:
    comparisons = (
        ("market_state_vs_unconditional", "market_state", "unconditional"),
        ("level_only_vs_unconditional", "level_only", "unconditional"),
        ("combined_current_vs_market_state", "combined_current", "market_state"),
        ("combined_shuffled_vs_market_state", "combined_shuffled", "market_state"),
        ("selected_mtf_vs_combined_current", "selected_mtf", "combined_current"),
        ("limited_interactions_vs_selected_mtf", "limited_interactions", "selected_mtf"),
        ("combined_current_vs_shuffled", "combined_current", "combined_shuffled"),
    )
    keys = ["scope", "scope_type", "period", "target"]
    metric_directions = {
        "mean_absolute_error": "lower",
        "root_mean_squared_error": "lower",
        "prediction_actual_spearman": "higher",
        "mean_absolute_band_error": "lower",
        "exact_band_accuracy": "higher",
        "top_minus_bottom": "higher",
    }
    rows: list[DataFrame] = []
    for comparison_id, profile, control in comparisons:
        left = scores.loc[scores["profile_id"].eq(profile)].copy()
        right = scores.loc[scores["profile_id"].eq(control)].copy()
        if left.empty or right.empty:
            continue
        keep = [*keys, "status", "rows", *metric_directions]
        merged = left[keep].merge(
            right[keep],
            on=keys,
            how="inner",
            suffixes=("_profile", "_control"),
            validate="one_to_one",
        )
        merged["comparison_id"] = comparison_id
        merged["profile_id"] = profile
        merged["control_profile_id"] = control
        for metric, direction in metric_directions.items():
            if direction == "lower":
                merged[f"{metric}_improvement"] = (
                    merged[f"{metric}_control"] - merged[f"{metric}_profile"]
                )
            else:
                merged[f"{metric}_improvement"] = (
                    merged[f"{metric}_profile"] - merged[f"{metric}_control"]
                )
        rows.append(merged)
    return pd.concat(rows, ignore_index=True) if rows else DataFrame()


def retention_decisions(
    manifest: dict[str, Any],
    comparisons: DataFrame,
) -> DataFrame:
    if manifest.get("technical_smoke_not_evidence"):
        return DataFrame(
            [
                {
                    "surface_id": manifest["surface_id"],
                    "target": target,
                    "status": "technical_smoke_not_evidence",
                    "reason": "The smoke run checks plumbing only.",
                }
                for target in TARGET_COLUMNS
            ]
        )
    required_profiles = set(PROFILES)
    if set(manifest["profiles"]) != required_profiles:
        return DataFrame(
            [
                {
                    "surface_id": manifest["surface_id"],
                    "target": target,
                    "status": "incomplete_profile_ladder",
                    "reason": "All seven frozen profiles are required for an evidence verdict.",
                }
                for target in TARGET_COLUMNS
            ]
        )

    cohort_scope = "normal_ex_btc" if manifest["cohort"] == "normal" else "meme_all"
    periods = tuple(str(period) for period in manifest["validation_periods"])
    pair_scopes = comparisons["scope_type"].eq("pair")
    if manifest["cohort"] == "normal":
        pair_scopes &= ~comparisons["scope"].eq("pair::BTC/USDT:USDT")
    rows: list[dict[str, Any]] = []
    for target in TARGET_COLUMNS:
        checks: list[dict[str, Any]] = []
        complete_support = True
        all_passed = True
        for period in periods:
            current = comparisons.loc[
                comparisons["comparison_id"].eq("combined_current_vs_market_state")
                & comparisons["scope"].eq(cohort_scope)
                & comparisons["period"].eq(period)
                & comparisons["target"].eq(target)
            ]
            placebo = comparisons.loc[
                comparisons["comparison_id"].eq("combined_current_vs_shuffled")
                & comparisons["scope"].eq(cohort_scope)
                & comparisons["period"].eq(period)
                & comparisons["target"].eq(target)
            ]
            coin_rows = comparisons.loc[
                comparisons["comparison_id"].eq("combined_current_vs_market_state")
                & pair_scopes
                & comparisons["period"].eq(period)
                & comparisons["target"].eq(target)
                & comparisons["status_profile"].eq("scored")
                & comparisons["status_control"].eq("scored")
            ]
            supported = (
                len(current) == 1
                and len(placebo) == 1
                and current.iloc[0]["status_profile"] == "scored"
                and current.iloc[0]["status_control"] == "scored"
                and placebo.iloc[0]["status_profile"] == "scored"
                and placebo.iloc[0]["status_control"] == "scored"
                and len(coin_rows) >= MIN_RETAIN_COINS
            )
            complete_support &= supported
            if supported:
                current_row = current.iloc[0]
                placebo_row = placebo.iloc[0]
                positive_coins = int(
                    pd.to_numeric(
                        coin_rows["mean_absolute_error_improvement"], errors="coerce"
                    ).gt(0.0).sum()
                )
                conditions = {
                    "aggregate_mae_improved": bool(
                        current_row["mean_absolute_error_improvement"] > 0.0
                    ),
                    "aggregate_rank_not_worse": bool(
                        current_row["prediction_actual_spearman_improvement"] >= 0.0
                    ),
                    "positive_coin_count_at_least_five": positive_coins >= MIN_RETAIN_COINS,
                    "current_better_than_shuffled_mae": bool(
                        placebo_row["mean_absolute_error_improvement"] > 0.0
                    ),
                }
                period_passed = all(conditions.values())
                check = {
                    "period": period,
                    "supported": True,
                    "current_vs_market_mae_improvement": float(
                        current_row["mean_absolute_error_improvement"]
                    ),
                    "current_vs_market_spearman_improvement": float(
                        current_row["prediction_actual_spearman_improvement"]
                    ),
                    "current_vs_shuffled_mae_improvement": float(
                        placebo_row["mean_absolute_error_improvement"]
                    ),
                    "positive_coin_count": positive_coins,
                    "scoreable_coin_count": len(coin_rows),
                    "conditions": conditions,
                    "passed": period_passed,
                }
            else:
                period_passed = False
                check = {
                    "period": period,
                    "supported": False,
                    "cohort_current_rows": len(current),
                    "cohort_placebo_rows": len(placebo),
                    "scoreable_coin_count": len(coin_rows),
                    "passed": False,
                }
            all_passed &= period_passed
            checks.append(check)
        if all_passed and complete_support:
            status = "retained_research_lead"
            reason = (
                "Current level/geometry information improved unseen reaction estimation "
                "in both validations, across at least five coins, and beat its shuffled control."
            )
        elif not complete_support:
            status = "parked_insufficient_support"
            reason = (
                "The identical-row validation surface did not contain enough scoreable "
                "cohort and coin cells to apply the frozen interpretation."
            )
        else:
            status = "parked_not_repeated_beyond_controls"
            reason = (
                "The current level/geometry block did not repeat beyond market state and "
                "the shuffled level control in both chronological validations."
            )
        rows.append(
            {
                "surface_id": manifest["surface_id"],
                "cohort": manifest["cohort"],
                "target": target,
                "status": status,
                "reason": reason,
                "period_checks": json.dumps(checks, sort_keys=True),
                "direction_prediction": False,
                "profit_optimization": False,
            }
        )
    return DataFrame(rows)


def score_run(
    manifest: dict[str, Any],
    *,
    record_dir: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    predictions, prediction_audit, thresholds = prediction_event_surface(
        manifest,
        record_dir=record_dir,
    )
    artifact_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = artifact_dir / "g4d_common_event_predictions.parquet"
    atomic_write_parquet(predictions, prediction_path)
    threshold_path = record_dir / "g4d_development_band_thresholds.parquet"
    atomic_write_parquet(thresholds, threshold_path)
    scores = score_scopes(manifest, predictions)
    score_path = record_dir / "g4d_regression_scores.parquet"
    atomic_write_parquet(scores, score_path)
    comparisons = profile_comparisons(scores)
    comparison_path = record_dir / "g4d_profile_comparisons.parquet"
    atomic_write_parquet(comparisons, comparison_path)
    decisions = retention_decisions(manifest, comparisons)
    decision_path = record_dir / "g4d_surface_decisions.parquet"
    atomic_write_parquet(decisions, decision_path)
    audit_path = record_dir / "g4d_prediction_audit.json"
    atomic_write_json(prediction_audit, audit_path)
    result = {
        "created_at_utc": utc_now(),
        "surface_id": manifest["surface_id"],
        "technical_smoke_not_evidence": manifest["technical_smoke_not_evidence"],
        "profiles": int(predictions["profile_id"].nunique()),
        "pairs": int(predictions["pair"].nunique()),
        "targets": int(predictions["target"].nunique()),
        "common_long_prediction_rows": len(predictions),
        "score_rows": len(scores),
        "comparison_rows": len(comparisons),
        "decision_counts": {
            str(key): int(value)
            for key, value in decisions["status"].value_counts().to_dict().items()
        },
        "direction_prediction": False,
        "profit_optimization": False,
        "prediction_artifact": str(prediction_path),
        "development_band_thresholds": str(threshold_path),
        "scores": str(score_path),
        "comparisons": str(comparison_path),
        "decisions": str(decision_path),
        "prediction_audit": str(audit_path),
    }
    atomic_write_json(result, record_dir / "g4d_result.json")
    return result


def batch_review(run_ids: Sequence[str], *, review_id: str) -> dict[str, Any]:
    manifests: list[dict[str, Any]] = []
    decisions: list[DataFrame] = []
    for run_id in run_ids:
        record_dir = RECORD_ROOT / run_id
        manifest_path = record_dir / "manifest.json"
        decision_path = record_dir / "g4d_surface_decisions.parquet"
        if not manifest_path.is_file() or not decision_path.is_file():
            raise FileNotFoundError(manifest_path if not manifest_path.is_file() else decision_path)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            raise ValueError(f"G4D source run is not complete: {run_id}")
        if manifest.get("technical_smoke_not_evidence"):
            raise ValueError(f"A technical smoke cannot enter the G4D review: {run_id}")
        if set(manifest.get("profiles", [])) != set(PROFILES):
            raise ValueError(f"G4D run lacks the complete profile ladder: {run_id}")
        frame = pd.read_parquet(decision_path)
        frame["run_id"] = run_id
        decisions.append(frame)
        manifests.append(manifest)
    supplied_surfaces = {str(manifest["surface_id"]) for manifest in manifests}
    if supplied_surfaces != set(SURFACES):
        raise ValueError(
            "The G4D batch review requires every frozen surface exactly once: "
            f"missing={sorted(set(SURFACES) - supplied_surfaces)}, "
            f"extra={sorted(supplied_surfaces - set(SURFACES))}."
        )
    combined = pd.concat(decisions, ignore_index=True)
    retained = combined.loc[combined["status"].eq("retained_research_lead")]
    if retained.empty:
        status = "parked_no_repeated_freqai_increment"
        interpretation = (
            "Across all five frozen G3A/G3D surfaces, no target repeated a current-level "
            "increment beyond market state and the shuffled control under the frozen gate."
        )
    else:
        status = "retained_surface_specific_freqai_leads"
        interpretation = (
            "At least one frozen surface/target repeated a current-level increment beyond "
            "market state and the shuffled control. These are research leads, not trading rules."
        )
    review_dir = RECORD_ROOT / review_id
    if review_dir.exists():
        raise FileExistsError(review_dir)
    review_dir.mkdir(parents=True, exist_ok=False)
    combined_path = review_dir / "g4d_all_surface_decisions.parquet"
    atomic_write_parquet(combined, combined_path)
    result = {
        "created_at_utc": utc_now(),
        "review_id": review_id,
        "status": status,
        "interpretation": interpretation,
        "surface_runs": [
            {"run_id": manifest["run_id"], "surface_id": manifest["surface_id"]}
            for manifest in manifests
        ],
        "decision_counts": {
            str(key): int(value)
            for key, value in combined["status"].value_counts().to_dict().items()
        },
        "retained_leads": retained[
            ["run_id", "surface_id", "cohort", "target", "status"]
        ].to_dict("records"),
        "feature_timing_interpretation": {
            "contact_volume_ratio": (
                "Same-event-candle description or nowcast. Arrival and contacted-cluster "
                "classification are known during/by completion of that candle, so this is "
                "not a pre-contact volume forecast."
            ),
            "one_hour_absolute_excursion_in_prior_atr": (
                "The path starts with the next hourly candle. Event classification is known "
                "before this forward target begins."
            ),
            "four_hour_dwell_fraction": (
                "The four-close path starts with the next hourly candle. Event classification "
                "is known before this forward target begins."
            ),
            "completed_manifest_field_clarification": (
                "The completed surface manifests' contact_candle_used_as_feature=false field "
                "means direct contact-candle OHLCV was not supplied. Event-derived arrival and "
                "contact-geometry classifications were supplied and must be interpreted with "
                "the target-specific timing above."
            ),
        },
        "direction_prediction": False,
        "profit_optimization": False,
        "combined_decisions": str(combined_path),
    }
    atomic_write_json(result, review_dir / "g4d_batch_review.json")
    return result


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901 - explicit CLI state machine
    parser = argparse.ArgumentParser(
        description=(
            "Generation 4D low-dimensional FreqAI regression ladder. It estimates "
            "non-directional reaction magnitude on frozen G3A/G3D event surfaces and "
            "compares current level information with market-state and shuffled controls."
        )
    )
    parser.add_argument("--surface", choices=tuple(SURFACES))
    parser.add_argument("--run-id")
    parser.add_argument("--pairs", default="all")
    parser.add_argument("--profiles", default="all")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--profile-workers", type=int, default=4)
    parser.add_argument("--cache-workers", type=int, default=4)
    parser.add_argument("--technical-smoke", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--batch-review-run-ids")
    parser.add_argument("--batch-review-id")
    args = parser.parse_args(argv)

    if args.batch_review_run_ids:
        if not args.batch_review_id:
            raise ValueError("--batch-review-id is required for a G4D batch review.")
        result = batch_review(parse_csv(args.batch_review_run_ids), review_id=args.batch_review_id)
        print(json.dumps(result, indent=2), flush=True)
        return 0
    if not args.surface or not args.run_id:
        raise ValueError("--surface and --run-id are required for a G4D surface run.")
    for label, value in (
        ("profile-workers", args.profile_workers),
        ("cache-workers", args.cache_workers),
    ):
        if value < 1 or value > MAX_WORKERS:
            raise ValueError(f"{label} must be between 1 and {MAX_WORKERS}; received {value}.")

    surface = SURFACES[args.surface]
    source_manifest = load_manifest(Path(surface["manifest"]))
    pairs = select_pairs(source_manifest, args.pairs)
    profiles = tuple(PROFILES) if args.profiles == "all" else parse_csv(args.profiles)
    unknown_profiles = sorted(set(profiles).difference(PROFILES))
    if not profiles or unknown_profiles:
        raise ValueError(f"Invalid G4D profiles: {unknown_profiles}")
    if not args.base_config.is_file():
        raise FileNotFoundError(args.base_config)

    manifest, manifest_path, artifact_dir = prepare_run(
        run_id=args.run_id,
        surface_id=args.surface,
        pairs=pairs,
        profiles=profiles,
        base_config=args.base_config,
        python_exe=args.python,
        profile_workers=args.profile_workers,
        cache_workers=args.cache_workers,
        technical_smoke=args.technical_smoke,
        overwrite=False,
    )
    record_dir = manifest_path.parent
    preflight = preflight_run(manifest, python_exe=args.python)
    atomic_write_json(preflight, record_dir / "g4d_launch_preflight.json")
    manifest["launch_preflight"] = preflight
    atomic_write_json(manifest, manifest_path)
    print(
        json.dumps(
            {
                "phase": "g4d_launch_preflight",
                "passed": preflight["passed"],
                "problems": preflight["problems"],
                "warnings": preflight["warnings"],
            }
        ),
        flush=True,
    )
    if not preflight["passed"]:
        manifest["status"] = "preflight_failed"
        atomic_write_json(manifest, manifest_path)
        return 2
    if args.prepare_only:
        return 0

    if args.retry_failed:
        for item in manifest["commands"]:
            if item.get("status") == "failed":
                item["status"] = "pending"
        manifest.pop("failed_profile_ids", None)
        manifest["status"] = "prepared"
        atomic_write_json(manifest, manifest_path)
    elif manifest.get("status") in {"failed", "preflight_failed"}:
        raise ValueError("Use --retry-failed only after diagnosing a failed G4D profile.")

    if not args.score_only and manifest.get("status") != "completed":
        returncode = run_manifest(manifest, manifest_path)
        if returncode:
            return returncode
    incomplete = [
        item["profile_id"]
        for item in manifest["commands"]
        if item.get("status") != "completed"
    ]
    if incomplete:
        raise ValueError(f"Cannot score incomplete G4D profiles: {incomplete}")
    result = score_run(
        manifest,
        record_dir=record_dir,
        artifact_dir=artifact_dir,
    )
    manifest["status"] = "completed"
    manifest["finished_at_utc"] = utc_now()
    manifest["result"] = result
    atomic_write_json(manifest, manifest_path)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
