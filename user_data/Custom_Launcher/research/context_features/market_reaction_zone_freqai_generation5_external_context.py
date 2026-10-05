from __future__ import annotations

# Bound numerical libraries before pandas/FreqAI helpers are imported.
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
import shutil
import subprocess
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation3_external_context as g3h,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation5_external_context_preflight as preflight,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_freqai_generation0 import (  # noqa: E501
    common_prediction_keys,
    eligibility_digest,
    load_predictions,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_freqai_generation1 import (  # noqa: E501
    run_profile as run_freqai_profile,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_freqai_generation4 import (  # noqa: E501
    MIN_SCORABLE_ROWS,
    regression_metrics,
    runtime_snapshot,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_freqai_generation5 import (  # noqa: E501
    BLOCK_BOOTSTRAP_SAMPLES,
    BLOCK_DAYS,
    deterministic_block_bootstrap,
)
from user_data.Custom_Launcher.research.context_features.market_reaction_zone_generation0 import (
    atomic_write_json,
    atomic_write_parquet,
    normalize_dates,
    sha256_file,
    utc_now,
)
from user_data.strategies.MarketReactionZoneFreqAIGeneration5ExternalContextStrategy import (
    MAX_TARGET_HORIZON_HOURS,
    TARGET_COLUMN,
)


USER_DATA_DIR = REPO_ROOT / "user_data"
DEFAULT_CONFIG = USER_DATA_DIR / "configs" / "config_market_reaction_zone_freqai.example.json"
DEFAULT_PYTHON = (
    REPO_ROOT / "runtime" / "venvs" / "freqtrade-backtest-08" / "Scripts" / "python.exe"
)
STRATEGY_PATH = USER_DATA_DIR / "strategies"
STRATEGY_FILE = STRATEGY_PATH / "MarketReactionZoneFreqAIGeneration5ExternalContextStrategy.py"
DATA_DIR = USER_DATA_DIR / "data" / "binance"
OUTPUT_ROOT = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "market_reaction_zones"
)
FROZEN_BATCH = OUTPUT_ROOT / "generation4_review" / "g5_frozen_branch_batch.json"
BRANCH_ID = "g5e_external_context_common_support_and_conditioning"
RECORD_ROOT = OUTPUT_ROOT / "generation5_branches" / BRANCH_ID
LARGE_ROOT = Path("D:/FreqTradeStuffLargeData/research_outputs/market_reaction_zones")
ARTIFACT_ROOT = LARGE_ROOT / "generation5_branches" / BRANCH_ID
G4D_RECORD_ROOT = OUTPUT_ROOT / "generation4_branches" / "g4d_freqai_reaction_ablation"
G4D_ARTIFACT_ROOT = LARGE_ROOT / "generation4_branches" / "g4d_freqai_reaction_ablation"
PREFLIGHT_RUN_ID = "g5e_external_overlap_20260820a"
PREFLIGHT_RECORD = RECORD_ROOT / PREFLIGHT_RUN_ID / "g5e_preflight_record.json"

CONTEXT_SNAPSHOT = preflight.CONTEXT_SNAPSHOT
ORDERBOOK_SNAPSHOT = preflight.ORDERBOOK_SNAPSHOT
OLD_TARGET_COLUMN = "&-g4d_contact_volume_ratio"
MAX_WORKERS = 4
MINIMUM_EVENTS_PER_PERIOD = 50
MINIMUM_COINS_PER_PERIOD = 5
MINIMUM_REGIME_EVENTS = 15
MINIMUM_TRAIN_EVENTS_PER_PAIR = 10
STALE_CONTEXT_HOURS = 24 * 7
ROLLING_CONTEXT_HOURS = 24 * 30
MINIMUM_CONTEXT_HISTORY_HOURS = 24 * 7
GLOBAL_DAILY_MAX_AGE_HOURS = 30.0
OUTPUT_SCHEMA_VERSION = 1
RUN_SCHEMA_VERSION = 1


SURFACES: dict[str, dict[str, Any]] = {
    "normal_thin_lvn_mixed_cluster": {
        "code": "ntlvn",
        "cohort": "normal",
        "g4d_run_id": "g4d_full_g3a_normal_20260820b_time_baseline_repair",
        "timerange": "20240101-20260401",
        "train_days": 730,
        "backtest_days": 90,
        "validation_periods": ("validation_early", "validation_late"),
    },
    "normal_isolated_confirmed_swing_density": {
        "code": "nisdens",
        "cohort": "normal",
        "g4d_run_id": "g4d_full_g3d_normal_isolated_20260820b_time_baseline_repair",
        "timerange": "20240101-20260401",
        "train_days": 730,
        "backtest_days": 90,
        "validation_periods": ("validation_early", "validation_late"),
    },
    "normal_confirmed_swing_density_cluster": {
        "code": "ncldens",
        "cohort": "normal",
        "g4d_run_id": "g4d_full_g3d_normal_cluster_20260820b_time_baseline_repair",
        "timerange": "20240101-20260401",
        "train_days": 730,
        "backtest_days": 90,
        "validation_periods": ("validation_early", "validation_late"),
    },
    "meme_confirmed_swing_density_cluster": {
        "code": "mcldens",
        "cohort": "meme",
        "g4d_run_id": "g4d_full_g3d_meme_cluster_20260820b_time_baseline_repair",
        "timerange": "20260101-20260714",
        "train_days": 160,
        "backtest_days": 30,
        "validation_periods": ("meme_validation_early", "meme_validation_late"),
    },
}

CONTEXTS: dict[str, dict[str, Any]] = {
    "gdelt_aggregate": {
        "code": "gdelt",
        "kind": "snapshot",
        "details": ("context_gdelt_events",),
        "columns": (
            "ctx_gdelt_event_count_1h",
            "ctx_gdelt_event_count_24h",
            "ctx_gdelt_num_mentions_24h",
            "ctx_gdelt_num_sources_24h",
            "ctx_gdelt_avg_tone_24h",
            "ctx_gdelt_goldstein_24h",
        ),
    },
    "global_market_macro": {
        "code": "macro",
        "kind": "snapshot",
        "details": ("context_global_market_macro",),
        "columns": (
            "ctx_fear_greed_value",
            "ctx_fear_greed_delta_24h",
            "ctx_btc_dominance_pct",
            "ctx_global_market_cap_change_24h",
            "ctx_btc_eth_avg_change_24h",
            "ctx_global_equity_risk_basket_change",
            "ctx_safe_haven_proxy_change",
        ),
    },
    "btc_bybit_orderbook_pressure": {
        "code": "ob",
        "kind": "orderbook",
    },
}

PROFILES: dict[str, dict[str, str]] = {
    "state_only": {
        "code": "state",
        "strategy": "MarketReactionZoneG5EStateOnlyFreqAIResearchStrategy",
        "role": "Prior-candle OHLCV, indicators, BTC, and cohort state only.",
    },
    "level": {
        "code": "level",
        "strategy": "MarketReactionZoneG5ELevelFreqAIResearchStrategy",
        "role": "State plus the fixed causal level and geometry block.",
    },
    "context_no_level": {
        "code": "ctx",
        "strategy": "MarketReactionZoneG5EContextNoLevelFreqAIResearchStrategy",
        "role": "State plus current context, with the candidate level removed.",
    },
    "level_context": {
        "code": "lvlctx",
        "strategy": "MarketReactionZoneG5ELevelContextFreqAIResearchStrategy",
        "role": "State, current level, and one current timestamp-safe context block.",
    },
    "level_stale_context": {
        "code": "lvlstale",
        "strategy": "MarketReactionZoneG5ELevelStaleContextFreqAIResearchStrategy",
        "role": "State and current level with the same context block shifted seven days.",
    },
}

COMPARISONS = (
    ("context_increment", "level_context", "level"),
    ("context_vs_seven_day_stale", "level_context", "level_stale_context"),
    ("level_residual_inside_context", "level_context", "context_no_level"),
    ("context_without_level_increment", "context_no_level", "state_only"),
)


def parse_csv(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def pair_stem(pair: str) -> str:
    return pair.strip().upper().replace("/", "_").replace(":", "_")


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def cell_id(surface_id: str, context_block: str) -> str:
    return f"{SURFACES[surface_id]['code']}_{CONTEXTS[context_block]['code']}"


def validate_frozen_contracts() -> tuple[dict[str, Any], dict[str, Any]]:
    batch = json.loads(FROZEN_BATCH.read_text(encoding="utf-8"))
    branches = [item for item in batch.get("branches", []) if item.get("id") == BRANCH_ID]
    if (
        batch.get("generation") != 5
        or batch.get("branch_layer") != 5
        or len(branches) != 1
        or branches[0].get("status") != "frozen_next_batch"
    ):
        raise ValueError("The frozen fifth-layer G5E branch contract changed.")
    if not PREFLIGHT_RECORD.is_file():
        raise FileNotFoundError(PREFLIGHT_RECORD)
    record = json.loads(PREFLIGHT_RECORD.read_text(encoding="utf-8"))
    if record.get("status") != "completed_with_fair_context_surfaces":
        raise ValueError("G5E outcome-blind external overlap did not pass.")
    if record.get("reaction_outcomes_opened") is not False:
        raise ValueError("The G5E overlap preflight was not outcome-blind.")
    opened = {
        (str(item["surface_id"]), str(item["context_block"]))
        for item in record["fair_surface_context_combinations"]
    }
    expected = {(surface, context) for surface in SURFACES for context in CONTEXTS}
    if opened != expected:
        raise ValueError(f"G5E fair-cell set changed: missing={expected - opened}")
    return branches[0], record


def surface_contract(surface_id: str) -> dict[str, Any]:
    surface = SURFACES[surface_id]
    record_dir = G4D_RECORD_ROOT / surface["g4d_run_id"]
    artifact_dir = G4D_ARTIFACT_ROOT / surface["g4d_run_id"]
    manifest_path = record_dir / "manifest.json"
    result_path = record_dir / "g4d_result.json"
    if not manifest_path.is_file() or not result_path.is_file():
        raise FileNotFoundError(manifest_path if not manifest_path.is_file() else result_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed" or result.get("surface_id") != manifest.get(
        "surface_id"
    ):
        raise ValueError(f"Incomplete G4D source for {surface_id}.")
    return {
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha256": sha256_file(manifest_path),
        "result_path": result_path,
        "result_sha256": sha256_file(result_path),
        "record_dir": record_dir,
        "artifact_dir": artifact_dir,
        "pairs": tuple(str(pair) for pair in manifest["pairs"]),
    }


def causal_regime(activity: Series, ready: Series) -> tuple[Series, DataFrame]:
    values = pd.to_numeric(activity, errors="coerce").where(ready)
    rolling = values.rolling(
        ROLLING_CONTEXT_HOURS,
        min_periods=MINIMUM_CONTEXT_HISTORY_HOURS,
        closed="left",
    )
    lower = rolling.quantile(1.0 / 3.0)
    upper = rolling.quantile(2.0 / 3.0)
    history_ready = lower.notna() & upper.notna() & upper.gt(lower)
    regime = Series("unavailable", index=activity.index, dtype="object")
    usable = ready & values.notna() & history_ready
    regime.loc[usable] = "middle"
    regime.loc[usable & values.le(lower)] = "quiet"
    regime.loc[usable & values.ge(upper)] = "shock"
    return regime, DataFrame({"activity_score": values, "lower": lower, "upper": upper})


def _snapshot_ready(
    context: DataFrame,
    masks: dict[str, dict[str, Series]],
    details: Sequence[str],
) -> Series:
    ready = Series(True, index=context.index)
    for detail in details:
        usable = masks.get(detail, {}).get("usable")
        ready &= usable.reindex(context.index).fillna(False) if usable is not None else False
    return ready


def gdelt_source(context: DataFrame, masks: dict[str, dict[str, Series]]) -> DataFrame:
    definition = CONTEXTS["gdelt_aggregate"]
    ready = _snapshot_ready(context, masks, definition["details"])
    output = DataFrame({"date": normalize_dates(context["date"])})
    for column in definition["columns"]:
        numeric = pd.to_numeric(context[column], errors="coerce")
        name = column.removeprefix("ctx_gdelt_")
        if any(token in name for token in ("count", "mentions", "sources")):
            numeric = np.log1p(numeric.clip(lower=0.0))
            name = f"log_{name}"
        output[f"value__{name}"] = numeric
    complete = output.filter(like="value__").notna().all(axis=1)
    ready &= complete
    activity = np.log1p(
        pd.to_numeric(context["ctx_gdelt_event_count_1h"], errors="coerce").clip(lower=0.0)
    )
    regime, thresholds = causal_regime(activity, ready)
    output["regime"] = regime
    output["ready"] = ready & regime.ne("unavailable")
    output["activity_score"] = thresholds["activity_score"]
    return output


def causal_daily_carry(context: DataFrame, columns: Sequence[str]) -> tuple[DataFrame, Series]:
    dates = normalize_dates(context["date"])
    output = DataFrame({"date": dates})
    ages: list[Series] = []
    safe = ~context["context_source_future_violation"].fillna(0.0).astype(bool)
    maximum = pd.to_datetime(context["ctx_max_source_available_at"], utc=True, errors="coerce")
    safe &= maximum.isna() | maximum.le(dates)
    for column in columns:
        raw = pd.to_numeric(context[column], errors="coerce").where(safe)
        observation_date = dates.where(raw.notna()).ffill()
        carried = raw.ffill()
        age = (dates - observation_date).dt.total_seconds() / 3600.0
        carried = carried.where(age.le(GLOBAL_DAILY_MAX_AGE_HOURS))
        output[f"value__{column.removeprefix('ctx_')}"] = carried
        ages.append(age.where(carried.notna()))
    age_frame = pd.concat(ages, axis=1)
    maximum_age = age_frame.max(axis=1)
    output["value__maximum_observation_age_hours"] = maximum_age
    ready = output.filter(like="value__").notna().all(axis=1)
    return output, ready


def macro_source(context: DataFrame, masks: dict[str, dict[str, Series]]) -> DataFrame:
    definition = CONTEXTS["global_market_macro"]
    output, carry_ready = causal_daily_carry(context, definition["columns"])
    declared_ready = _snapshot_ready(context, masks, definition["details"])
    ready = carry_ready & declared_ready
    changes = (
        "value__fear_greed_delta_24h",
        "value__global_market_cap_change_24h",
        "value__btc_eth_avg_change_24h",
        "value__global_equity_risk_basket_change",
        "value__safe_haven_proxy_change",
    )
    components: list[Series] = []
    for column in changes:
        values = pd.to_numeric(output[column], errors="coerce").abs()
        scale = values.rolling(
            ROLLING_CONTEXT_HOURS,
            min_periods=MINIMUM_CONTEXT_HISTORY_HOURS,
            closed="left",
        ).median()
        components.append((values / scale.replace(0.0, np.nan)).clip(0.0, 10.0))
    activity = pd.concat(components, axis=1).mean(axis=1)
    regime, thresholds = causal_regime(activity, ready)
    output["regime"] = regime
    output["ready"] = ready & regime.ne("unavailable")
    output["activity_score"] = thresholds["activity_score"]
    return output


def load_context_sources() -> tuple[dict[str, DataFrame], dict[str, Any]]:
    context, audit, masks = preflight.load_context_masks(CONTEXT_SNAPSHOT)
    sources = {
        "gdelt_aggregate": gdelt_source(context, masks),
        "global_market_macro": macro_source(context, masks),
    }
    records: dict[str, Any] = {}
    for block, source in sources.items():
        records[block] = {
            "rows": len(source),
            "ready_rows": int(source["ready"].sum()),
            "first_ready": str(source.loc[source["ready"], "date"].min()),
            "last_ready": str(source.loc[source["ready"], "date"].max()),
            "regime_rows": source.loc[source["ready"], "regime"].value_counts().to_dict(),
            "feature_columns": [column for column in source if column.startswith("value__")],
        }
    records["source_block_audit_rows"] = len(audit)
    return sources, records


def add_selection_keys(events: DataFrame, surface_id: str) -> DataFrame:
    output = events.copy()
    output["selection_key"] = output.apply(
        lambda row: (
            f"g5e|{surface_id}|{row['period']}|{row['pair']}|"
            f"{pd.Timestamp(row['date']).isoformat()}|{row['event_state']}|"
            f"{row['level_identity']}"
        ),
        axis=1,
    )
    output["selection_hash"] = output["selection_key"].map(stable_hash)
    if output["selection_key"].duplicated().any():
        raise ValueError(f"Duplicate G5E event keys for {surface_id}.")
    return output


def load_surface_rows(surface_id: str, source: dict[str, Any]) -> DataFrame:
    parts: list[DataFrame] = []
    feature_columns: tuple[str, ...] | None = None
    for pair in source["pairs"]:
        stem = pair_stem(pair)
        feature_path = source["artifact_dir"] / "feature_cache" / f"{stem}.parquet"
        event_path = source["artifact_dir"] / "event_cache" / f"{stem}.parquet"
        if not feature_path.is_file() or not event_path.is_file():
            raise FileNotFoundError(feature_path if not feature_path.is_file() else event_path)
        schema = pq.ParquetFile(feature_path).schema.names
        selected = tuple(
            column
            for column in schema
            if column == "date" or column.startswith(("wide__", "level__", "geometry__"))
        )
        if feature_columns is None:
            feature_columns = selected
        elif selected != feature_columns:
            raise ValueError(f"G5E source feature schema differs by pair for {surface_id}.")
        features = pd.read_parquet(feature_path, columns=list(selected))
        features["date"] = normalize_dates(features["date"])
        events = pd.read_parquet(event_path)
        events["date"] = normalize_dates(events["date"])
        events["source_available_at"] = normalize_dates(events["source_available_at"])
        events["pair"] = pair
        merged = events.merge(features, on="date", how="left", validate="one_to_one")
        parts.append(merged.rename(columns={OLD_TARGET_COLUMN: TARGET_COLUMN}))
    output = pd.concat(parts, ignore_index=True, sort=False)
    if output["source_available_at"].isna().any() or output["source_available_at"].gt(
        output["date"]
    ).any():
        raise ValueError(f"G5E level feature source is not causal for {surface_id}.")
    output = add_selection_keys(output, surface_id)
    validations = set(SURFACES[surface_id]["validation_periods"])
    selected = preflight.select_global_independent(
        output.loc[output["period"].isin(validations)].copy(),
        hours=preflight.GLOBAL_INDEPENDENCE_HOURS,
    )
    selected_keys = set(selected["selection_key"])
    output["score_selected"] = output["selection_key"].isin(selected_keys)
    return output.sort_values(["pair", "date"]).reset_index(drop=True)


def attach_snapshot_context(events: DataFrame, source: DataFrame) -> DataFrame:
    value_columns = [column for column in source if column.startswith("value__")]
    current = source[["date", "ready", "regime", *value_columns]].copy()
    current = current.rename(
        columns={
            "ready": "current_ready",
            "regime": "context_regime",
            **{column: column.replace("value__", "context__", 1) for column in value_columns},
        }
    )
    output = events.merge(current, on="date", how="left", validate="many_to_one")
    output["stale_target_time"] = output["date"] - pd.Timedelta(hours=STALE_CONTEXT_HOURS)
    stale = source[["date", "ready", *value_columns]].rename(
        columns={
            "date": "stale_target_time",
            "ready": "stale_ready",
            **{
                column: column.replace("value__", "stale_context__", 1)
                for column in value_columns
            },
        }
    )
    return output.merge(stale, on="stale_target_time", how="left", validate="many_to_one")


def attach_orderbook_context(events: DataFrame, orderbook: DataFrame) -> DataFrame:
    working = events.copy()
    working["high_event_time"] = working["date"]
    working = g3h.attach_orderbook_state(working, orderbook, side="high", label="current")
    working = g3h.attach_orderbook_state(
        working,
        orderbook,
        side="high",
        label="stale",
        lag_hours=STALE_CONTEXT_HOURS,
    )
    for label, prefix in (("current", "context"), ("stale", "stale_context")):
        signed = pd.to_numeric(working[f"high_{label}_signed_pressure"], errors="coerce")
        absolute = pd.to_numeric(working[f"high_{label}_absolute_pressure"], errors="coerce")
        upper = pd.to_numeric(working[f"high_{label}_upper_tertile"], errors="coerce")
        regime = working[f"high_{label}_regime"].astype(str)
        working[f"{prefix}__signed_pressure"] = signed
        working[f"{prefix}__absolute_pressure"] = absolute
        working[f"{prefix}__absolute_vs_upper_tertile"] = (
            absolute / upper.replace(0.0, np.nan)
        ).clip(0.0, 20.0)
        for state in ("quiet", "middle", "active"):
            working[f"{prefix}__regime_{state}"] = regime.eq(state).astype(float)
    working["current_ready"] = working["high_current_usable"].fillna(False).astype(bool)
    working["stale_ready"] = working["high_stale_usable"].fillna(False).astype(bool)
    working["context_regime"] = working["high_current_regime"].replace(
        {"active": "shock"}
    )
    return working


def attach_context(
    events: DataFrame,
    *,
    context_block: str,
    snapshot_sources: dict[str, DataFrame],
    orderbook: DataFrame,
) -> DataFrame:
    if CONTEXTS[context_block]["kind"] == "orderbook":
        return attach_orderbook_context(events, orderbook)
    return attach_snapshot_context(events, snapshot_sources[context_block])


def coverage_rows(
    events: DataFrame,
    *,
    surface_id: str,
    context_block: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cohort = str(SURFACES[surface_id]["cohort"])
    for period in SURFACES[surface_id]["validation_periods"]:
        selected = events.loc[events["score_selected"] & events["period"].eq(period)].copy()
        primary = (
            selected.loc[~selected["pair"].str.startswith("BTC/")]
            if cohort == "normal"
            else selected
        )
        btc = selected.loc[selected["pair"].str.startswith("BTC/")]
        regime_counts = {
            regime: {
                "rows": len(group),
                "coins": int(group["pair"].nunique()),
            }
            for regime, group in primary.groupby("context_regime", observed=True)
        }
        passed = len(primary) >= MINIMUM_EVENTS_PER_PERIOD and primary["pair"].nunique() >= 5
        rows.append(
            {
                "cell_id": cell_id(surface_id, context_block),
                "surface_id": surface_id,
                "cohort": cohort,
                "context_block": context_block,
                "period": period,
                "primary_rows": len(primary),
                "primary_coins": int(primary["pair"].nunique()),
                "btc_rows": len(btc),
                "regime_counts": json.dumps(regime_counts, sort_keys=True),
                "status": "supported" if passed else "insufficient_post_control_support",
                "reaction_outcomes_used_for_gate": False,
            }
        )
    return rows


def write_cell_caches(
    events: DataFrame,
    *,
    surface_id: str,
    context_block: str,
    feature_dir: Path,
    event_dir: Path,
) -> tuple[list[dict[str, Any]], DataFrame]:
    context_columns = [column for column in events if column.startswith("context__")]
    stale_columns = [column for column in events if column.startswith("stale_context__")]
    base_columns = [
        column
        for column in events
        if column.startswith(("wide__", "level__", "geometry__"))
    ]
    required = [*base_columns, *context_columns, *stale_columns, TARGET_COLUMN]
    numeric = events[required].apply(pd.to_numeric, errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )
    ready = (
        events["current_ready"].fillna(False).astype(bool)
        & events["stale_ready"].fillna(False).astype(bool)
        & events["context_regime"].isin(("quiet", "middle", "shock"))
        & numeric.notna().all(axis=1)
    )
    complete = events.loc[ready].copy().reset_index(drop=True)
    validation_periods = set(SURFACES[surface_id]["validation_periods"])
    training_counts = (
        complete.loc[~complete["period"].isin(validation_periods)]
        .groupby("pair", observed=True)
        .size()
    )
    eligible_pairs = set(
        training_counts.loc[
            training_counts.ge(MINIMUM_TRAIN_EVENTS_PER_PAIR)
        ].index
    )
    complete = complete.loc[complete["pair"].isin(eligible_pairs)].reset_index(drop=True)
    coverage = DataFrame(
        coverage_rows(
            complete,
            surface_id=surface_id,
            context_block=context_block,
        )
    )
    inventories: list[dict[str, Any]] = []
    for pair, frame in complete.groupby("pair", sort=True, observed=True):
        feature_cache = frame[["date", *base_columns, *context_columns, *stale_columns]].copy()
        event_cache = frame[
            [
                "date",
                "period",
                "event_state",
                "level_identity",
                "source_available_at",
                "context_regime",
                "score_selected",
                TARGET_COLUMN,
            ]
        ].copy()
        stem = pair_stem(str(pair))
        feature_path = feature_dir / f"{stem}.parquet"
        event_path = event_dir / f"{stem}.parquet"
        feature_path.parent.mkdir(parents=True, exist_ok=True)
        event_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_parquet(feature_cache, feature_path)
        atomic_write_parquet(event_cache, event_path)
        inventories.append(
            {
                "pair": str(pair),
                "feature_path": str(feature_path),
                "feature_sha256": sha256_file(feature_path),
                "event_path": str(event_path),
                "event_sha256": sha256_file(event_path),
                "complete_identical_profile_rows": len(event_cache),
                "score_rows": int(event_cache["score_selected"].sum()),
                "period_rows": event_cache.groupby("period").size().to_dict(),
                "event_key_digest": eligibility_digest(
                    event_cache.assign(pair=pair)[["pair", "date"]]
                ),
            }
        )
    return inventories, coverage


def profile_config(
    base: dict[str, Any],
    *,
    identifier: str,
    pairs: Sequence[str],
    feature_dir: Path,
    event_dir: Path,
    surface: dict[str, Any],
    technical_smoke: bool,
) -> dict[str, Any]:
    config = json.loads(json.dumps(base))
    config["exchange"]["pair_whitelist"] = list(pairs)
    freqai = config["freqai"]
    freqai["enabled"] = True
    freqai["identifier"] = identifier
    freqai["train_period_days"] = int(surface["train_days"])
    freqai["backtest_period_days"] = int(surface["backtest_days"])
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
    config["market_reaction_zone_g5e"] = {
        "feature_cache_dir": str(feature_dir.resolve()),
        "event_cache_dir": str(event_dir.resolve()),
        "maximum_target_horizon_hours": MAX_TARGET_HORIZON_HOURS,
        "context_missing_policy": "unavailable_rows_excluded_not_quiet_or_zero",
        "stale_context_hours": STALE_CONTEXT_HOURS,
    }
    return config


def build_cell_commands(
    *,
    run_id: str,
    compact_cell_dir: Path,
    artifact_cell_dir: Path,
    surface_id: str,
    context_block: str,
    profiles: Sequence[str],
    pairs: Sequence[str],
    base_config: Path,
    python_exe: Path,
    feature_dir: Path,
    event_dir: Path,
    technical_smoke: bool,
) -> list[dict[str, Any]]:
    base = json.loads(base_config.read_text(encoding="utf-8"))
    surface = SURFACES[surface_id]
    cell = cell_id(surface_id, context_block)
    run_token = stable_hash(run_id)[:8]
    commands: list[dict[str, Any]] = []
    for profile_id in profiles:
        definition = PROFILES[profile_id]
        profile_dir = artifact_cell_dir / "profiles" / definition["code"]
        userdir = profile_dir / "user_data"
        export_dir = profile_dir / "export"
        identifier = f"g5e_{run_token}_{cell}_{definition['code']}"
        config_path = compact_cell_dir / f"config_{definition['code']}.json"
        atomic_write_json(
            profile_config(
                base,
                identifier=identifier,
                pairs=pairs,
                feature_dir=feature_dir,
                event_dir=event_dir,
                surface=surface,
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
            definition["strategy"],
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
                "cell_id": cell,
                "surface_id": surface_id,
                "context_block": context_block,
                "profile_id": profile_id,
                "strategy": definition["strategy"],
                "role": definition["role"],
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
    return commands


def prepare_run(
    *,
    run_id: str,
    surfaces: Sequence[str],
    contexts: Sequence[str],
    profiles: Sequence[str],
    base_config: Path,
    python_exe: Path,
    profile_workers: int,
    technical_smoke: bool,
) -> tuple[dict[str, Any], Path, Path]:
    branch, overlap_record = validate_frozen_contracts()
    record_dir = RECORD_ROOT / run_id
    artifact_dir = ARTIFACT_ROOT / run_id
    manifest_path = record_dir / "manifest.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8")), manifest_path, artifact_dir
    record_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    snapshot_sources, source_representation = load_context_sources()
    orderbook = g3h.load_causal_pressure_surface(ORDERBOOK_SNAPSHOT)
    surface_sources = {surface_id: surface_contract(surface_id) for surface_id in surfaces}
    commands: list[dict[str, Any]] = []
    cell_records: list[dict[str, Any]] = []
    all_coverage: list[DataFrame] = []
    for surface_id in surfaces:
        source = surface_sources[surface_id]
        surface_rows = load_surface_rows(surface_id, source)
        expected_count = int(
            overlap_record["input_event_counts"][surface_id]["globally_independent_events"]
        )
        actual_count = int(surface_rows["score_selected"].sum())
        if actual_count != expected_count:
            raise ValueError(
                f"G5E preflight selection changed for {surface_id}: "
                f"expected={expected_count}, actual={actual_count}"
            )
        for context_block in contexts:
            cell = cell_id(surface_id, context_block)
            compact_cell_dir = record_dir / "cells" / cell
            artifact_cell_dir = artifact_dir / "cells" / cell
            feature_dir = artifact_cell_dir / "feature_cache"
            event_dir = artifact_cell_dir / "event_cache"
            compact_cell_dir.mkdir(parents=True, exist_ok=True)
            enriched = attach_context(
                surface_rows,
                context_block=context_block,
                snapshot_sources=snapshot_sources,
                orderbook=orderbook,
            )
            inventory, coverage = write_cell_caches(
                enriched,
                surface_id=surface_id,
                context_block=context_block,
                feature_dir=feature_dir,
                event_dir=event_dir,
            )
            inventory_path = compact_cell_dir / "cache_inventory.parquet"
            coverage_path = compact_cell_dir / "post_control_coverage.parquet"
            atomic_write_parquet(DataFrame(inventory), inventory_path)
            atomic_write_parquet(coverage, coverage_path)
            all_coverage.append(coverage)
            passed = bool(
                len(coverage) == 2
                and coverage["status"].eq("supported").all()
                and len(inventory) >= MINIMUM_COINS_PER_PERIOD
            )
            cell_pairs = tuple(str(item["pair"]) for item in inventory)
            if passed:
                commands.extend(
                    build_cell_commands(
                        run_id=run_id,
                        compact_cell_dir=compact_cell_dir,
                        artifact_cell_dir=artifact_cell_dir,
                        surface_id=surface_id,
                        context_block=context_block,
                        profiles=profiles,
                        pairs=cell_pairs,
                        base_config=base_config,
                        python_exe=python_exe,
                        feature_dir=feature_dir,
                        event_dir=event_dir,
                        technical_smoke=technical_smoke,
                    )
                )
            cell_records.append(
                {
                    "cell_id": cell,
                    "surface_id": surface_id,
                    "context_block": context_block,
                    "cohort": SURFACES[surface_id]["cohort"],
                    "status": (
                        "prepared_for_outcome_models"
                        if passed
                        else "parked_post_control_support"
                    ),
                    "pairs": list(cell_pairs),
                    "profiles": list(profiles) if passed else [],
                    "feature_cache_dir": str(feature_dir),
                    "event_cache_dir": str(event_dir),
                    "inventory_path": str(inventory_path),
                    "inventory_sha256": sha256_file(inventory_path),
                    "coverage_path": str(coverage_path),
                    "coverage_sha256": sha256_file(coverage_path),
                    "cache_inventory": inventory,
                    "reaction_outcomes_interpretable": passed,
                }
            )
    combined_coverage = pd.concat(all_coverage, ignore_index=True)
    combined_coverage_path = record_dir / "g5e_post_control_coverage.parquet"
    atomic_write_parquet(combined_coverage, combined_coverage_path)
    manifest = {
        "schema_version": RUN_SCHEMA_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": run_id,
        "status": "prepared",
        "created_at_utc": utc_now(),
        "technical_smoke_not_evidence": bool(technical_smoke),
        "frozen_branch_id": BRANCH_ID,
        "objective": branch["hypothesis"],
        "strongest_alternative": branch["strongest_alternative"],
        "retain_interpretation": branch["retain_interpretation"],
        "park_interpretation": branch["park_interpretation"],
        "iteration_cap": int(branch["iteration_cap"]),
        "surfaces": list(surfaces),
        "contexts": list(contexts),
        "profiles": list(profiles),
        "profile_workers": int(profile_workers),
        "model_threads_per_profile": 1,
        "model_class": "LightGBMRegressorMultiTarget",
        "target": TARGET_COLUMN,
        "target_definition": (
            "Contact-candle volume divided by prior typical volume; reaction strength only, "
            "with no price direction or profit target."
        ),
        "comparison_ladder": [list(item) for item in COMPARISONS],
        "context_regime_contract": {
            "quiet": "bottom causal trailing-30-day third of source activity",
            "middle": "recorded but excluded from quiet-versus-shock contrast",
            "shock": "top causal trailing-30-day third of source activity",
            "minimum_history_hours": MINIMUM_CONTEXT_HISTORY_HOURS,
            "missing": "unavailable and excluded; never quiet or zero",
            "global_daily_carry_max_age_hours": GLOBAL_DAILY_MAX_AGE_HOURS,
            "stale_control_hours": STALE_CONTEXT_HOURS,
        },
        "leakage_controls": {
            "availability_mask_on_identical_training_and_scoring_rows": True,
            "level_and_context_source_not_after_event_time": True,
            "quiet_missing_and_unavailable_distinct": True,
            "seven_day_stale_context_on_same_rows": True,
            "global_validation_events_four_hours_independent": True,
            "target_values_used_for_coverage_or_regime_definition": False,
            "normal_meme_and_btc_reported_separately": True,
        },
        "declared_review_thresholds_not_native_scores": {
            "minimum_events_per_period": MINIMUM_EVENTS_PER_PERIOD,
            "minimum_coins_per_period": MINIMUM_COINS_PER_PERIOD,
            "minimum_regime_events": MINIMUM_REGIME_EVENTS,
            "bootstrap_samples": BLOCK_BOOTSTRAP_SAMPLES,
            "bootstrap_block_days": BLOCK_DAYS,
            "quiet_shock_contrast_must_repeat_with_same_sign": True,
            "quiet_shock_contrast_interval_must_exclude_zero": True,
            "current_context_must_beat_seven_day_stale_context": True,
            "residual_level_contribution_reported": True,
            "arbitrary_percentage_gate": None,
        },
        "source_contracts": {
            "frozen_batch": str(FROZEN_BATCH),
            "frozen_batch_sha256": sha256_file(FROZEN_BATCH),
            "outcome_blind_preflight": str(PREFLIGHT_RECORD),
            "outcome_blind_preflight_sha256": sha256_file(PREFLIGHT_RECORD),
            "context_snapshot": str(CONTEXT_SNAPSHOT),
            "context_snapshot_sha256": sha256_file(CONTEXT_SNAPSHOT),
            "orderbook_snapshot": str(ORDERBOOK_SNAPSHOT),
            "orderbook_snapshot_sha256": sha256_file(ORDERBOOK_SNAPSHOT),
            "analysis_script_sha256": sha256_file(Path(__file__)),
            "strategy_sha256": sha256_file(STRATEGY_FILE),
            "source_representations": source_representation,
            "g4d_surfaces": {
                key: {
                    "manifest": str(value["manifest_path"]),
                    "manifest_sha256": value["manifest_sha256"],
                    "result": str(value["result_path"]),
                    "result_sha256": value["result_sha256"],
                }
                for key, value in surface_sources.items()
            },
            "post_control_coverage": str(combined_coverage_path),
            "post_control_coverage_sha256": sha256_file(combined_coverage_path),
        },
        "storage": {
            "compact_record_dir": str(record_dir),
            "bulky_artifact_dir": str(artifact_dir),
            "save_backtest_models": False,
        },
        "cells": cell_records,
        "commands": commands,
        "direction_prediction": False,
        "profit_optimization": False,
    }
    atomic_write_json(manifest, manifest_path)
    return manifest, manifest_path, artifact_dir


def runtime_preflight(  # noqa: C901
    manifest: dict[str, Any], *, python_exe: Path
) -> dict[str, Any]:
    problems: list[str] = []
    warnings: list[str] = []
    if not python_exe.is_file():
        problems.append(f"missing worker interpreter: {python_exe}")
    if not STRATEGY_FILE.is_file():
        problems.append(f"missing strategy: {STRATEGY_FILE}")
    if not 1 <= int(manifest["profile_workers"]) <= MAX_WORKERS:
        problems.append("profile worker count is outside 1..4")
    for cell in manifest["cells"]:
        if cell["status"] != "prepared_for_outcome_models":
            continue
        for item in cell["cache_inventory"]:
            for label in ("feature", "event"):
                path = Path(str(item[f"{label}_path"]))
                if not path.is_file():
                    problems.append(f"missing cache: {path}")
                elif sha256_file(path) != item[f"{label}_sha256"]:
                    problems.append(f"cache hash changed: {path}")
    dependency: dict[str, Any] = {"returncode": None, "stdout": "", "stderr": ""}
    if python_exe.is_file():
        result = subprocess.run(
            [
                str(python_exe),
                "-c",
                "import freqtrade, lightgbm, pyarrow; print('g5e-worker-dependencies-ok')",
            ],
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
        if result.returncode:
            problems.append("worker dependency import failed")
    free_gib = round(shutil.disk_usage(ARTIFACT_ROOT).free / (1024**3), 3)
    if free_gib < 5.0:
        warnings.append(f"Only {free_gib} GiB is free on the bulky artifact drive.")
    return {
        "created_at_utc": utc_now(),
        "passed": not problems,
        "problems": problems,
        "warnings": warnings,
        "runtime_snapshot": runtime_snapshot(),
        "worker_interpreter": str(python_exe),
        "dependency_check": dependency,
        "prepared_cells": int(
            sum(cell["status"] == "prepared_for_outcome_models" for cell in manifest["cells"])
        ),
        "parked_cells": int(
            sum(cell["status"] != "prepared_for_outcome_models" for cell in manifest["cells"])
        ),
        "profiles": len(manifest["commands"]),
        "profile_workers": manifest["profile_workers"],
        "model_threads_per_profile": 1,
        "bulky_storage_free_gib": free_gib,
        "direction_prediction": False,
        "profit_optimization": False,
    }


def run_manifest(manifest: dict[str, Any], manifest_path: Path) -> int:
    manifest["status"] = "running"
    manifest["orchestrator_pid"] = os.getpid()
    manifest.setdefault("started_at_utc", utc_now())
    atomic_write_json(manifest, manifest_path)
    items_by_key = {
        (item["cell_id"], item["profile_id"]): item for item in manifest["commands"]
    }
    pending = [item for item in manifest["commands"] if item.get("status") != "completed"]
    processed = len(manifest["commands"]) - len(pending)
    workers = int(manifest["profile_workers"])
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
                submitted = futures[future]
                item = items_by_key[(submitted["cell_id"], str(result["profile_id"]))]
                item.update(result)
                item["status"] = "completed" if result["returncode"] == 0 else "failed"
                processed += 1
                atomic_write_json(manifest, manifest_path)
                print(
                    json.dumps(
                        {
                            "phase": "g5e_freqai_profiles",
                            "processed": processed,
                            "total": len(manifest["commands"]),
                            "cell_id": item["cell_id"],
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
            manifest["failed_commands"] = [
                {"cell_id": item["cell_id"], "profile_id": item["profile_id"]}
                for item in failures
            ]
            manifest["finished_at_utc"] = utc_now()
            atomic_write_json(manifest, manifest_path)
            return int(failures[0]["returncode"])
    return 0


def load_event_cache(event_dir: Path, pair: str) -> DataFrame:
    frame = pd.read_parquet(event_dir / f"{pair_stem(pair)}.parquet")
    frame["date"] = normalize_dates(frame["date"])
    return frame


def target_band_thresholds(
    events_by_pair: dict[str, DataFrame],
    *,
    validation_periods: Sequence[str],
) -> tuple[dict[str, tuple[float, float]], DataFrame]:
    validation = set(validation_periods)
    thresholds: dict[str, tuple[float, float]] = {}
    rows: list[dict[str, Any]] = []
    for pair, events in events_by_pair.items():
        values = pd.to_numeric(
            events.loc[~events["period"].isin(validation), TARGET_COLUMN],
            errors="coerce",
        ).dropna()
        low = float(values.quantile(1.0 / 3.0)) if len(values) >= MIN_SCORABLE_ROWS else np.nan
        high = float(values.quantile(2.0 / 3.0)) if len(values) >= MIN_SCORABLE_ROWS else np.nan
        supported = bool(np.isfinite(low) and np.isfinite(high) and low < high)
        if supported:
            thresholds[pair] = (low, high)
        rows.append(
            {
                "pair": pair,
                "target": TARGET_COLUMN,
                "exposed_training_rows": len(values),
                "lower_to_middle_boundary": low,
                "middle_to_upper_boundary": high,
                "status": "supported" if supported else "unscorable",
                "validation_outcomes_used": False,
            }
        )
    return thresholds, DataFrame(rows)


def assign_bands(frame: DataFrame, *, thresholds: dict[str, tuple[float, float]]) -> DataFrame:
    output = frame.copy()
    output["actual_band"] = np.nan
    output["prediction_band"] = np.nan
    for pair, (low, high) in thresholds.items():
        selected = output["pair"].eq(pair)
        for source, destination in (("actual", "actual_band"), ("prediction", "prediction_band")):
            values = pd.to_numeric(output.loc[selected, source], errors="coerce")
            output.loc[selected, destination] = np.select(
                (values.le(low), values.le(high)),
                (0.0, 1.0),
                default=2.0,
            )
    return output


def prediction_event_surface(
    manifest: dict[str, Any],
    *,
    cell: dict[str, Any],
    record_dir: Path,
) -> tuple[DataFrame, dict[str, Any], DataFrame]:
    cell_name = str(cell["cell_id"])
    pairs = tuple(str(pair) for pair in cell["pairs"])
    commands = [item for item in manifest["commands"] if item["cell_id"] == cell_name]
    predictions: dict[str, DataFrame] = {}
    file_audits: list[dict[str, Any]] = []
    for item in commands:
        if item.get("status") != "completed":
            continue
        profile_id = str(item["profile_id"])
        frame, audit = load_predictions(Path(str(item["model_dir"])), pairs)
        audit["profile_id"] = profile_id
        file_audits.append(audit)
        if frame.empty or TARGET_COLUMN not in frame:
            raise ValueError(f"No complete G5E predictions for {cell_name}/{profile_id}.")
        predictions[profile_id] = frame
    if set(predictions) != set(PROFILES):
        raise ValueError(f"G5E cell {cell_name} lacks the five fixed profiles.")
    common_keys, eligibility = common_prediction_keys(predictions, pairs)
    if common_keys.empty:
        raise ValueError(f"G5E profiles have no common prediction keys for {cell_name}.")
    atomic_write_parquet(eligibility, record_dir / "prediction_eligibility.parquet")
    event_dir = Path(str(cell["event_cache_dir"]))
    events_by_pair = {pair: load_event_cache(event_dir, pair) for pair in pairs}
    validation_periods = tuple(SURFACES[str(cell["surface_id"])]["validation_periods"])
    thresholds, threshold_frame = target_band_thresholds(
        events_by_pair,
        validation_periods=validation_periods,
    )
    rows: list[DataFrame] = []
    coverage: list[dict[str, Any]] = []
    for profile_id, frame in predictions.items():
        fair = frame.merge(common_keys, on=["pair", "date"], how="inner", validate="one_to_one")
        for pair in pairs:
            events = events_by_pair[pair]
            events = events.loc[
                events["period"].isin(validation_periods)
                & events["score_selected"].astype(bool)
            ].copy()
            pair_predictions = fair.loc[fair["pair"].eq(pair)]
            merged = events.merge(
                pair_predictions[["date", TARGET_COLUMN]],
                on="date",
                how="inner",
                suffixes=("_actual", "_prediction"),
                validate="one_to_one",
            )
            coverage.append(
                {
                    "profile_id": profile_id,
                    "pair": pair,
                    "eligible_events": len(events),
                    "common_prediction_events": len(merged),
                    "excluded_events": len(events) - len(merged),
                }
            )
            block = merged[
                [
                    "date",
                    "period",
                    "context_regime",
                    f"{TARGET_COLUMN}_actual",
                    f"{TARGET_COLUMN}_prediction",
                ]
            ].rename(
                columns={
                    f"{TARGET_COLUMN}_actual": "actual",
                    f"{TARGET_COLUMN}_prediction": "prediction",
                }
            )
            block["profile_id"] = profile_id
            block["pair"] = pair
            block["target"] = TARGET_COLUMN
            rows.append(block)
    output = pd.concat(rows, ignore_index=True) if rows else DataFrame()
    output = assign_bands(output, thresholds=thresholds)
    if output.empty or output[["prediction", "actual"]].isna().any().any():
        raise ValueError(f"G5E common event surface is empty or incomplete for {cell_name}.")
    if output.duplicated(["profile_id", "pair", "date", "target"]).any():
        raise ValueError(f"G5E scoring surface has duplicate keys for {cell_name}.")
    audit = {
        "created_at_utc": utc_now(),
        "cell_id": cell_name,
        "profile_files": file_audits,
        "common_prediction_rows": len(common_keys),
        "common_prediction_key_digest": eligibility_digest(common_keys),
        "common_event_prediction_rows": len(output),
        "event_coverage": coverage,
        "identical_rows_for_every_profile": True,
        "development_bands_exclude_validation_outcomes": True,
    }
    return output, audit, threshold_frame


def score_scopes(cell: dict[str, Any], predictions: DataFrame) -> DataFrame:
    pairs = tuple(str(pair) for pair in cell["pairs"])
    cohort = str(cell["cohort"])
    primary = (
        tuple(pair for pair in pairs if not pair.startswith("BTC/"))
        if cohort == "normal"
        else pairs
    )
    scopes: dict[str, tuple[str, ...]] = {
        "primary_cohort": primary,
        **{f"pair::{pair}": (pair,) for pair in pairs},
    }
    if cohort == "normal":
        scopes["btc_only"] = tuple(pair for pair in pairs if pair.startswith("BTC/"))
    rows: list[dict[str, Any]] = []
    for scope, members in scopes.items():
        selected = predictions.loc[predictions["pair"].isin(members)]
        scope_type = (
            "pair"
            if scope.startswith("pair::")
            else "btc"
            if scope == "btc_only"
            else "cohort"
        )
        for (profile_id, period), frame in selected.groupby(
            ["profile_id", "period"], observed=True, sort=False
        ):
            rows.append(
                {
                    "cell_id": cell["cell_id"],
                    "surface_id": cell["surface_id"],
                    "context_block": cell["context_block"],
                    "scope": scope,
                    "scope_type": scope_type,
                    "member_pairs": json.dumps(list(members)),
                    "member_pair_count": len(members),
                    "profile_id": profile_id,
                    "period": period,
                    "target": TARGET_COLUMN,
                    **regression_metrics(frame),
                }
            )
    return DataFrame(rows)


def paired_profile_comparisons(
    manifest: dict[str, Any],
    cell: dict[str, Any],
    predictions: DataFrame,
    scores: DataFrame,
) -> DataFrame:
    pairs = tuple(str(pair) for pair in cell["pairs"])
    members = (
        tuple(pair for pair in pairs if not pair.startswith("BTC/"))
        if cell["cohort"] == "normal"
        else pairs
    )
    rows: list[dict[str, Any]] = []
    for comparison_id, profile, control in COMPARISONS:
        selected = predictions.loc[
            predictions["profile_id"].isin((profile, control))
            & predictions["pair"].isin(members)
        ].copy()
        pivot = selected.pivot(
            index=["pair", "date", "period", "context_regime", "actual"],
            columns="profile_id",
            values="prediction",
        ).reset_index()
        if profile not in pivot or control not in pivot:
            continue
        pivot["paired_error_gain"] = (
            (pivot[control] - pivot["actual"]).abs()
            - (pivot[profile] - pivot["actual"]).abs()
        )
        for period in SURFACES[str(cell["surface_id"])]["validation_periods"]:
            frame = pivot.loc[pivot["period"].eq(period)].copy()
            point, lower, upper = deterministic_block_bootstrap(
                frame,
                seed_key=(
                    f"{manifest['run_id']}|{cell['cell_id']}|{comparison_id}|{period}"
                ),
            )
            coin = frame.groupby("pair", observed=True)["paired_error_gain"].mean()
            loo = [float(coin.drop(index=pair).mean()) for pair in coin.index if len(coin) > 1]
            control_mae = (
                float(
                    frame.assign(control_error=(frame[control] - frame["actual"]).abs())
                    .groupby("pair", observed=True)["control_error"]
                    .mean()
                    .mean()
                )
                if not frame.empty
                else np.nan
            )
            profile_score = scores.loc[
                scores["scope"].eq("primary_cohort")
                & scores["profile_id"].eq(profile)
                & scores["period"].eq(period)
            ]
            control_score = scores.loc[
                scores["scope"].eq("primary_cohort")
                & scores["profile_id"].eq(control)
                & scores["period"].eq(period)
            ]
            supported = bool(
                len(frame) >= MINIMUM_EVENTS_PER_PERIOD
                and len(coin) >= MINIMUM_COINS_PER_PERIOD
                and len(profile_score) == 1
                and len(control_score) == 1
                and profile_score.iloc[0]["status"] == "scored"
                and control_score.iloc[0]["status"] == "scored"
            )
            rows.append(
                {
                    "cell_id": cell["cell_id"],
                    "surface_id": cell["surface_id"],
                    "cohort": cell["cohort"],
                    "context_block": cell["context_block"],
                    "period": period,
                    "comparison_id": comparison_id,
                    "profile_id": profile,
                    "control_profile_id": control,
                    "rows": len(frame),
                    "coins": len(coin),
                    "supported": supported,
                    "equal_coin_paired_mae_gain": point,
                    "paired_mae_gain_percent_of_control": (
                        100.0 * point / control_mae
                        if np.isfinite(point) and np.isfinite(control_mae) and control_mae > 0.0
                        else np.nan
                    ),
                    "block_bootstrap_lower_95": lower,
                    "block_bootstrap_upper_95": upper,
                    "positive_coin_count": int(coin.gt(0.0).sum()),
                    "scoreable_coin_count": len(coin),
                    "leave_one_coin_out_minimum_gain": min(loo) if loo else np.nan,
                    "leave_one_coin_out_positive": bool(loo and min(loo) > 0.0),
                    "spearman_change": (
                        float(
                            profile_score.iloc[0]["prediction_actual_spearman"]
                            - control_score.iloc[0]["prediction_actual_spearman"]
                        )
                        if supported
                        else np.nan
                    ),
                    "profile_calibration_slope": (
                        float(profile_score.iloc[0]["calibration_slope"])
                        if supported
                        else np.nan
                    ),
                    "profile_band_rows": (
                        int(profile_score.iloc[0]["band_rows"]) if supported else 0
                    ),
                    "direction_prediction": False,
                    "profit_optimization": False,
                }
            )
    return DataFrame(rows)


def comparison_passed(row: pd.Series) -> bool:
    return bool(
        row["supported"]
        and row["equal_coin_paired_mae_gain"] > 0.0
        and np.isfinite(row["block_bootstrap_lower_95"])
        and row["block_bootstrap_lower_95"] > 0.0
        and row["positive_coin_count"] >= MINIMUM_COINS_PER_PERIOD
        and row["leave_one_coin_out_positive"]
        and row["spearman_change"] >= 0.0
        and np.isfinite(row["profile_calibration_slope"])
        and row["profile_calibration_slope"] > 0.0
        and row["profile_band_rows"] >= MIN_SCORABLE_ROWS
    )


def regime_contrast_bootstrap(
    frame: DataFrame,
    *,
    seed_key: str,
    samples: int = BLOCK_BOOTSTRAP_SAMPLES,
) -> tuple[float, float, float, int]:
    selected = frame.loc[frame["context_regime"].isin(("quiet", "shock"))].copy()
    if selected.empty:
        return np.nan, np.nan, np.nan, 0
    coin_regime = (
        selected.groupby(["pair", "context_regime"], observed=True)["level_error_gain"]
        .mean()
        .unstack()
        .dropna(subset=["quiet", "shock"])
    )
    if coin_regime.empty:
        return np.nan, np.nan, np.nan, 0
    point = float((coin_regime["shock"] - coin_regime["quiet"]).mean())
    selected["week"] = pd.to_datetime(selected["date"], utc=True).dt.floor(f"{BLOCK_DAYS}D")
    blocks = (
        selected.groupby(["pair", "context_regime", "week"], observed=True)[
            "level_error_gain"
        ]
        .agg(block_mean="mean", block_rows="size")
        .reset_index()
    )
    groups = {
        (str(pair), str(regime)): (
            group["block_mean"].to_numpy(dtype=float),
            group["block_rows"].to_numpy(dtype=float),
        )
        for (pair, regime), group in blocks.groupby(
            ["pair", "context_regime"], observed=True
        )
    }
    seed = int.from_bytes(hashlib.sha256(seed_key.encode("utf-8")).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        coin_changes: list[float] = []
        for pair in coin_regime.index:
            sampled: dict[str, float] = {}
            for regime in ("quiet", "shock"):
                means, row_counts = groups[(str(pair), regime)]
                positions = rng.integers(0, len(means), len(means))
                sampled[regime] = float(
                    np.average(means[positions], weights=row_counts[positions])
                )
            coin_changes.append(sampled["shock"] - sampled["quiet"])
        values.append(float(np.mean(coin_changes)))
    return (
        point,
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
        len(coin_regime),
    )


def conditional_level_effects(
    manifest: dict[str, Any],
    cell: dict[str, Any],
    predictions: DataFrame,
) -> DataFrame:
    pairs = tuple(str(pair) for pair in cell["pairs"])
    members = (
        tuple(pair for pair in pairs if not pair.startswith("BTC/"))
        if cell["cohort"] == "normal"
        else pairs
    )
    selected = predictions.loc[
        predictions["profile_id"].isin(("level_context", "context_no_level"))
        & predictions["pair"].isin(members)
    ].copy()
    pivot = selected.pivot(
        index=["pair", "date", "period", "context_regime", "actual"],
        columns="profile_id",
        values="prediction",
    ).reset_index()
    pivot["level_error_gain"] = (
        (pivot["context_no_level"] - pivot["actual"]).abs()
        - (pivot["level_context"] - pivot["actual"]).abs()
    )
    rows: list[dict[str, Any]] = []
    for period in SURFACES[str(cell["surface_id"])]["validation_periods"]:
        period_frame = pivot.loc[pivot["period"].eq(period)].copy()
        regime_stats: dict[str, dict[str, Any]] = {}
        for regime in ("quiet", "shock"):
            frame = period_frame.loc[period_frame["context_regime"].eq(regime)].copy()
            bootstrap_frame = frame.rename(columns={"level_error_gain": "paired_error_gain"})
            point, lower, upper = deterministic_block_bootstrap(
                bootstrap_frame,
                seed_key=f"{manifest['run_id']}|{cell['cell_id']}|{period}|{regime}",
            )
            coin = frame.groupby("pair", observed=True)["level_error_gain"].mean()
            regime_stats[regime] = {
                "rows": len(frame),
                "coins": len(coin),
                "equal_coin_level_mae_gain": point,
                "lower_95": lower,
                "upper_95": upper,
                "positive_coin_count": int(coin.gt(0.0).sum()),
            }
        contrast, lower, upper, contrast_coins = regime_contrast_bootstrap(
            period_frame,
            seed_key=f"{manifest['run_id']}|{cell['cell_id']}|{period}|contrast",
        )
        quiet = regime_stats["quiet"]
        shock = regime_stats["shock"]
        supported = bool(
            quiet["rows"] >= MINIMUM_REGIME_EVENTS
            and shock["rows"] >= MINIMUM_REGIME_EVENTS
            and quiet["coins"] >= MINIMUM_COINS_PER_PERIOD
            and shock["coins"] >= MINIMUM_COINS_PER_PERIOD
            and contrast_coins >= MINIMUM_COINS_PER_PERIOD
            and np.isfinite(lower)
            and np.isfinite(upper)
        )
        contrast_excludes_zero = bool(supported and (lower > 0.0 or upper < 0.0))
        residual_supported = bool(
            (
                np.isfinite(quiet["lower_95"])
                and quiet["lower_95"] > 0.0
                and quiet["positive_coin_count"] >= MINIMUM_COINS_PER_PERIOD
            )
            or (
                np.isfinite(shock["lower_95"])
                and shock["lower_95"] > 0.0
                and shock["positive_coin_count"] >= MINIMUM_COINS_PER_PERIOD
            )
        )
        rows.append(
            {
                "cell_id": cell["cell_id"],
                "surface_id": cell["surface_id"],
                "cohort": cell["cohort"],
                "context_block": cell["context_block"],
                "period": period,
                "quiet_rows": quiet["rows"],
                "quiet_coins": quiet["coins"],
                "quiet_equal_coin_level_mae_gain": quiet["equal_coin_level_mae_gain"],
                "quiet_lower_95": quiet["lower_95"],
                "quiet_upper_95": quiet["upper_95"],
                "quiet_positive_coin_count": quiet["positive_coin_count"],
                "shock_rows": shock["rows"],
                "shock_coins": shock["coins"],
                "shock_equal_coin_level_mae_gain": shock["equal_coin_level_mae_gain"],
                "shock_lower_95": shock["lower_95"],
                "shock_upper_95": shock["upper_95"],
                "shock_positive_coin_count": shock["positive_coin_count"],
                "shock_minus_quiet_level_gain": contrast,
                "contrast_lower_95": lower,
                "contrast_upper_95": upper,
                "contrast_coins": contrast_coins,
                "supported": supported,
                "contrast_excludes_zero": contrast_excludes_zero,
                "residual_level_contribution_supported": residual_supported,
                "direction_prediction": False,
                "profit_optimization": False,
            }
        )
    return DataFrame(rows)


def retention_decision(
    manifest: dict[str, Any],
    cell: dict[str, Any],
    comparisons: DataFrame,
    conditional: DataFrame,
) -> DataFrame:
    periods = tuple(SURFACES[str(cell["surface_id"])]["validation_periods"])
    required = {"context_increment", "context_vs_seven_day_stale"}
    period_checks: list[dict[str, Any]] = []
    complete = set(manifest["profiles"]) == set(PROFILES)
    repeated = complete
    contrast_signs: list[int] = []
    for period in periods:
        selected = comparisons.loc[
            comparisons["period"].eq(period)
            & comparisons["comparison_id"].isin(required)
        ]
        present = set(selected["comparison_id"]) == required
        comparison_passes = {
            str(row.comparison_id): comparison_passed(pd.Series(row._asdict()))
            for row in selected.itertuples(index=False)
        }
        conditional_row = conditional.loc[conditional["period"].eq(period)]
        conditional_present = len(conditional_row) == 1
        if conditional_present:
            row = conditional_row.iloc[0]
            contrast = float(row["shock_minus_quiet_level_gain"])
            contrast_sign = int(np.sign(contrast)) if np.isfinite(contrast) else 0
            contrast_signs.append(contrast_sign)
            conditional_passed = bool(
                row["supported"]
                and row["contrast_excludes_zero"]
                and row["residual_level_contribution_supported"]
                and contrast_sign != 0
            )
        else:
            contrast = np.nan
            contrast_sign = 0
            conditional_passed = False
        passed = bool(
            present
            and comparison_passes
            and all(comparison_passes.values())
            and conditional_passed
        )
        complete &= present and conditional_present
        repeated &= passed
        period_checks.append(
            {
                "period": period,
                "required_comparisons": sorted(required),
                "present_comparisons": sorted(selected["comparison_id"].tolist()),
                "comparison_passes": comparison_passes,
                "conditional_supported": bool(
                    conditional_present and conditional_row.iloc[0]["supported"]
                ),
                "quiet_shock_contrast": contrast,
                "quiet_shock_contrast_sign": contrast_sign,
                "conditional_passed": conditional_passed,
                "passed": passed,
            }
        )
    same_sign = bool(
        len(contrast_signs) == len(periods)
        and all(sign != 0 for sign in contrast_signs)
        and len(set(contrast_signs)) == 1
    )
    repeated &= same_sign
    if manifest.get("technical_smoke_not_evidence"):
        status = "technical_smoke_not_evidence"
        reason = "The smoke run checks plumbing only."
    elif not complete:
        status = "parked_insufficient_common_support"
        reason = "The fixed profiles or quiet/shock cells lacked fair common support."
    elif repeated:
        status = "retained_context_dependent_volume_relationship"
        direction = "stronger_in_shock" if contrast_signs[0] > 0 else "stronger_in_quiet"
        reason = (
            "Current context beat the level-only and seven-day stale controls in both "
            "periods, while the level contribution changed consistently between quiet "
            f"and shock states ({direction}) and retained a measurable residual."
        )
    else:
        status = "parked_not_repeated_beyond_context_controls"
        reason = (
            "Context conditioning did not repeat across both periods beyond level-only, "
            "seven-day stale, breadth, uncertainty, and quiet/shock controls."
        )
    return DataFrame(
        [
            {
                "cell_id": cell["cell_id"],
                "surface_id": cell["surface_id"],
                "cohort": cell["cohort"],
                "context_block": cell["context_block"],
                "status": status,
                "reason": reason,
                "quiet_shock_same_sign_both_periods": same_sign,
                "period_checks": json.dumps(period_checks, sort_keys=True),
                "direction_prediction": False,
                "profit_optimization": False,
            }
        ]
    )


def score_cell(
    manifest: dict[str, Any],
    *,
    cell: dict[str, Any],
    record_dir: Path,
    artifact_dir: Path,
) -> tuple[dict[str, Any], DataFrame, DataFrame, DataFrame]:
    predictions, audit, thresholds = prediction_event_surface(
        manifest,
        cell=cell,
        record_dir=record_dir,
    )
    scores = score_scopes(cell, predictions)
    comparisons = paired_profile_comparisons(manifest, cell, predictions, scores)
    conditional = conditional_level_effects(manifest, cell, predictions)
    decision = retention_decision(manifest, cell, comparisons, conditional)
    prediction_path = artifact_dir / "common_event_predictions.parquet"
    atomic_write_parquet(predictions, prediction_path)
    atomic_write_parquet(scores, record_dir / "regression_scores.parquet")
    atomic_write_parquet(comparisons, record_dir / "paired_profile_comparisons.parquet")
    atomic_write_parquet(conditional, record_dir / "quiet_shock_level_effects.parquet")
    atomic_write_parquet(decision, record_dir / "cell_decision.parquet")
    atomic_write_parquet(thresholds, record_dir / "development_band_thresholds.parquet")
    atomic_write_json(audit, record_dir / "prediction_audit.json")
    result = {
        "cell_id": cell["cell_id"],
        "surface_id": cell["surface_id"],
        "context_block": cell["context_block"],
        "status": "completed",
        "decision": decision.iloc[0]["status"],
        "profiles": int(predictions["profile_id"].nunique()),
        "pairs": int(predictions["pair"].nunique()),
        "prediction_rows": len(predictions),
        "comparison_rows": len(comparisons),
        "conditional_rows": len(conditional),
        "prediction_path": str(prediction_path),
        "direction_prediction": False,
        "profit_optimization": False,
    }
    atomic_write_json(result, record_dir / "cell_result.json")
    return result, decision, comparisons, conditional


def score_run(
    manifest: dict[str, Any],
    *,
    record_dir: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    cell_results: list[dict[str, Any]] = []
    decisions: list[DataFrame] = []
    comparisons: list[DataFrame] = []
    conditional: list[DataFrame] = []
    for cell in manifest["cells"]:
        if cell["status"] != "prepared_for_outcome_models":
            decision = DataFrame(
                [
                    {
                        "cell_id": cell["cell_id"],
                        "surface_id": cell["surface_id"],
                        "cohort": cell["cohort"],
                        "context_block": cell["context_block"],
                        "status": "parked_post_control_support",
                        "reason": (
                            "Current plus seven-day-stale context did not retain the "
                            "outcome-blind 50-event/five-coin gate in both periods."
                        ),
                        "quiet_shock_same_sign_both_periods": False,
                        "period_checks": "[]",
                        "direction_prediction": False,
                        "profit_optimization": False,
                    }
                ]
            )
            decisions.append(decision)
            cell_results.append(
                {
                    "cell_id": cell["cell_id"],
                    "status": "parked_post_control_support",
                    "decision": "parked_post_control_support",
                }
            )
            continue
        compact_cell_dir = record_dir / "cells" / str(cell["cell_id"])
        artifact_cell_dir = artifact_dir / "cells" / str(cell["cell_id"])
        result, decision, cell_comparisons, cell_conditional = score_cell(
            manifest,
            cell=cell,
            record_dir=compact_cell_dir,
            artifact_dir=artifact_cell_dir,
        )
        cell_results.append(result)
        decisions.append(decision)
        comparisons.append(cell_comparisons)
        conditional.append(cell_conditional)
    combined_decisions = pd.concat(decisions, ignore_index=True)
    combined_comparisons = pd.concat(comparisons, ignore_index=True) if comparisons else DataFrame()
    combined_conditional = pd.concat(conditional, ignore_index=True) if conditional else DataFrame()
    decision_path = record_dir / "g5e_all_cell_decisions.parquet"
    comparison_path = record_dir / "g5e_all_profile_comparisons.parquet"
    conditional_path = record_dir / "g5e_all_quiet_shock_effects.parquet"
    atomic_write_parquet(combined_decisions, decision_path)
    atomic_write_parquet(combined_comparisons, comparison_path)
    atomic_write_parquet(combined_conditional, conditional_path)
    retained = combined_decisions.loc[
        combined_decisions["status"].eq("retained_context_dependent_volume_relationship")
    ]
    result = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "run_id": manifest["run_id"],
        "status": "completed",
        "completed_at_utc": utc_now(),
        "cells": len(combined_decisions),
        "completed_model_cells": int(
            sum(item.get("status") == "completed" for item in cell_results)
        ),
        "decision_counts": combined_decisions["status"].value_counts().to_dict(),
        "retained_cells": retained[
            ["cell_id", "surface_id", "cohort", "context_block", "reason"]
        ].to_dict(orient="records"),
        "cell_results": cell_results,
        "decisions": str(decision_path),
        "profile_comparisons": str(comparison_path),
        "quiet_shock_effects": str(conditional_path),
        "interpretation": (
            "Any retained cell describes repeatable context dependence of contact-volume "
            "reaction strength. It does not predict price direction or establish profit."
        ),
        "direction_prediction": False,
        "profit_optimization": False,
    }
    atomic_write_json(result, record_dir / "g5e_result.json")
    return result


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901
    parser = argparse.ArgumentParser(
        description="G5E timestamp-safe external-context FreqAI conditioning outcomes."
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--surfaces", default="all")
    parser.add_argument("--contexts", default="all")
    parser.add_argument("--profiles", default="all")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--profile-workers", type=int, default=4)
    parser.add_argument("--technical-smoke", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args(argv)
    if not 1 <= args.profile_workers <= MAX_WORKERS:
        raise ValueError(f"profile-workers must be in 1..{MAX_WORKERS}.")
    surfaces = tuple(SURFACES) if args.surfaces == "all" else parse_csv(args.surfaces)
    contexts = tuple(CONTEXTS) if args.contexts == "all" else parse_csv(args.contexts)
    profiles = tuple(PROFILES) if args.profiles == "all" else parse_csv(args.profiles)
    for label, selected, allowed in (
        ("surfaces", surfaces, SURFACES),
        ("contexts", contexts, CONTEXTS),
        ("profiles", profiles, PROFILES),
    ):
        unknown = sorted(set(selected).difference(allowed))
        if not selected or unknown:
            raise ValueError(f"Invalid G5E {label}: {unknown}")
    if not args.base_config.is_file():
        raise FileNotFoundError(args.base_config)
    manifest, manifest_path, artifact_dir = prepare_run(
        run_id=args.run_id,
        surfaces=surfaces,
        contexts=contexts,
        profiles=profiles,
        base_config=args.base_config,
        python_exe=args.python,
        profile_workers=args.profile_workers,
        technical_smoke=args.technical_smoke,
    )
    record_dir = manifest_path.parent
    launch = runtime_preflight(manifest, python_exe=args.python)
    atomic_write_json(launch, record_dir / "g5e_launch_preflight.json")
    manifest["launch_preflight"] = launch
    atomic_write_json(manifest, manifest_path)
    if not launch["passed"]:
        manifest["status"] = "preflight_failed"
        atomic_write_json(manifest, manifest_path)
        return 2
    if args.prepare_only:
        return 0
    if args.retry_failed:
        for item in manifest["commands"]:
            if item.get("status") == "failed":
                item["status"] = "pending"
        manifest.pop("failed_commands", None)
        manifest["status"] = "prepared"
        atomic_write_json(manifest, manifest_path)
    elif manifest.get("status") in {"failed", "preflight_failed"}:
        raise ValueError("Use --retry-failed after diagnosing a failed G5E profile.")
    if not args.score_only and manifest.get("status") != "completed":
        returncode = run_manifest(manifest, manifest_path)
        if returncode:
            return returncode
    incomplete = [
        (item["cell_id"], item["profile_id"])
        for item in manifest["commands"]
        if item.get("status") != "completed"
    ]
    if incomplete:
        raise ValueError(f"Cannot score incomplete G5E profiles: {incomplete}")
    result = score_run(manifest, record_dir=record_dir, artifact_dir=artifact_dir)
    manifest["status"] = "completed"
    manifest["finished_at_utc"] = utc_now()
    manifest["result"] = result
    atomic_write_json(manifest, manifest_path)
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
