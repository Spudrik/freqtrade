from __future__ import annotations

# Bound numerical libraries before importing pandas/pyarrow.
# ruff: noqa: E402, E501
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
import json
import sys
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation6_preflight as g6,
)


OUTPUT_ROOT = (
    REPO_ROOT / "user_data" / "research_news_data" / "context_features" / "market_reaction_zones"
)
FROZEN_BATCH = OUTPUT_ROOT / "generation7_review" / "g8_frozen_attribution_batch.json"
G7_SHARED_RECORD = OUTPUT_ROOT / "generation7_shared" / "g7_enriched_cache_20260821a"
G6_SOURCE_MANIFESTS = {
    "normal": OUTPUT_ROOT / "generation6_shared" / "g6_normal_source_manifest.json",
    "meme": OUTPUT_ROOT / "generation6_shared" / "g6_meme_source_manifest.json",
}
HISTORICAL_ORDERBOOK = (
    REPO_ROOT
    / "user_data"
    / "orderbook_data"
    / "historical_bybit"
    / "features"
    / "orderbook_trader_state_1h_bybit_linear.parquet"
)
RECORD_ROOT = OUTPUT_ROOT / "generation8_branches" / "g8_preflight"
DEFAULT_RUN_ID = "g8_outcome_blind_preflight_20260821a"
MAX_WORKERS = 4

DEVELOPMENT_PERIOD = {"normal": "development", "meme": "meme_development"}
VALIDATION_PERIODS = {
    "normal": ("validation_early", "validation_late"),
    "meme": ("meme_validation_early", "meme_validation_late"),
}
SMART_CONTRACT_PLATFORMS = frozenset(g6.GROUPS["smart_contract_platforms"])

RAW_PROVENANCE_COLUMNS = (
    "cohort",
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
    "attr_vp_value_area_width_pct",
    "attr_vp_state",
    "attr_tlv2_support_pivot_count_rank0",
    "attr_tlv2_support_absorbed_pivot_count_rank0",
    "attr_tlv2_forecast_support_pivot_count",
    "attr_tlv2_resistance_pivot_count_rank0",
    "attr_tlv2_resistance_absorbed_pivot_count_rank0",
    "attr_tlv2_forecast_resistance_pivot_count",
    "attr_generic_round_step",
    "mechanism_group",
    "level_side",
)

ORDERBOOK_REQUIRED_COLUMNS = (
    "date",
    "canonical_pair",
    "source_min_ts",
    "source_max_ts",
    "obts_feature_present",
    "obts_coverage_ratio",
    "obts_pressure_25bps_mean",
    "obts_pressure_25bps_last",
    "obts_pressure_25bps_slope",
    "obts_pressure_flip_count_1h",
    "obts_pressure_duration_hours",
    "obts_pressure_flip_strength",
    "obts_pressure_acceleration",
)

LEVEL_CACHE_REQUIRED_COLUMNS = (
    "source_open",
    "available_at",
    "source_open_price",
    "source_high",
    "source_low",
    "source_close",
    "source_volume",
    "source_atr_14",
)

PREDICTOR_GROUPS: dict[str, tuple[str, ...]] = {
    "level_summary": (
        "level__event_count",
        "level__distinct_family_count",
        "level__distinct_timeframe_count",
        "level__mean_zone_half_width_atr",
        "level__median_pre_distance_atr",
        "level__mean_contact_close_distance_atr",
    ),
    "participation_volume": (
        "ohlcv_volume_pressure__relative_volume",
        "ohlcv_volume_pressure__volume_acceleration_magnitude",
    ),
    "participation_pressure": (
        "ohlcv_volume_pressure__absolute_pressure",
        "ohlcv_volume_pressure__pressure_persistence",
    ),
    "absolute_volatility": (
        "ohlcv_volatility_range__atr_fraction",
        "ohlcv_volatility_range__prior_range_atr",
    ),
    "compression": (
        "ohlcv_volatility_range__bollinger_width",
        "ohlcv_volatility_range__range_contraction_ratio",
    ),
    "persistent_trend": (
        "ohlcv_trend_momentum__ema20_slope_magnitude",
        "ohlcv_trend_momentum__ma_separation",
        "ohlcv_trend_momentum__adx14",
    ),
    "return_acceleration": (
        "ohlcv_trend_momentum__return_slope_magnitude",
        "ohlcv_trend_momentum__return_acceleration_magnitude",
    ),
    "oscillator_displacement": (
        "ohlcv_trend_momentum__rsi_distance_from_50",
        "ohlcv_trend_momentum__rsi_change_magnitude",
        "ohlcv_trend_momentum__macd_histogram_magnitude",
        "ohlcv_trend_momentum__macd_change_magnitude",
    ),
    "btc_horizons": (
        "cross_market_btc__btc_activity_1h",
        "cross_market_btc__btc_activity_4h",
        "cross_market_btc__btc_activity_24h",
        "cross_market_btc__btc_relative_volume",
    ),
    "timeframe_relationships": tuple(
        f"timeframe_{timeframe}__{relationship}"
        for timeframe in ("4h", "8h", "1d")
        for relationship in (
            "isolated_native_1h",
            "isolated_higher_timeframe",
            "same_mechanism_agreement",
            "different_mechanism_agreement",
            "any_cross_timeframe_cluster",
            "opposing_side_overlap",
        )
    ),
    "orderbook_parent": (
        "orderbook_btc__btc_absolute_pressure",
        "orderbook_btc__btc_pressure_change_from_quiet_threshold",
        "orderbook_btc__btc_orderbook_coverage",
    ),
}


@dataclass(frozen=True)
class CellSpec:
    branch_id: str
    cohort: str
    surface: str
    parent_ready: str
    predictor_groups: tuple[str, ...]
    pair_scope: str = "all"

    @property
    def cell_id(self) -> str:
        suffix = "" if self.surface == "main" else f"__{self.surface}"
        return f"{self.branch_id}__{self.cohort}{suffix}"


def build_cells() -> tuple[CellSpec, ...]:
    cells: list[CellSpec] = []
    shared = (
        (
            "g8a_level_geometry_attribution_for_participation_and_volatility",
            "g7_common_b",
            (
                "level_summary",
                "participation_volume",
                "participation_pressure",
                "absolute_volatility",
                "compression",
            ),
        ),
        (
            "g8b_local_participation_attribution_under_volatility",
            "g7_common_b",
            (
                "level_summary",
                "participation_volume",
                "participation_pressure",
                "absolute_volatility",
                "compression",
            ),
        ),
        (
            "g8c_volatility_compression_attribution_under_participation",
            "g7_common_b",
            (
                "level_summary",
                "participation_volume",
                "participation_pressure",
                "absolute_volatility",
                "compression",
            ),
        ),
        (
            "g8d_trend_momentum_attribution_with_participation",
            "g7_common_c",
            (
                "level_summary",
                "participation_volume",
                "participation_pressure",
                "persistent_trend",
                "return_acceleration",
                "oscillator_displacement",
            ),
        ),
        (
            "g8e_btc_context_freshness_horizon_and_intensity",
            "g7_common_a",
            ("level_summary", "participation_volume", "participation_pressure", "btc_horizons"),
        ),
        (
            "g8f_cross_timeframe_incremental_value_and_room_geometry",
            "g7_common_d",
            (
                "level_summary",
                "participation_volume",
                "participation_pressure",
                "timeframe_relationships",
            ),
        ),
    )
    for branch_id, parent_ready, groups in shared:
        for cohort in ("normal", "meme"):
            cells.append(CellSpec(branch_id, cohort, "main", parent_ready, groups))
    cells.append(
        CellSpec(
            "g8g_orderbook_btc_attribution_for_smart_contract_platforms",
            "normal",
            "main",
            "g7_common_i",
            ("level_summary", "btc_horizons", "orderbook_parent"),
            "smart_contract_platforms",
        )
    )
    for surface, ready, groups in (
        (
            "participation_volatility",
            "g7_common_b",
            (
                "level_summary",
                "participation_volume",
                "participation_pressure",
                "absolute_volatility",
                "compression",
            ),
        ),
        (
            "participation_trend",
            "g7_common_c",
            (
                "level_summary",
                "participation_volume",
                "participation_pressure",
                "persistent_trend",
                "return_acceleration",
                "oscillator_displacement",
            ),
        ),
        (
            "participation_btc",
            "g7_common_a",
            ("level_summary", "participation_volume", "participation_pressure", "btc_horizons"),
        ),
    ):
        for cohort in ("normal", "meme"):
            cells.append(
                CellSpec(
                    "g8h_first_contact_occupancy_and_retest_state",
                    cohort,
                    surface,
                    ready,
                    groups,
                )
            )
    return tuple(cells)


CELLS = build_cells()


def load_frozen_batch() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if not FROZEN_BATCH.is_file():
        raise FileNotFoundError(FROZEN_BATCH)
    batch = json.loads(FROZEN_BATCH.read_text(encoding="utf-8"))
    if batch.get("status") != "frozen_before_generation8_attribution_outcomes":
        raise ValueError("Generation 8 was not frozen before attribution outcomes.")
    boundary = batch.get("research_boundary", {})
    if boundary.get("profit_optimization") is not False:
        raise ValueError("Profit optimization is outside Generation 8.")
    if boundary.get("direction_prediction") is not False:
        raise ValueError("Direction prediction is outside Generation 8.")
    branches = {str(item["id"]): item for item in batch.get("branches", [])}
    expected = {cell.branch_id for cell in CELLS}
    if set(branches) != expected:
        raise ValueError(
            f"Generation 8 frozen/runtime drift: missing={set(branches) - expected}, "
            f"extra={expected - set(branches)}"
        )
    return batch, branches


def load_g7_manifest(cohort: str) -> dict[str, Any]:
    path = G7_SHARED_RECORD / f"{cohort}_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "completed_generation7_enriched_shared_cache":
        raise ValueError(f"Generation 7 {cohort} cache is not terminal.")
    if manifest.get("cohort") != cohort:
        raise ValueError(f"Generation 7 cohort drift in {path}")
    if not manifest.get("outcome_blind_sequence", {}).get(
        "exact_support_decisions_written_before_target_read"
    ):
        raise ValueError(f"Generation 7 support was not outcome blind: {path}")
    return manifest


def selected_pairs(cell: CellSpec, manifest: dict[str, Any]) -> tuple[str, ...]:
    pairs = tuple(str(pair) for pair in manifest["pairs"])
    if cell.pair_scope == "all":
        return pairs
    if cell.pair_scope == "smart_contract_platforms":
        return tuple(pair for pair in pairs if pair in SMART_CONTRACT_PLATFORMS)
    raise ValueError(f"Unknown pair scope {cell.pair_scope!r}")


def required_predictor_columns(cell: CellSpec) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            column for group in cell.predictor_groups for column in PREDICTOR_GROUPS[group]
        )
    )


def inspect_pair_sources(
    *, cohort: str, item: dict[str, Any], level_cache_dir: Path
) -> dict[str, Any]:
    pair = str(item["pair"])
    feature_path = Path(item["feature_path"])
    support_path = Path(item["support_path"])
    raw_path = Path(item["source_path"])
    for path in (feature_path, support_path, raw_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    feature_schema = set(pq.ParquetFile(feature_path).schema.names)
    support_schema = set(pq.ParquetFile(support_path).schema.names)
    raw_schema = set(pq.ParquetFile(raw_path).schema.names)
    expected_predictors = {column for columns in PREDICTOR_GROUPS.values() for column in columns}
    missing_predictors = sorted(expected_predictors - feature_schema)
    missing_raw = sorted(set(RAW_PROVENANCE_COLUMNS) - raw_schema)
    ready_columns = {
        f"ready__{cell.parent_ready}"
        for cell in CELLS
        if cell.cohort == cohort
        and (
            cell.pair_scope == "all"
            or (cell.pair_scope == "smart_contract_platforms" and pair in SMART_CONTRACT_PLATFORMS)
        )
    }
    missing_ready = sorted(ready_columns - support_schema)
    level_cache_rows: list[dict[str, Any]] = []
    for timeframe in ("1h", "4h", "8h", "1d"):
        cache_path = g0.level_cache_path(
            pair,
            timeframe,
            ("core", "generic"),
            cache_dir=level_cache_dir,
        )
        if not cache_path.is_file():
            level_cache_rows.append(
                {"timeframe": timeframe, "path": str(cache_path), "status": "missing"}
            )
            continue
        schema = set(pq.ParquetFile(cache_path).schema.names)
        missing = sorted(set(LEVEL_CACHE_REQUIRED_COLUMNS) - schema)
        level_cache_rows.append(
            {
                "timeframe": timeframe,
                "path": str(cache_path),
                "rows": int(pq.ParquetFile(cache_path).metadata.num_rows),
                "missing_required_columns": missing,
                "status": "ready" if not missing else "missing_required_columns",
            }
        )
    problems = [
        *(f"missing predictor {column}" for column in missing_predictors),
        *(f"missing raw provenance {column}" for column in missing_raw),
        *(f"missing parent readiness {column}" for column in missing_ready),
        *(
            f"{row['timeframe']} level cache {row['status']}"
            for row in level_cache_rows
            if row["status"] != "ready"
        ),
    ]
    return {
        "cohort": cohort,
        "pair": pair,
        "feature_path": str(feature_path),
        "support_path": str(support_path),
        "raw_provenance_path": str(raw_path),
        "feature_rows": int(pq.ParquetFile(feature_path).metadata.num_rows),
        "raw_rows": int(pq.ParquetFile(raw_path).metadata.num_rows),
        "missing_predictors": missing_predictors,
        "missing_raw_provenance": missing_raw,
        "missing_parent_ready_columns": missing_ready,
        "level_caches": level_cache_rows,
        "status": "ready" if not problems else "blocked",
        "problems": problems,
        "reaction_outcome_columns_read": False,
    }


def inspect_orderbook() -> dict[str, Any]:
    if not HISTORICAL_ORDERBOOK.is_file():
        return {"status": "missing", "path": str(HISTORICAL_ORDERBOOK)}
    parquet = pq.ParquetFile(HISTORICAL_ORDERBOOK)
    schema = set(parquet.schema.names)
    missing = sorted(set(ORDERBOOK_REQUIRED_COLUMNS) - schema)
    columns = [
        column
        for column in (
            "date",
            "canonical_pair",
            "source_max_ts",
            "obts_feature_present",
            "obts_coverage_ratio",
        )
        if column in schema
    ]
    frame = pd.read_parquet(HISTORICAL_ORDERBOOK, columns=columns)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame["source_max_ts"] = pd.to_datetime(frame["source_max_ts"], utc=True, errors="coerce")
    safe = frame["source_max_ts"].notna() & frame["source_max_ts"].le(frame["date"])
    if "obts_feature_present" in frame:
        safe &= pd.to_numeric(frame["obts_feature_present"], errors="coerce").eq(1.0)
    first_date = frame["date"].min()
    last_date = frame["date"].max()
    return {
        "path": str(HISTORICAL_ORDERBOOK),
        "sha256": g0.sha256_file(HISTORICAL_ORDERBOOK),
        "rows": int(parquet.metadata.num_rows),
        "first_date": None if pd.isna(first_date) else first_date.isoformat(),
        "last_date": None if pd.isna(last_date) else last_date.isoformat(),
        "causal_source_rows": int(safe.sum()),
        "source_after_feature_hour_violations": int(
            (
                frame["source_max_ts"].notna()
                & frame["date"].notna()
                & frame["source_max_ts"].gt(frame["date"])
            ).sum()
        ),
        "missing_required_columns": missing,
        "status": "ready" if not missing and safe.any() else "blocked",
        "venue_scope": "Bybit BTCUSDT linear; BTC pair-local and BTC-wide context only",
        "reaction_outcome_columns_read": False,
    }


def finite_complete(frame: DataFrame, columns: Sequence[str]) -> pd.Series:
    numeric = frame[list(columns)].apply(pd.to_numeric, errors="coerce")
    return numeric.replace([np.inf, -np.inf], np.nan).notna().all(axis=1)


def load_cell_pair_frame(cell: CellSpec, item: dict[str, Any]) -> DataFrame:
    columns = required_predictor_columns(cell)
    feature = pd.read_parquet(item["feature_path"], columns=["date", *columns])
    support_column = f"ready__{cell.parent_ready}"
    support = pd.read_parquet(item["support_path"], columns=["date", "period", support_column])
    feature["date"] = pd.to_datetime(feature["date"], utc=True, errors="coerce")
    support["date"] = pd.to_datetime(support["date"], utc=True, errors="coerce")
    if feature["date"].duplicated().any() or support["date"].duplicated().any():
        raise ValueError(f"Duplicate G7 parent dates for {item['pair']}")
    frame = support.merge(feature, on="date", how="inner", validate="one_to_one")
    ready = frame[support_column].fillna(False).astype(bool)
    complete = finite_complete(frame, columns)
    frame = frame.loc[ready & complete].copy()
    frame.insert(0, "pair", str(item["pair"]))
    return frame


def predictor_bin_rows(cell: CellSpec, frame: DataFrame) -> list[dict[str, Any]]:
    development = frame.loc[frame["period"].eq(DEVELOPMENT_PERIOD[cell.cohort])]
    rows: list[dict[str, Any]] = []
    for column in required_predictor_columns(cell):
        values = pd.to_numeric(development[column], errors="coerce")
        values = values.loc[np.isfinite(values)]
        unique = int(values.nunique())
        if len(values) < 100 or unique < 5:
            rows.append(
                {
                    "cell_id": cell.cell_id,
                    "branch_id": cell.branch_id,
                    "cohort": cell.cohort,
                    "surface": cell.surface,
                    "predictor": column,
                    "development_rows": len(values),
                    "unique_values": unique,
                    "lower_boundary": None,
                    "upper_boundary": None,
                    "status": "not_binned_discrete_or_insufficient_variation",
                    "reaction_outcomes_opened": False,
                }
            )
            continue
        lower, upper = values.quantile([1.0 / 3.0, 2.0 / 3.0]).tolist()
        stable = bool(np.isfinite(lower) and np.isfinite(upper) and lower < upper)
        rows.append(
            {
                "cell_id": cell.cell_id,
                "branch_id": cell.branch_id,
                "cohort": cell.cohort,
                "surface": cell.surface,
                "predictor": column,
                "development_rows": len(values),
                "unique_values": unique,
                "lower_boundary": float(lower) if stable else None,
                "upper_boundary": float(upper) if stable else None,
                "status": "frozen_adjacent_tertiles" if stable else "not_binned_tied_boundaries",
                "reaction_outcomes_opened": False,
            }
        )
    return rows


def cell_support_record(cell: CellSpec, frame: DataFrame) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    periods = (DEVELOPMENT_PERIOD[cell.cohort], *VALIDATION_PERIODS[cell.cohort])
    for period in periods:
        selected = frame.loc[frame["period"].eq(period)]
        counts = selected.groupby("pair", observed=True).size()
        checks.append(
            {
                "period": period,
                "events": len(selected),
                "coins": int(counts.gt(0).sum()),
                "minimum_pair_events": int(counts.min()) if len(counts) else 0,
            }
        )
    development, *validation = checks
    minimum_coins = 5
    supported = (
        development["events"] >= 100
        and development["coins"] >= minimum_coins
        and all(check["events"] >= 50 and check["coins"] >= minimum_coins for check in validation)
    )
    return {
        "cell_id": cell.cell_id,
        "branch_id": cell.branch_id,
        "cohort": cell.cohort,
        "surface": cell.surface,
        "parent_ready_block": cell.parent_ready,
        "pair_scope": cell.pair_scope,
        "required_predictor_groups": list(cell.predictor_groups),
        "period_checks": checks,
        "status": (
            "supported_for_generation8_outcome_blind_feature_construction"
            if supported
            else "parked_outcome_blind_insufficient_exact_parent_support"
        ),
        "reaction_outcomes_opened": False,
    }


def representation_contract() -> dict[str, Any]:
    return {
        "lineage_identity": {
            "explicit_identity": "Use finite indicator-supplied level_identity when available.",
            "deterministic_fallback": (
                "Otherwise follow the source timeframe, level family, level name, and "
                "representation as one calculated series; begin a new lineage segment when "
                "the available level moves by more than its active zone half-width or returns "
                "after an unavailable gap."
            ),
            "reason": (
                "This mirrors the existing causal episode boundary and avoids either treating "
                "every small floating-point change as a new level or joining unrelated nearby "
                "levels with an outcome-selected tolerance."
            ),
        },
        "touch_state": {
            "first_recorded_contact": "No earlier causal contact for the lineage in the observed source history.",
            "continued_occupancy": "The same lineage was in contact on the immediately preceding complete 1h candle.",
            "first_retest": "One earlier independent contact episode exists and the preceding candle was not in contact.",
            "later_retest": "At least two earlier independent contact episodes exist and the preceding candle was not in contact.",
            "episode_separation": "Use the existing six-complete-candle contact cooldown; do not search retest numbers.",
            "left_censoring": "First means first recorded after available source history begins, not first ever in the market.",
        },
        "age_and_persistence": {
            "level_age_hours": "event_time minus source_available_at, clipped only at zero for valid causal rows",
            "persistence_hours": "continuous hours for which the deterministic lineage remains available before contact",
            "prior_touch_count": "number of prior independent contact episodes for the lineage",
            "time_since_prior_contact_hours": "elapsed complete hours since the previous recorded contact",
        },
        "duration_features": {
            "construction": (
                "Run lengths are calculated on the continuous causal 1h predictor timeline, "
                "using only values known before each event hour."
            ),
            "thresholds": (
                "Quiet/ordinary/elevated or compressed/ordinary/expanded boundaries are the "
                "development-only adjacent tertiles written by this preflight."
            ),
        },
        "room_geometry": {
            "source": "All available calculated levels in the frozen 1h, 4h, 8h, and 1d level caches.",
            "merge": "Backward as-of on available_at; a source row after the event hour is forbidden.",
            "measures": [
                "nearest non-overlapping level gap above in causal 1h ATR units",
                "nearest non-overlapping level gap below in causal 1h ATR units",
                "smaller direction-neutral open-room distance",
                "level counts within 0.5, 1.0, and 2.0 ATR",
                "same- and opposing-side overlap fractions",
            ],
            "boundary": "Room is context around a reaction point; it is not a future direction label.",
        },
        "prior_reaction_history": {
            "allowed_only_after_support_freeze": True,
            "causal_lag": "Only prior episodes whose full four-hour reaction window ended before the current event may contribute.",
            "measures": "Prior absolute volume/range/excursion magnitudes and observation count; never prior signed direction or profit.",
        },
        "orderbook": {
            "venue": "Bybit BTCUSDT linear",
            "scope": "BTC pair-local for BTC; BTC-wide context for altcoins, never pair-local altcoin order books",
            "freshness": "event hour minus source_max_ts, with source_max_ts not after the event hour",
            "direction_neutral": (
                "Use absolute pressure, absolute slope/acceleration, flip count, duration, "
                "coverage, and frozen pressure-intensity state."
            ),
        },
        "outcome_boundary": {
            "profit": False,
            "future_signed_direction": False,
            "entry_exit_construction": False,
            "reaction_targets_read_by_preflight": False,
        },
    }


def run_preflight(*, run_id: str, workers: int) -> dict[str, Any]:
    _frozen, frozen_branches = load_frozen_batch()
    workers = max(1, min(int(workers), MAX_WORKERS))
    run_dir = RECORD_ROOT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    manifests = {cohort: load_g7_manifest(cohort) for cohort in ("normal", "meme")}
    source_manifests = {
        cohort: json.loads(path.read_text(encoding="utf-8"))
        for cohort, path in G6_SOURCE_MANIFESTS.items()
    }
    audits: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for cohort, manifest in manifests.items():
            cache_dir = Path(source_manifests[cohort]["storage"]["cache_dir"])
            for item in manifest["inventory"]:
                future = pool.submit(
                    inspect_pair_sources,
                    cohort=cohort,
                    item=item,
                    level_cache_dir=cache_dir,
                )
                futures[future] = (cohort, str(item["pair"]))
        for future in as_completed(futures):
            audit = future.result()
            audits.append(audit)
            print(
                json.dumps(
                    {
                        "phase": "g8_outcome_blind_source_audit",
                        "cohort": audit["cohort"],
                        "pair": audit["pair"],
                        "status": audit["status"],
                    }
                ),
                flush=True,
            )
    audits.sort(key=lambda row: (row["cohort"], row["pair"]))
    orderbook = inspect_orderbook()
    support_records: list[dict[str, Any]] = []
    bin_records: list[dict[str, Any]] = []
    for cell in CELLS:
        manifest = manifests[cell.cohort]
        pair_set = set(selected_pairs(cell, manifest))
        item_by_pair = {str(item["pair"]): item for item in manifest["inventory"]}
        frames = [
            load_cell_pair_frame(cell, item_by_pair[pair])
            for pair in manifest["pairs"]
            if pair in pair_set
        ]
        frame = pd.concat(frames, ignore_index=True) if frames else DataFrame()
        support = cell_support_record(cell, frame)
        support_records.append(support)
        bin_records.extend(predictor_bin_rows(cell, frame))
        print(
            json.dumps(
                {
                    "phase": "g8_outcome_blind_exact_parent_support",
                    "cell_id": cell.cell_id,
                    "status": support["status"],
                }
            ),
            flush=True,
        )
    source_audit_path = run_dir / "g8_source_audit.json"
    support_path = run_dir / "g8_exact_parent_support.json"
    bins_path = run_dir / "g8_frozen_adjacent_predictor_ranges.csv"
    contract_path = run_dir / "g8_representation_contract.json"
    g0.atomic_write_json(audits, source_audit_path)
    g0.atomic_write_json(support_records, support_path)
    g0.atomic_write_csv(DataFrame.from_records(bin_records), bins_path)
    g0.atomic_write_json(representation_contract(), contract_path)
    all_sources_ready = all(row["status"] == "ready" for row in audits)
    all_cells_supported = all(row["status"].startswith("supported") for row in support_records)
    branch_status: list[dict[str, Any]] = []
    for branch_id, branch in frozen_branches.items():
        rows = [row for row in support_records if row["branch_id"] == branch_id]
        branch_status.append(
            {
                "branch_id": branch_id,
                "plain_name": branch["plain_name"],
                "cells": len(rows),
                "supported_cells": sum(row["status"].startswith("supported") for row in rows),
                "status": (
                    "supported_for_generation8_feature_construction"
                    if rows and all(row["status"].startswith("supported") for row in rows)
                    else "partly_or_fully_parked_outcome_blind"
                ),
            }
        )
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "status": (
            "completed_all_generation8_siblings_supported_outcome_blind"
            if all_sources_ready and all_cells_supported and orderbook["status"] == "ready"
            else "completed_with_outcome_blind_parks_or_blockers"
        ),
        "created_at_utc": g0.utc_now(),
        "generation": 8,
        "frozen_batch": str(FROZEN_BATCH.resolve()),
        "frozen_batch_sha256": g0.sha256_file(FROZEN_BATCH),
        "parent_manifests": {
            cohort: str((G7_SHARED_RECORD / f"{cohort}_manifest.json").resolve())
            for cohort in manifests
        },
        "cells": len(CELLS),
        "supported_cells": sum(row["status"].startswith("supported") for row in support_records),
        "parked_cells": sum(not row["status"].startswith("supported") for row in support_records),
        "branches": branch_status,
        "source_pairs": len(audits),
        "source_pairs_ready": sum(row["status"] == "ready" for row in audits),
        "historical_orderbook": orderbook,
        "artifacts": {
            "source_audit": str(source_audit_path.resolve()),
            "source_audit_sha256": g0.sha256_file(source_audit_path),
            "exact_parent_support": str(support_path.resolve()),
            "exact_parent_support_sha256": g0.sha256_file(support_path),
            "frozen_adjacent_predictor_ranges": str(bins_path.resolve()),
            "frozen_adjacent_predictor_ranges_sha256": g0.sha256_file(bins_path),
            "representation_contract": str(contract_path.resolve()),
            "representation_contract_sha256": g0.sha256_file(contract_path),
        },
        "compute": {"workers": workers, "maximum_workers": MAX_WORKERS},
        "outcome_blind_sequence": {
            "reaction_outcome_columns_read": False,
            "profit_columns_read": False,
            "future_signed_direction_columns_read": False,
            "adjacent_ranges_frozen_from_development_predictors_only": True,
            "representation_contract_written_before_generation8_target_materialization": True,
        },
        "next_step": (
            "Build the Generation 8 causal feature/support cache for every supported cell, "
            "write exact support decisions, and only then materialize reaction targets."
        ),
    }
    manifest_path = run_dir / "manifest.json"
    g0.atomic_write_json(manifest, manifest_path)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Outcome-blind Generation 8 representation and support preflight."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    manifest = run_preflight(run_id=args.run_id, workers=args.workers)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
