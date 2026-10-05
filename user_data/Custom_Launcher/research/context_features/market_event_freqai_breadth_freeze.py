"""Freeze a broad, outcome-blind event/context FreqAI experiment batch."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_cross_asset_relevance_direct as cross_asset,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_esma_sovereign_rating_catalogue as esma,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_hierarchy_cpi_direct as cpi,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_hierarchy_layer2_direct as layer2,
)
from user_data.Custom_Launcher.research.context_features import (
    market_event_layer2_breadth_freeze as breadth,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation23_freqai_cache as g23_cache,
)


ANALYSIS_PATH = Path(__file__).resolve()
OUTPUT_ROOT = (
    REPO_ROOT
    / "user_data"
    / "research_news_data"
    / "context_features"
    / "event_hierarchy"
    / "event_freqai_breadth_20260908a"
)
EVENTS_PATH = OUTPUT_ROOT / "event_freqai_event_catalog.csv"
SAMPLES_PATH = OUTPUT_ROOT / "event_freqai_sample_catalog.csv"
REGISTRY_PATH = OUTPUT_ROOT / "event_freqai_profile_registry.json"
FREEZE_PATH = OUTPUT_ROOT / "event_freqai_breadth_freeze.json"

SOURCE_START_UTC = pd.Timestamp("2021-06-01T00:00:00Z")
SOURCE_STOP_UTC = pd.Timestamp("2026-01-01T00:00:00Z")
PARTITIONS = ("development_2021_2023", "internal_validation_2024_2025")
HORIZONS = (1, 4, 8)
MAX_CONTROLS_PER_EVENT = 4
CONTROL_SEARCH_WEEKS = 52
CONTROL_EVENT_EXCLUSION_HOURS = 8
EVENT_EPISODE_GAP_HOURS = 8
READY_BLOCK = "event_freqai_breadth_samples"
NORMAL_PAIRS = (
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "BNB/USDT:USDT",
    "ADA/USDT:USDT",
    "TRX/USDT:USDT",
)


FAMILY_SPECS: dict[str, dict[str, Any]] = {
    "fomc_policy_decision": {
        "kind": "scheduled_us_policy",
        "exact_clock": True,
        "scheduled": True,
    },
    "gdelt_unexpected_activity": {
        "kind": "unexpected_media_activity",
        "exact_clock": False,
        "scheduled": False,
    },
    "us_cpi": {
        "kind": "scheduled_us_macro",
        "exact_clock": True,
        "scheduled": True,
    },
    "uk_cpi": {
        "kind": "scheduled_uk_macro",
        "exact_clock": True,
        "scheduled": True,
    },
    "japan_tankan": {
        "kind": "scheduled_asia_macro",
        "exact_clock": True,
        "scheduled": True,
    },
    "esma_sovereign_rating": {
        "kind": "european_finance_news",
        "exact_clock": True,
        "scheduled": False,
    },
    "sec_hyperscaler_earnings": {
        "kind": "corporate_technology_news",
        "exact_clock": False,
        "scheduled": False,
    },
    **{
        f"cross_{family}": {
            "kind": "daily_cross_market_state",
            "exact_clock": False,
            "scheduled": False,
        }
        for family in (
            "market_fear",
            "technology_equities",
            "broad_us_dollar",
            "ten_year_yield",
            "yield_curve",
            "financial_conditions",
            "crude_oil",
        )
    },
}
FAMILIES = tuple(FAMILY_SPECS)
KINDS = tuple(dict.fromkeys(str(spec["kind"]) for spec in FAMILY_SPECS.values()))

RECENT_FEATURES = (
    "recent__pair_return_4h",
    "recent__pair_return_24h",
    "recent__relative_volume",
    "recent__range_over_atr_4h",
    "recent__absolute_return_over_atr_4h",
    "recent__atr_fraction",
)
BACKGROUND_FEATURES = (
    "background__pair_return_168h",
    "background__pair_return_720h",
    "background__realised_volatility_168h",
    "background__realised_volatility_720h",
    "background__range_position_720h",
)
CROSS_MARKET_FEATURES = (
    "cross_market__btc_return_4h",
    "cross_market__btc_return_24h",
    "cross_market__eth_return_4h",
    "cross_market__eth_return_24h",
    "cross_market__equal_weight_return_4h",
    "cross_market__equal_weight_return_24h",
    "cross_market__positive_breadth_4h",
    "cross_market__dispersion_4h",
)
EVENT_IDENTITY_FEATURES = (
    "event_identity__is_actual_event",
    "event_identity__parent_exact_clock_fraction",
    "event_identity__parent_scheduled_fraction",
    *(f"event_identity__kind_{kind}" for kind in KINDS),
    *(f"event_identity__family_{family}" for family in FAMILIES),
)
SIGNED_SOURCE_FEATURES = (
    "signed_source__available_count",
    "signed_source__primary_mean",
    "signed_source__secondary_mean",
    "signed_source__crypto_relation_available_count",
    "signed_source__crypto_relation_mean",
)
EVENT_CONFLUENCE_FEATURES = (
    "event_confluence__current_event_count",
    "event_confluence__current_family_count",
    "event_confluence__current_kind_count",
    "event_confluence__events_known_within_24h",
    "event_confluence__families_known_within_24h",
    "event_confluence__positive_signed_known_within_24h",
    "event_confluence__negative_signed_known_within_24h",
    "event_confluence__signed_balance_known_within_24h",
)
SINGLE_LEVEL_FEATURES = (
    "single_level__present",
    "single_level__nearest_distance_atr",
    "single_level__source_role_encoding",
    "single_level__source_timeframe_encoding",
    "single_level__pre_crossing_count_4h",
)
CLUSTER_FEATURES = (
    "cluster__present",
    "cluster__independent_family_count",
    "cluster__nearest_distance_atr",
    "cluster__pre_crossing_count_4h",
)
ALL_FEATURES = tuple(
    dict.fromkeys(
        (
            *RECENT_FEATURES,
            *BACKGROUND_FEATURES,
            *CROSS_MARKET_FEATURES,
            *EVENT_IDENTITY_FEATURES,
            *SIGNED_SOURCE_FEATURES,
            *EVENT_CONFLUENCE_FEATURES,
            *SINGLE_LEVEL_FEATURES,
            *CLUSTER_FEATURES,
        )
    )
)

TARGET_METADATA = tuple(
    {
        "target": f"&-meb_{metric}_h{horizon}",
        "metric": metric,
        "horizon_hours": horizon,
        "plain_meaning": meaning,
    }
    for metric, meaning in (
        ("log_range_atr", "Future high-to-low range relative to known prior ATR."),
        ("log_volume_ratio", "Future volume relative to known prior typical volume."),
        ("close_return_atr", "Signed future close movement relative to known prior ATR."),
        (
            "excursion_balance_atr",
            "Upside excursion minus downside excursion relative to known prior ATR.",
        ),
    )
    for horizon in HORIZONS
)
TARGETS = tuple(str(item["target"]) for item in TARGET_METADATA)


def _profile(*blocks: tuple[str, ...]) -> list[str]:
    return list(dict.fromkeys(column for block in blocks for column in block))


PROFILE_DEFINITIONS: dict[str, dict[str, Any]] = {
    "recent_market_only": {"blocks": ["recent_market"], "features": list(RECENT_FEATURES)},
    "slow_background_only": {
        "blocks": ["slow_background"],
        "features": list(BACKGROUND_FEATURES),
    },
    "cross_market_only": {
        "blocks": ["cross_market"],
        "features": list(CROSS_MARKET_FEATURES),
    },
    "event_identity_only": {
        "blocks": ["event_identity"],
        "features": list(EVENT_IDENTITY_FEATURES),
    },
    "signed_source_only": {
        "blocks": ["signed_source"],
        "features": list(SIGNED_SOURCE_FEATURES),
    },
    "event_confluence_only": {
        "blocks": ["event_confluence"],
        "features": list(EVENT_CONFLUENCE_FEATURES),
    },
    "single_level_only": {
        "blocks": ["single_level"],
        "features": list(SINGLE_LEVEL_FEATURES),
    },
    "cluster_only": {"blocks": ["cluster"], "features": list(CLUSTER_FEATURES)},
    "event_plus_recent_market": {
        "blocks": ["event_identity", "recent_market"],
        "features": _profile(EVENT_IDENTITY_FEATURES, RECENT_FEATURES),
    },
    "event_plus_background": {
        "blocks": ["event_identity", "slow_background"],
        "features": _profile(EVENT_IDENTITY_FEATURES, BACKGROUND_FEATURES),
    },
    "event_plus_cross_market": {
        "blocks": ["event_identity", "cross_market"],
        "features": _profile(EVENT_IDENTITY_FEATURES, CROSS_MARKET_FEATURES),
    },
    "event_plus_signed_source": {
        "blocks": ["event_identity", "signed_source"],
        "features": _profile(EVENT_IDENTITY_FEATURES, SIGNED_SOURCE_FEATURES),
    },
    "event_plus_single_level": {
        "blocks": ["event_identity", "single_level"],
        "features": _profile(EVENT_IDENTITY_FEATURES, SINGLE_LEVEL_FEATURES),
    },
    "event_plus_cluster": {
        "blocks": ["event_identity", "cluster"],
        "features": _profile(EVENT_IDENTITY_FEATURES, CLUSTER_FEATURES),
    },
}
COMBINATION_COMPONENTS = {
    "event_plus_recent_market": ("event_identity_only", "recent_market_only"),
    "event_plus_background": ("event_identity_only", "slow_background_only"),
    "event_plus_cross_market": ("event_identity_only", "cross_market_only"),
    "event_plus_signed_source": ("event_identity_only", "signed_source_only"),
    "event_plus_single_level": ("event_identity_only", "single_level_only"),
    "event_plus_cluster": ("event_identity_only", "cluster_only"),
}


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def decision_hour(value: Any) -> pd.Timestamp:
    """Return the first hourly candle whose opening is not before the decision time."""

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.ceil("h")


def model_period(value: Any) -> str:
    timestamp = pd.Timestamp(value)
    year = timestamp.year
    if year <= 2023:
        return "development_2021_2023"
    if year == 2024:
        return "walk_forward_validation_2024"
    if year == 2025:
        return "walk_forward_validation_2025"
    return "outside_scored_period"


def _numeric(value: Any, default: float = np.nan) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else float(default)


def _event_record(
    *,
    raw_event_id: str,
    source_key: str,
    family: str,
    decision_utc: Any,
    partition: str,
    timestamp_quality: str,
    source_sign_primary: float = np.nan,
    source_sign_secondary: float = np.nan,
    crypto_relation_sign: float = np.nan,
    signed_semantics: str = "unavailable_not_imputed",
) -> dict[str, Any]:
    spec = FAMILY_SPECS[family]
    decision = pd.Timestamp(decision_utc)
    if decision.tzinfo is None:
        decision = decision.tz_localize("UTC")
    else:
        decision = decision.tz_convert("UTC")
    return {
        "event_id": f"{family}__{raw_event_id}",
        "raw_event_id": raw_event_id,
        "source_key": source_key,
        "event_family": family,
        "event_kind": spec["kind"],
        "decision_utc": decision,
        "model_anchor_utc": decision_hour(decision),
        "source_partition": partition,
        "model_period": model_period(decision),
        "timestamp_quality": timestamp_quality,
        "exact_clock": bool(spec["exact_clock"]),
        "scheduled": bool(spec["scheduled"]),
        "source_sign_primary": source_sign_primary,
        "source_sign_secondary": source_sign_secondary,
        "crypto_relation_sign": crypto_relation_sign,
        "source_sign_available": bool(np.isfinite(source_sign_primary)),
        "crypto_relation_sign_available": bool(np.isfinite(crypto_relation_sign)),
        "signed_semantics": signed_semantics,
    }


def _esma_cluster_signs() -> dict[str, tuple[float, float]]:
    rating = pd.read_csv(esma.CATALOG_PATH)
    rating = rating.loc[
        rating["whole_event_partition"].isin(PARTITIONS)
        & rating["jurisdiction_group"].eq("selected_euro_area")
        & ~rating["source_action_sign"].eq("unsigned_or_lifecycle_action")
    ].copy()
    clustered = breadth.fixed_window_clusters(
        rating,
        anchor_column="available_at_utc",
        prefix="esma_euro_signed_window",
    )
    sign_map = {
        "positive_rating_action": 1.0,
        "negative_rating_action": -1.0,
        "mixed_rating_actions": 0.0,
        "unsigned_or_lifecycle_action": 0.0,
    }
    output: dict[str, tuple[float, float]] = {}
    for group_id, group in clustered.groupby("market_window_group_id", sort=False):
        values = group["source_action_sign"].map(sign_map).astype(float)
        output[str(group_id)] = (float(values.mean()), float(values.abs().sum()))
    return output


def build_event_catalog() -> DataFrame:
    records: list[dict[str, Any]] = []

    core = pd.read_csv(layer2.EVENTS_PATH)
    source_to_family = {
        "official_fomc": "fomc_policy_decision",
        "gdelt_activity_spike": "gdelt_unexpected_activity",
    }
    for row in core.loc[core["whole_event_partition"].isin(PARTITIONS)].itertuples(
        index=False
    ):
        family = source_to_family[str(row.event_source)]
        records.append(
            _event_record(
                raw_event_id=str(row.event_id),
                source_key=str(row.event_source),
                family=family,
                decision_utc=row.anchor_utc,
                partition=str(row.whole_event_partition),
                timestamp_quality=(
                    "official_release_clock"
                    if row.event_source == "official_fomc"
                    else "hourly_media_activity_clock"
                ),
            )
        )

    cpi_catalog = pd.read_csv(cpi.CATALOG_PATH)
    for row in cpi_catalog.loc[
        cpi_catalog["whole_event_partition"].isin(PARTITIONS)
    ].itertuples(index=False):
        primary = _numeric(row.temperature_core_mom)
        secondary = _numeric(row.temperature_headline_mom)
        relation = -primary if np.isfinite(primary) else np.nan
        records.append(
            _event_record(
                raw_event_id=str(row.event_id),
                source_key="us_cpi",
                family="us_cpi",
                decision_utc=row.anchor_utc,
                partition=str(row.whole_event_partition),
                timestamp_quality="official_bls_release_clock",
                source_sign_primary=primary,
                source_sign_secondary=secondary,
                crypto_relation_sign=relation,
                signed_semantics=(
                    "Hotter/cooler than the previous release, not hotter/cooler than "
                    "the historical market expectation."
                ),
            )
        )

    esma_signs = _esma_cluster_signs()
    broad = pd.read_csv(breadth.EVENTS_PATH)
    wanted_sources = {
        "uk_ons_cpi": "uk_cpi",
        "japan_tankan": "japan_tankan",
        "esma_euro_signed_rating": "esma_sovereign_rating",
        "sec_hyperscaler_earnings": "sec_hyperscaler_earnings",
    }
    broad = broad.loc[
        broad["whole_event_partition"].isin(PARTITIONS)
        & broad["event_source"].isin(wanted_sources)
    ]
    for row in broad.itertuples(index=False):
        source = str(row.event_source)
        primary = np.nan
        secondary = np.nan
        semantics = "unavailable_not_imputed"
        if source == "esma_euro_signed_rating":
            primary, secondary = esma_signs.get(str(row.base_event_id), (np.nan, np.nan))
            semantics = "Official rating-action sign; no crypto direction is assumed."
        records.append(
            _event_record(
                raw_event_id=str(row.event_id),
                source_key=source,
                family=wanted_sources[source],
                decision_utc=row.decision_utc,
                partition=str(row.whole_event_partition),
                timestamp_quality=str(row.timestamp_quality),
                source_sign_primary=primary,
                source_sign_secondary=secondary,
                signed_semantics=semantics,
            )
        )

    cross = pd.read_csv(cross_asset.CATALOG_PATH)
    cross = cross.loc[cross["whole_event_partition"].isin(PARTITIONS)]
    relation_multiplier = {
        "same_direction_later_test_only": 1.0,
        "opposite_direction_later_test_only": -1.0,
    }
    for row in cross.itertuples(index=False):
        raw = _numeric(row.source_change)
        threshold = abs(_numeric(row.shock_threshold))
        normalized = raw / threshold if np.isfinite(threshold) and threshold > 0 else np.nan
        normalized = float(np.clip(normalized, -10.0, 10.0))
        multiplier = relation_multiplier.get(str(row.expected_crypto_relation))
        relation = normalized * multiplier if multiplier is not None else np.nan
        records.append(
            _event_record(
                raw_event_id=str(row.event_id),
                source_key=f"cross_asset_{row.event_family}",
                family=f"cross_{row.event_family}",
                decision_utc=row.anchor_utc,
                partition=str(row.whole_event_partition),
                timestamp_quality="conservative_next_day_source_availability",
                source_sign_primary=normalized,
                crypto_relation_sign=relation,
                signed_semantics=(
                    "Causal source change divided by its past-only shock threshold; "
                    "any crypto relation remains a hypothesis."
                ),
            )
        )

    events = DataFrame.from_records(records)
    events = events.loc[
        events["model_anchor_utc"].ge(SOURCE_START_UTC)
        & events["model_anchor_utc"].lt(SOURCE_STOP_UTC)
    ].copy()
    if events.empty or events["event_id"].duplicated().any():
        raise ValueError("Unified event catalogue is empty or has duplicate identifiers.")
    if set(events["event_family"]).difference(FAMILIES):
        raise ValueError("Unified event catalogue contains an undeclared event family.")
    return assign_event_episodes(events).sort_values(
        ["model_anchor_utc", "event_family", "event_id"], kind="stable"
    ).reset_index(drop=True)


def assign_event_episodes(events: DataFrame) -> DataFrame:
    output = events.copy()
    anchors = sorted(pd.to_datetime(output["model_anchor_utc"], utc=True).unique())
    mapping: dict[pd.Timestamp, str] = {}
    episode = 0
    previous: pd.Timestamp | None = None
    for raw_anchor in anchors:
        anchor = pd.Timestamp(raw_anchor)
        if previous is None or anchor - previous > pd.Timedelta(hours=EVENT_EPISODE_GAP_HOURS):
            episode += 1
        mapping[anchor] = f"event_episode_{episode:04d}"
        previous = anchor
    output["event_episode_id"] = pd.to_datetime(
        output["model_anchor_utc"], utc=True
    ).map(mapping)
    return output


def _candidate_controls_from_maps(events: DataFrame) -> DataFrame:
    event_lookup = events.set_index(["source_key", "raw_event_id"])[
        ["event_id", "event_episode_id"]
    ].to_dict("index")
    records: list[dict[str, Any]] = []
    for source_path, source_column in (
        (layer2.CONTROLS_PATH, "event_source"),
        (breadth.CONTROLS_PATH, "event_source"),
    ):
        frame = pd.read_csv(source_path)
        if source_path == layer2.CONTROLS_PATH:
            frame = frame.loc[frame["control_type"].eq("matched_prior_state")]
        for row in frame.itertuples(index=False):
            source = str(getattr(row, source_column))
            key = (source, str(row.event_id))
            parent = event_lookup.get(key)
            rank = int(row.control_rank)
            if parent is None or rank > MAX_CONTROLS_PER_EVENT:
                continue
            records.append(
                {
                    "parent_event_id": parent["event_id"],
                    "parent_episode_id": parent["event_episode_id"],
                    "control_anchor_utc": decision_hour(row.control_anchor_utc),
                    "control_rank": rank,
                    "control_method": str(row.control_type),
                }
            )
    return DataFrame.from_records(records)


def _generated_prior_week_controls(events: DataFrame) -> DataFrame:
    records: list[dict[str, Any]] = []
    generated_families = {
        "us_cpi",
        *(family for family in FAMILIES if family.startswith("cross_")),
    }
    all_anchors = list(pd.to_datetime(events["model_anchor_utc"], utc=True))
    for event in events.loc[events["event_family"].isin(generated_families)].itertuples(
        index=False
    ):
        found = 0
        anchor = pd.Timestamp(event.model_anchor_utc)
        for weeks in range(1, CONTROL_SEARCH_WEEKS + 1):
            candidate = anchor - pd.Timedelta(weeks=weeks)
            if candidate < SOURCE_START_UTC:
                break
            nearest = min(
                abs((candidate - actual).total_seconds()) / 3600.0
                for actual in all_anchors
            )
            if nearest <= CONTROL_EVENT_EXCLUSION_HOURS:
                continue
            found += 1
            records.append(
                {
                    "parent_event_id": event.event_id,
                    "parent_episode_id": event.event_episode_id,
                    "control_anchor_utc": candidate,
                    "control_rank": found,
                    "control_method": "prior_same_weekday_and_utc_clock",
                }
            )
            if found >= MAX_CONTROLS_PER_EVENT:
                break
    return DataFrame.from_records(records)


def build_control_catalog(events: DataFrame) -> DataFrame:
    candidates = pd.concat(
        (_candidate_controls_from_maps(events), _generated_prior_week_controls(events)),
        ignore_index=True,
    )
    if candidates.empty:
        raise ValueError("No event-matched controls were produced.")
    parents = events.set_index("event_id")
    candidates = candidates.loc[
        candidates["control_anchor_utc"].ge(SOURCE_START_UTC)
        & candidates["control_anchor_utc"].lt(SOURCE_STOP_UTC)
    ].copy()
    actual_anchors = pd.to_datetime(events["model_anchor_utc"], utc=True).drop_duplicates()
    clean = candidates["control_anchor_utc"].map(
        lambda candidate: (
            (actual_anchors - pd.Timestamp(candidate)).abs().min()
            > pd.Timedelta(hours=CONTROL_EVENT_EXCLUSION_HOURS)
        )
    )
    candidates = candidates.loc[clean].copy()
    candidates["event_family"] = candidates["parent_event_id"].map(
        parents["event_family"]
    )
    candidates["event_kind"] = candidates["parent_event_id"].map(parents["event_kind"])
    candidates["parent_exact_clock"] = candidates["parent_event_id"].map(
        parents["exact_clock"]
    )
    candidates["parent_scheduled"] = candidates["parent_event_id"].map(
        parents["scheduled"]
    )
    candidates["model_period"] = candidates["control_anchor_utc"].map(model_period)
    candidates.drop_duplicates(
        ["parent_event_id", "control_anchor_utc"], keep="first", inplace=True
    )
    counts = candidates.groupby("parent_event_id").size()
    if counts.empty or int(counts.max()) > MAX_CONTROLS_PER_EVENT:
        raise ValueError("Control construction violated the frozen per-event cap.")
    return candidates.sort_values(
        ["control_anchor_utc", "event_family", "parent_event_id"], kind="stable"
    ).reset_index(drop=True)


def _json_values(values: Iterable[Any]) -> str:
    return json.dumps(sorted({str(value) for value in values}), separators=(",", ":"))


def build_sample_catalog(events: DataFrame, controls: DataFrame) -> DataFrame:
    records: list[dict[str, Any]] = []
    for sample_kind, frame, anchor_column in (
        ("actual_event", events, "model_anchor_utc"),
        ("matched_control", controls, "control_anchor_utc"),
    ):
        for anchor, group in frame.groupby(anchor_column, sort=True):
            actual = sample_kind == "actual_event"
            families = group["event_family"].astype(str)
            kinds = group["event_kind"].astype(str)
            if actual:
                parent_ids = group["event_id"].astype(str)
                episode_ids = group["event_episode_id"].astype(str)
                exact = group["exact_clock"].astype(float)
                scheduled = group["scheduled"].astype(float)
                primary = pd.to_numeric(group["source_sign_primary"], errors="coerce")
                secondary = pd.to_numeric(group["source_sign_secondary"], errors="coerce")
                relation = pd.to_numeric(group["crypto_relation_sign"], errors="coerce")
            else:
                parent_ids = group["parent_event_id"].astype(str)
                episode_ids = group["parent_episode_id"].astype(str)
                exact = group["parent_exact_clock"].astype(float)
                scheduled = group["parent_scheduled"].astype(float)
                primary = pd.Series(dtype=float)
                secondary = pd.Series(dtype=float)
                relation = pd.Series(dtype=float)
            primary_clean = primary.dropna()
            secondary_clean = secondary.dropna()
            relation_clean = relation.dropna()
            timestamp = pd.Timestamp(anchor)
            records.append(
                {
                    "sample_id": f"{sample_kind}_{timestamp:%Y%m%dT%H%M%SZ}",
                    "sample_kind": sample_kind,
                    "model_anchor_utc": timestamp,
                    "model_period": model_period(timestamp),
                    "parent_event_ids_json": _json_values(parent_ids),
                    "parent_episode_ids_json": _json_values(episode_ids),
                    "event_families_json": _json_values(families),
                    "event_kinds_json": _json_values(kinds),
                    "parent_family_count": int(families.nunique()),
                    "parent_kind_count": int(kinds.nunique()),
                    "parent_exact_clock_fraction": float(exact.mean()),
                    "parent_scheduled_fraction": float(scheduled.mean()),
                    "current_event_count": len(group) if actual else 0,
                    "current_family_count": int(families.nunique()) if actual else 0,
                    "current_kind_count": int(kinds.nunique()) if actual else 0,
                    "source_sign_available_count": int(primary_clean.size),
                    "source_sign_primary_mean": (
                        float(primary_clean.mean()) if not primary_clean.empty else 0.0
                    ),
                    "source_sign_secondary_mean": (
                        float(secondary_clean.mean()) if not secondary_clean.empty else 0.0
                    ),
                    "crypto_relation_sign_available_count": int(relation_clean.size),
                    "crypto_relation_sign_mean": (
                        float(relation_clean.mean()) if not relation_clean.empty else 0.0
                    ),
                }
            )
    samples = DataFrame.from_records(records).sort_values(
        ["model_anchor_utc", "sample_kind"], kind="stable"
    )
    if samples["model_anchor_utc"].duplicated().any():
        raise ValueError("Actual and control sample anchors unexpectedly overlap.")
    return samples.reset_index(drop=True)


def build_registry() -> dict[str, Any]:
    profiles = {
        name: {
            "profile_id": name,
            "role": name,
            "feature_blocks": definition["blocks"],
            "feature_columns": definition["features"],
            "required_ready_blocks": [READY_BLOCK],
            "targets": list(TARGETS),
            "seed": 2026090801,
        }
        for name, definition in PROFILE_DEFINITIONS.items()
    }
    comparisons = [
        {
            "candidate": candidate,
            "components": list(components),
            "plain_question": (
                f"Does {candidate.replace('_', ' ')} improve on both isolated components?"
            ),
        }
        for candidate, components in COMBINATION_COMPONENTS.items()
    ]
    return {
        "schema_version": 1,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_before_event_freqai_outcome_materialization",
        "profile_count": len(profiles),
        "profiles": profiles,
        "comparisons": comparisons,
        "all_feature_columns": list(ALL_FEATURES),
        "targets_declared_without_values": list(TARGETS),
        "target_metadata": list(TARGET_METADATA),
        "future_outcomes_read": False,
    }


def source_contracts() -> dict[str, Any]:
    return {
        "layer2_events": artifact(layer2.EVENTS_PATH),
        "layer2_controls": artifact(layer2.CONTROLS_PATH),
        "cpi_catalog": artifact(cpi.CATALOG_PATH),
        "source_activity_events": artifact(breadth.EVENTS_PATH),
        "source_activity_controls": artifact(breadth.CONTROLS_PATH),
        "cross_asset_catalog": artifact(cross_asset.CATALOG_PATH),
        "esma_catalog": artifact(esma.CATALOG_PATH),
        "normal_outcome_blind_feature_support": artifact(
            g23_cache.support_manifest_path("normal")
        ),
        "analysis_script": artifact(ANALYSIS_PATH),
    }


def load_freeze() -> dict[str, Any]:
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if freeze.get("status") != "frozen_event_freqai_breadth_before_outcomes":
        raise ValueError("Event FreqAI breadth freeze is not valid.")
    if freeze.get("future_outcomes_read") or freeze.get("profit_used"):
        raise ValueError("Event FreqAI breadth freeze crossed its research boundary.")
    for contract in freeze["source_contracts"].values():
        path = Path(contract["path"])
        if not path.is_file() or g0.sha256_file(path) != contract["sha256"]:
            raise ValueError(f"Frozen event FreqAI source changed: {path}")
    for contract in freeze["frozen_artifacts"].values():
        path = Path(contract["path"])
        if not path.is_file() or g0.sha256_file(path) != contract["sha256"]:
            raise ValueError(f"Frozen event FreqAI artifact changed: {path}")
    return freeze


def execute() -> dict[str, Any]:
    if FREEZE_PATH.is_file():
        return load_freeze()
    events = build_event_catalog()
    controls = build_control_catalog(events)
    samples = build_sample_catalog(events, controls)
    registry = build_registry()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_csv(events, EVENTS_PATH)
    g0.atomic_write_csv(samples, SAMPLES_PATH)
    g0.atomic_write_json(registry, REGISTRY_PATH)
    freeze = {
        "schema_version": 1,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_event_freqai_breadth_before_outcomes",
        "plain_objective": (
            "Use one broad FreqAI sweep to compare fourteen isolated or paired questions "
            "about event timing, signed source information, market state, levels, and "
            "level clusters. Predict reaction, volume and direction behaviour rather than "
            "profit."
        ),
        "event_count": len(events),
        "event_episode_count": int(events["event_episode_id"].nunique()),
        "sample_count": len(samples),
        "actual_sample_count": int(samples["sample_kind"].eq("actual_event").sum()),
        "control_sample_count": int(samples["sample_kind"].eq("matched_control").sum()),
        "event_families": list(FAMILIES),
        "pairs": list(NORMAL_PAIRS),
        "horizons_hours": list(HORIZONS),
        "profile_count": len(PROFILE_DEFINITIONS),
        "profile_ids": list(PROFILE_DEFINITIONS),
        "source_start_utc": SOURCE_START_UTC,
        "source_stop_exclusive_utc": SOURCE_STOP_UTC,
        "event_episode_gap_hours": EVENT_EPISODE_GAP_HOURS,
        "controls_per_event_cap": MAX_CONTROLS_PER_EVENT,
        "control_event_exclusion_hours": CONTROL_EVENT_EXCLUSION_HOURS,
        "evaluation_periods": [
            "walk_forward_validation_2024",
            "walk_forward_validation_2025",
        ],
        "research_relevance_thresholds": {
            "minimum_repeat_rate": 0.55,
            "strong_repeat_rate": 0.65,
            "minimum_combination_uplift_over_each_component": 0.02,
            "minimum_unique_event_episodes_per_period": 20,
            "meaning": "Exploratory programme gates, not native FreqAI pass/fail scores.",
        },
        "causal_alignment": (
            "Each decision is aligned to the first hourly candle opening at or after the "
            "recorded decision time. All OHLCV, market, indicator and level features use "
            "only completed earlier candles."
        ),
        "source_boundaries": [
            (
                "US CPI sign is hotter/cooler than the previous release, not true "
                "expectation surprise."
            ),
            "SEC acceptance time is not claimed to be the first public earnings-news minute.",
            (
                "Cross-market daily anchors are conservative availability clocks, not "
                "intraday releases."
            ),
            "UK CPI and Tankan have event clocks but no historical expectation surprise values.",
            "GDELT is an unsigned activity clock, not story meaning or direction.",
        ],
        "source_contracts": source_contracts(),
        "frozen_artifacts": {
            "events": artifact(EVENTS_PATH),
            "samples": artifact(SAMPLES_PATH),
            "profile_registry": artifact(REGISTRY_PATH),
        },
        "future_outcomes_read": False,
        "profit_used": False,
        "trading_rule_tested": False,
    }
    g0.atomic_write_json(freeze, FREEZE_PATH)
    return freeze


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    result = execute()
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
