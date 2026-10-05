"""Materialize frozen feature support and behavioural targets for event FreqAI."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_freqai_breadth_freeze as frozen,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation23_freqai_cache as g23_cache,
)


ANALYSIS_PATH = Path(__file__).resolve()
RECORD_ROOT = frozen.OUTPUT_ROOT / "freqai_cache"
SUPPORT_PATH = RECORD_ROOT / "event_freqai_outcome_blind_support.json"
CACHE_MANIFEST_PATH = RECORD_ROOT / "event_freqai_cache_manifest.json"
ARTIFACT_ROOT = Path(
    "D:/FreqTradeStuffLargeData/research_outputs/event_hierarchy/"
    "event_freqai_breadth_20260908a/cache"
)
PARENT_READY_BLOCK = g23_cache.READY_BLOCK
READY_BLOCK = frozen.READY_BLOCK
PARENT_COLUMN_MAP = {
    "recent__relative_volume": "relative_volume",
    "recent__range_over_atr_4h": "range_over_atr_4h",
    "recent__absolute_return_over_atr_4h": "absolute_return_over_atr_4h",
    "recent__atr_fraction": "atr_fraction",
}
PARENT_LEVEL_COLUMNS = {
    "distance": "nearest_level_distance_atr",
    "count": "independent_level_family_count_within_0p5atr",
    "role": "source_role_fixed_encoding",
    "timeframe": "source_timeframe_fixed_encoding",
    "crossings": "pre_crossing_count_4h",
}
RAW_METRICS = (
    "range_atr",
    "volume_ratio",
    "close_return_atr",
    "excursion_balance_atr",
)


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def pair_stem(pair: str) -> str:
    return g0.pair_file_stem(pair)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_sources() -> tuple[dict[str, Any], dict[str, Any], DataFrame, DataFrame]:
    freeze = frozen.load_freeze()
    registry = _load_json(frozen.REGISTRY_PATH)
    if registry.get("status") != "frozen_before_event_freqai_outcome_materialization":
        raise ValueError("Event FreqAI profile registry is not frozen.")
    parent_path = g23_cache.support_manifest_path("normal")
    parent = _load_json(parent_path)
    if parent.get("status") != "frozen_outcome_blind_generation23_freqai_support":
        raise ValueError("Generation 23 outcome-blind feature support is not valid.")
    events = pd.read_csv(frozen.EVENTS_PATH)
    samples = pd.read_csv(frozen.SAMPLES_PATH)
    for frame, column in ((events, "model_anchor_utc"), (samples, "model_anchor_utc")):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    return freeze, parent, events, samples


def _parent_inventory(parent: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    inventory = {str(item["pair"]): dict(item) for item in parent["inventory"]}
    missing = sorted(set(frozen.NORMAL_PAIRS).difference(inventory))
    if missing:
        raise ValueError(f"Parent feature support lacks required pairs: {missing}")
    for pair, item in inventory.items():
        path = Path(item["feature_path"])
        if not path.is_file() or g0.sha256_file(path) != item["feature_sha256"]:
            raise ValueError(f"Parent feature cache changed for {pair}: {path}")
    return inventory


def _ohlcv(pair: str) -> DataFrame:
    frame = g0.load_ohlcv(g0.ohlcv_path(pair, "1h")).sort_values("date", kind="stable")
    frame = frame.drop_duplicates("date", keep="last").reset_index(drop=True)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="raise")
    return frame


def _known_return(close: Series, hours: int) -> Series:
    known = close.shift(1)
    return known / close.shift(hours + 1).replace(0.0, np.nan) - 1.0


def shared_cross_market_context(pairs: Sequence[str]) -> DataFrame:
    returns_4h: dict[str, Series] = {}
    returns_24h: dict[str, Series] = {}
    for pair in pairs:
        frame = _ohlcv(pair).set_index("date")
        close = pd.to_numeric(frame["close"], errors="coerce")
        returns_4h[pair] = _known_return(close, 4)
        returns_24h[pair] = _known_return(close, 24)
    four = DataFrame(returns_4h).sort_index()
    day = DataFrame(returns_24h).reindex(four.index)
    output = DataFrame(index=four.index)
    output["cross_market__btc_return_4h"] = four["BTC/USDT:USDT"]
    output["cross_market__btc_return_24h"] = day["BTC/USDT:USDT"]
    output["cross_market__eth_return_4h"] = four["ETH/USDT:USDT"]
    output["cross_market__eth_return_24h"] = day["ETH/USDT:USDT"]
    output["cross_market__equal_weight_return_4h"] = four.median(axis=1)
    output["cross_market__equal_weight_return_24h"] = day.median(axis=1)
    output["cross_market__positive_breadth_4h"] = four.gt(0.0).mean(axis=1)
    output["cross_market__dispersion_4h"] = four.std(axis=1, ddof=0)
    return output.reset_index().rename(columns={"index": "date"})


def _json_set(value: Any) -> set[str]:
    parsed = json.loads(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a JSON list, got {value!r}")
    return {str(item) for item in parsed}


def event_sample_context(samples: DataFrame, events: DataFrame) -> DataFrame:
    event_dates = pd.to_datetime(events["model_anchor_utc"], utc=True, errors="raise")
    records: list[dict[str, Any]] = []
    for sample in samples.itertuples(index=False):
        timestamp = pd.Timestamp(sample.model_anchor_utc)
        actual = str(sample.sample_kind) == "actual_event"
        families = _json_set(sample.event_families_json)
        kinds = _json_set(sample.event_kinds_json)
        known = events.loc[
            event_dates.le(timestamp) & event_dates.gt(timestamp - pd.Timedelta(hours=24))
        ]
        known_signs = pd.to_numeric(known["source_sign_primary"], errors="coerce").dropna()
        row: dict[str, Any] = {
            "date": timestamp,
            "event_identity__is_actual_event": float(actual),
            "event_identity__parent_exact_clock_fraction": float(
                sample.parent_exact_clock_fraction
            ),
            "event_identity__parent_scheduled_fraction": float(
                sample.parent_scheduled_fraction
            ),
            "signed_source__available_count": float(
                sample.source_sign_available_count if actual else 0
            ),
            "signed_source__primary_mean": float(
                sample.source_sign_primary_mean if actual else 0.0
            ),
            "signed_source__secondary_mean": float(
                sample.source_sign_secondary_mean if actual else 0.0
            ),
            "signed_source__crypto_relation_available_count": float(
                sample.crypto_relation_sign_available_count if actual else 0
            ),
            "signed_source__crypto_relation_mean": float(
                sample.crypto_relation_sign_mean if actual else 0.0
            ),
            "event_confluence__current_event_count": float(
                sample.current_event_count if actual else 0
            ),
            "event_confluence__current_family_count": float(
                sample.current_family_count if actual else 0
            ),
            "event_confluence__current_kind_count": float(
                sample.current_kind_count if actual else 0
            ),
            "event_confluence__events_known_within_24h": float(len(known)),
            "event_confluence__families_known_within_24h": float(
                known["event_family"].nunique()
            ),
            "event_confluence__positive_signed_known_within_24h": float(
                known_signs.gt(0.0).sum()
            ),
            "event_confluence__negative_signed_known_within_24h": float(
                known_signs.lt(0.0).sum()
            ),
            "event_confluence__signed_balance_known_within_24h": float(
                known_signs.sum() if not known_signs.empty else 0.0
            ),
        }
        for kind in frozen.KINDS:
            row[f"event_identity__kind_{kind}"] = float(kind in kinds)
        for family in frozen.FAMILIES:
            row[f"event_identity__family_{family}"] = float(family in families)
        records.append(row)
    output = DataFrame.from_records(records)
    if output["date"].duplicated().any():
        raise ValueError("Event feature context contains duplicate sample dates.")
    return output


def _pair_market_context(frame: DataFrame) -> DataFrame:
    output = DataFrame({"date": frame["date"]})
    close = pd.to_numeric(frame["close"], errors="coerce").replace(0.0, np.nan)
    returns = close.pct_change().shift(1)
    known_close = close.shift(1)
    output["recent__pair_return_4h"] = _known_return(close, 4)
    output["recent__pair_return_24h"] = _known_return(close, 24)
    output["background__pair_return_168h"] = _known_return(close, 168)
    output["background__pair_return_720h"] = _known_return(close, 720)
    output["background__realised_volatility_168h"] = returns.rolling(
        168, min_periods=84
    ).std(ddof=0)
    output["background__realised_volatility_720h"] = returns.rolling(
        720, min_periods=360
    ).std(ddof=0)
    high = pd.to_numeric(frame["high"], errors="coerce").shift(1).rolling(
        720, min_periods=360
    ).max()
    low = pd.to_numeric(frame["low"], errors="coerce").shift(1).rolling(
        720, min_periods=360
    ).min()
    output["background__range_position_720h"] = (
        (known_close - low) / (high - low).replace(0.0, np.nan)
    ).clip(0.0, 1.0)
    return output


def _level_context(parent: DataFrame) -> DataFrame:
    count = pd.to_numeric(parent[PARENT_LEVEL_COLUMNS["count"]], errors="coerce").fillna(0.0)
    distance = pd.to_numeric(
        parent[PARENT_LEVEL_COLUMNS["distance"]], errors="coerce"
    ).clip(0.0, 10.0)
    role = pd.to_numeric(parent[PARENT_LEVEL_COLUMNS["role"]], errors="coerce")
    timeframe = pd.to_numeric(
        parent[PARENT_LEVEL_COLUMNS["timeframe"]], errors="coerce"
    )
    crossings = pd.to_numeric(
        parent[PARENT_LEVEL_COLUMNS["crossings"]], errors="coerce"
    ).clip(0.0, 100.0)
    single = count.eq(1.0)
    cluster = count.ge(2.0)
    return DataFrame(
        {
            "single_level__present": single.astype(float),
            "single_level__nearest_distance_atr": distance.where(single, 0.0).fillna(0.0),
            "single_level__source_role_encoding": role.where(single, 0.0).fillna(0.0),
            "single_level__source_timeframe_encoding": timeframe.where(
                single, 0.0
            ).fillna(0.0),
            "single_level__pre_crossing_count_4h": crossings.where(
                single, 0.0
            ).fillna(0.0),
            "cluster__present": cluster.astype(float),
            "cluster__independent_family_count": count.where(cluster, 0.0).clip(
                0.0, 20.0
            ),
            "cluster__nearest_distance_atr": distance.where(cluster, 0.0).fillna(0.0),
            "cluster__pre_crossing_count_4h": crossings.where(
                cluster, 0.0
            ).fillna(0.0),
        }
    )


def build_pair_features(
    *,
    pair: str,
    parent_item: Mapping[str, Any],
    samples: DataFrame,
    event_context: DataFrame,
    cross_context: DataFrame,
) -> DataFrame:
    parent = pd.read_parquet(parent_item["feature_path"])
    parent["date"] = pd.to_datetime(parent["date"], utc=True, errors="raise")
    if parent["date"].duplicated().any():
        raise ValueError(f"Parent feature cache has duplicate dates for {pair}.")
    ohlcv = _ohlcv(pair)
    pair_context = _pair_market_context(ohlcv)
    output = parent[["date", f"ready__{PARENT_READY_BLOCK}"]].copy()
    for target, source in PARENT_COLUMN_MAP.items():
        output[target] = pd.to_numeric(parent[source], errors="coerce")
    output = output.join(_level_context(parent))
    output = output.merge(pair_context, on="date", how="left", validate="one_to_one")
    output = output.merge(cross_context, on="date", how="left", validate="one_to_one")
    output = output.merge(event_context, on="date", how="left", validate="one_to_one")
    event_columns = (
        *frozen.EVENT_IDENTITY_FEATURES,
        *frozen.SIGNED_SOURCE_FEATURES,
        *frozen.EVENT_CONFLUENCE_FEATURES,
    )
    output[list(event_columns)] = output[list(event_columns)].fillna(0.0)
    for column in frozen.ALL_FEATURES:
        output[column] = pd.to_numeric(output[column], errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
    sample_dates = set(pd.to_datetime(samples["model_anchor_utc"], utc=True))
    ready = output[f"ready__{PARENT_READY_BLOCK}"].fillna(False).astype(bool)
    ready &= output["date"].isin(sample_dates)
    ready &= output[list(frozen.ALL_FEATURES)].notna().all(axis=1)
    output[f"ready__{READY_BLOCK}"] = ready
    return output[["date", *frozen.ALL_FEATURES, f"ready__{READY_BLOCK}"]]


def freeze_support(*, overwrite: bool = False) -> dict[str, Any]:
    _, parent, events, samples = load_sources()
    if SUPPORT_PATH.is_file() and not overwrite:
        support = _load_json(SUPPORT_PATH)
        if support.get("status") != "frozen_event_freqai_outcome_blind_support":
            raise ValueError("Existing event FreqAI support is not terminal.")
        return support
    inventory = _parent_inventory(parent)
    cross_context = shared_cross_market_context(tuple(inventory))
    event_context = event_sample_context(samples, events)
    records: list[dict[str, Any]] = []
    for number, pair in enumerate(frozen.NORMAL_PAIRS, start=1):
        features = build_pair_features(
            pair=pair,
            parent_item=inventory[pair],
            samples=samples,
            event_context=event_context,
            cross_context=cross_context,
        )
        path = ARTIFACT_ROOT / "feature_cache" / f"{pair_stem(pair)}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        g0.atomic_write_parquet(features, path)
        records.append(
            {
                "pair": pair,
                "rows": len(features),
                "ready_rows": int(features[f"ready__{READY_BLOCK}"].sum()),
                "feature_path": str(path.resolve()),
                "feature_sha256": g0.sha256_file(path),
                "parent_feature_path": inventory[pair]["feature_path"],
                "parent_feature_sha256": inventory[pair]["feature_sha256"],
                "future_outcomes_read": False,
            }
        )
        print(
            json.dumps(
                {"phase": "event_freqai_support", "processed": number, "total": 5}
            ),
            flush=True,
        )
    support = {
        "schema_version": 1,
        "created_at_utc": g0.utc_now(),
        "status": "frozen_event_freqai_outcome_blind_support",
        "pairs": list(frozen.NORMAL_PAIRS),
        "feature_columns": list(frozen.ALL_FEATURES),
        "targets_declared_without_values": list(frozen.TARGETS),
        "inventory": records,
        "source_contracts": {
            "breadth_freeze": artifact(frozen.FREEZE_PATH),
            "profile_registry": artifact(frozen.REGISTRY_PATH),
            "parent_outcome_blind_support": artifact(
                g23_cache.support_manifest_path("normal")
            ),
            "analysis_script": artifact(ANALYSIS_PATH),
        },
        "future_outcomes_read": False,
        "profit_used": False,
    }
    RECORD_ROOT.mkdir(parents=True, exist_ok=True)
    g0.atomic_write_json(support, SUPPORT_PATH)
    return support


def _true_range(frame: DataFrame) -> Series:
    close = pd.to_numeric(frame["close"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    return pd.concat(
        ((high - low).abs(), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()),
        axis=1,
    ).max(axis=1)


def _future_rolling(series: Series, horizon: int, method: str) -> Series:
    reversed_series = series.iloc[::-1]
    rolling = getattr(reversed_series.rolling(horizon, min_periods=horizon), method)()
    return rolling.iloc[::-1]


def outcome_surface(frame: DataFrame) -> DataFrame:
    output = DataFrame({"date": frame["date"]})
    open_ = pd.to_numeric(frame["open"], errors="coerce")
    high = pd.to_numeric(frame["high"], errors="coerce")
    low = pd.to_numeric(frame["low"], errors="coerce")
    close = pd.to_numeric(frame["close"], errors="coerce")
    volume = pd.to_numeric(frame["volume"], errors="coerce").clip(lower=0.0)
    prior_atr = _true_range(frame).rolling(14, min_periods=14).mean().shift(1)
    prior_volume = volume.shift(1).rolling(168, min_periods=84).median()
    for horizon in frozen.HORIZONS:
        future_high = _future_rolling(high, horizon, "max")
        future_low = _future_rolling(low, horizon, "min")
        future_volume = _future_rolling(volume, horizon, "sum")
        future_close = close.shift(-(horizon - 1))
        range_atr = ((future_high - future_low) / prior_atr.replace(0.0, np.nan)).clip(
            0.0, 100.0
        )
        volume_ratio = (
            future_volume / (prior_volume.replace(0.0, np.nan) * horizon)
        ).clip(0.001, 100.0)
        close_return = ((future_close - open_) / prior_atr.replace(0.0, np.nan)).clip(
            -20.0, 20.0
        )
        excursion_balance = (
            ((future_high - open_) - (open_ - future_low))
            / prior_atr.replace(0.0, np.nan)
        ).clip(-20.0, 20.0)
        output[f"raw_range_atr_h{horizon}"] = range_atr
        output[f"raw_volume_ratio_h{horizon}"] = volume_ratio
        output[f"raw_close_return_atr_h{horizon}"] = close_return
        output[f"raw_excursion_balance_atr_h{horizon}"] = excursion_balance
        output[f"&-meb_log_range_atr_h{horizon}"] = np.log1p(range_atr)
        output[f"&-meb_log_volume_ratio_h{horizon}"] = np.log(volume_ratio)
        output[f"&-meb_close_return_atr_h{horizon}"] = close_return
        output[f"&-meb_excursion_balance_atr_h{horizon}"] = excursion_balance
    return output


def _verify_support() -> dict[str, Any]:
    support = _load_json(SUPPORT_PATH)
    if support.get("status") != "frozen_event_freqai_outcome_blind_support":
        raise ValueError("Event FreqAI feature support is not frozen.")
    for contract in support["source_contracts"].values():
        path = Path(contract["path"])
        if not path.is_file() or g0.sha256_file(path) != contract["sha256"]:
            raise ValueError(f"Event FreqAI support dependency changed: {path}")
    for item in support["inventory"]:
        path = Path(item["feature_path"])
        if not path.is_file() or g0.sha256_file(path) != item["feature_sha256"]:
            raise ValueError(f"Event FreqAI feature support changed: {path}")
    return support


def materialize_outcomes(*, overwrite: bool = False) -> dict[str, Any]:
    frozen.load_freeze()
    support = _verify_support()
    if CACHE_MANIFEST_PATH.is_file() and not overwrite:
        existing = _load_json(CACHE_MANIFEST_PATH)
        if existing.get("status") != "completed_event_freqai_cache":
            raise ValueError("Existing event FreqAI cache is not terminal.")
        return existing
    samples = pd.read_csv(frozen.SAMPLES_PATH)
    samples["model_anchor_utc"] = pd.to_datetime(
        samples["model_anchor_utc"], utc=True, errors="raise"
    )
    sample_metadata = samples.rename(columns={"model_anchor_utc": "date"})
    completed: list[dict[str, Any]] = []
    for number, item in enumerate(support["inventory"], start=1):
        pair = str(item["pair"])
        features = pd.read_parquet(item["feature_path"])
        features["date"] = pd.to_datetime(features["date"], utc=True, errors="raise")
        outcomes = outcome_surface(_ohlcv(pair))
        merged = features[["date", f"ready__{READY_BLOCK}"]].merge(
            outcomes, on="date", how="left", validate="one_to_one"
        )
        ready = merged[f"ready__{READY_BLOCK}"].fillna(False).astype(bool)
        ready &= merged[list(frozen.TARGETS)].notna().all(axis=1)
        merged[f"ready__{READY_BLOCK}"] = ready
        for target in frozen.TARGETS:
            merged.loc[~ready, target] = np.nan
        merged = merged.merge(
            sample_metadata,
            on="date",
            how="left",
            validate="one_to_one",
        )
        merged["period"] = merged["model_period"].fillna("outside_sample_catalog")
        event_cache = merged[["date", "period", f"ready__{READY_BLOCK}", *frozen.TARGETS]]
        raw_columns = [
            f"raw_{metric}_h{horizon}"
            for metric in RAW_METRICS
            for horizon in frozen.HORIZONS
        ]
        evaluation = merged.loc[
            ready,
            [
                "date",
                "period",
                "sample_id",
                "sample_kind",
                "parent_event_ids_json",
                "parent_episode_ids_json",
                "event_families_json",
                "event_kinds_json",
                *frozen.TARGETS,
                *raw_columns,
            ],
        ].copy()
        feature_lookup = features.set_index("date")
        evaluation["baseline_pair_return_4h"] = evaluation["date"].map(
            feature_lookup["recent__pair_return_4h"]
        )
        evaluation["baseline_btc_return_4h"] = evaluation["date"].map(
            feature_lookup["cross_market__btc_return_4h"]
        )
        root = ARTIFACT_ROOT
        event_path = root / "event_cache" / f"{pair_stem(pair)}.parquet"
        evaluation_path = root / "evaluation_cache" / f"{pair_stem(pair)}.parquet"
        event_path.parent.mkdir(parents=True, exist_ok=True)
        evaluation_path.parent.mkdir(parents=True, exist_ok=True)
        g0.atomic_write_parquet(event_cache, event_path)
        g0.atomic_write_parquet(evaluation, evaluation_path)
        completed.append(
            {
                **item,
                "event_path": str(event_path.resolve()),
                "event_sha256": g0.sha256_file(event_path),
                "evaluation_path": str(evaluation_path.resolve()),
                "evaluation_sha256": g0.sha256_file(evaluation_path),
                "eligible_rows": int(ready.sum()),
                "actual_event_rows": int(
                    evaluation["sample_kind"].eq("actual_event").sum()
                ),
                "matched_control_rows": int(
                    evaluation["sample_kind"].eq("matched_control").sum()
                ),
            }
        )
        print(
            json.dumps(
                {"phase": "event_freqai_targets", "processed": number, "total": 5}
            ),
            flush=True,
        )
    result = {
        "schema_version": 1,
        "created_at_utc": g0.utc_now(),
        "status": "completed_event_freqai_cache",
        "pairs": list(frozen.NORMAL_PAIRS),
        "feature_columns": list(frozen.ALL_FEATURES),
        "targets": list(frozen.TARGETS),
        "target_metadata": list(frozen.TARGET_METADATA),
        "inventory": completed,
        "source_contracts": {
            "breadth_freeze": artifact(frozen.FREEZE_PATH),
            "profile_registry": artifact(frozen.REGISTRY_PATH),
            "outcome_blind_support": artifact(SUPPORT_PATH),
            "analysis_script": artifact(ANALYSIS_PATH),
        },
        "integrity": {
            "feature_definitions_frozen_before_outcomes": True,
            "feature_values_materialized_and_hashed_before_outcomes": True,
            "matched_controls_in_training_surface": True,
            "profit_used": False,
        },
    }
    g0.atomic_write_json(result, CACHE_MANIFEST_PATH)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-support", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    result = (
        freeze_support(overwrite=args.overwrite)
        if args.prepare_support
        else materialize_outcomes(overwrite=args.overwrite)
    )
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
