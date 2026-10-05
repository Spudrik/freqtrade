"""Run the frozen Layer 2 market-event individual-link tests."""

from __future__ import annotations

# Repository-local imports follow root bootstrap.
# ruff: noqa: E402
import argparse
import json
import math
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_event_hierarchy_layer2_freeze as frozen,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)


ANALYSIS_PATH = Path(__file__).resolve()
FREEZE_RESULT = frozen.OUTPUT_ROOT / "layer2_freeze_result.json"
FREEZE_PATH = frozen.OUTPUT_ROOT / "layer2_freeze.json"
EVENTS_PATH = frozen.OUTPUT_ROOT / "layer2_event_catalog.csv"
CONTROLS_PATH = frozen.OUTPUT_ROOT / "layer2_control_map.csv"
OUTPUT_ROOT = frozen.OUTPUT_ROOT / "direct_review_20260904a"
DETAIL_ROOT = (
    Path(r"D:\FreqTradeStuffLargeData")
    / "research_outputs"
    / "event_hierarchy"
    / "layer2_individual_links_20260904a"
)

HORIZONS = frozen.HORIZONS
BASELINE_HOURS = 24 * 90
BASELINE_MINIMUM = 24 * 30
PRIMARY_CONTROL = "matched_prior_state"
MINIMUM_GENERAL_EVENTS = 10
MINIMUM_MEME_COMMON_EVENTS = 10

LEVEL_COLUMNS = {
    "prior_7d_high": ("generic_rolling_high_168", "prior_extreme"),
    "prior_7d_low": ("generic_rolling_low_168", "prior_extreme"),
    "prior_30d_high": ("generic_rolling_high_720", "prior_extreme"),
    "prior_30d_low": ("generic_rolling_low_720", "prior_extreme"),
    "ema50": ("generic_ema_50", "moving_average"),
    "ema200": ("generic_ema_200", "moving_average"),
    "bollinger20_upper": ("generic_bb20_upper", "bollinger"),
    "bollinger20_mid": ("generic_bb20_mid", "bollinger"),
    "bollinger20_lower": ("generic_bb20_lower", "bollinger"),
    "vwap168": ("generic_vwap_168", "vwap"),
    "round_nearest": ("generic_round_nearest", "round_number"),
}


def artifact(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": g0.sha256_file(path)}


def load_frozen_inputs() -> tuple[dict[str, Any], DataFrame, DataFrame, DataFrame]:
    result = json.loads(FREEZE_RESULT.read_text(encoding="utf-8"))
    freeze = json.loads(FREEZE_PATH.read_text(encoding="utf-8"))
    if result.get("status") != "completed_layer2_outcome_blind_freeze":
        raise ValueError("Layer 2 freeze result is not terminal.")
    if freeze.get("status") != "frozen_before_layer2_individual_link_outcomes":
        raise ValueError("Layer 2 questions are not frozen.")
    if freeze.get("outcomes_read"):
        raise ValueError("Layer 2 freeze unexpectedly opened outcomes.")
    for name, path in (
        ("freeze", FREEZE_PATH),
        ("events", EVENTS_PATH),
        ("controls", CONTROLS_PATH),
    ):
        if result["artifacts"][name]["sha256"] != g0.sha256_file(path):
            raise ValueError(f"Frozen Layer 2 artifact changed: {path}")
    events = pd.read_csv(EVENTS_PATH)
    controls = pd.read_csv(CONTROLS_PATH)
    for frame, columns in (
        (events, ["anchor_utc"]),
        (controls, ["event_anchor_utc", "control_anchor_utc"]),
    ):
        for column in columns:
            frame[column] = pd.to_datetime(frame[column], utc=True)
    samples = sample_table(events, controls)
    return freeze, events, controls, samples


def sample_table(events: DataFrame, controls: DataFrame) -> DataFrame:
    event_samples = events[
        [
            "event_id",
            "event_source",
            "event_family",
            "anchor_utc",
            "whole_event_partition",
        ]
    ].copy()
    event_samples["sample_type"] = "event"
    event_samples["control_type"] = "event"
    event_samples["control_rank"] = 0
    event_samples["sample_anchor_utc"] = event_samples["anchor_utc"]
    event_samples = event_samples.drop(columns="anchor_utc")

    metadata = events[
        ["event_id", "event_family", "whole_event_partition"]
    ].drop_duplicates("event_id")
    control_samples = controls.merge(metadata, on="event_id", how="left", validate="many_to_one")
    control_samples["sample_type"] = "control"
    control_samples["sample_anchor_utc"] = control_samples["control_anchor_utc"]
    control_samples = control_samples[
        [
            "event_id",
            "event_source",
            "event_family",
            "whole_event_partition",
            "sample_type",
            "control_type",
            "control_rank",
            "sample_anchor_utc",
        ]
    ]
    output = pd.concat([event_samples, control_samples], ignore_index=True)
    output["sample_id"] = (
        output["event_id"].astype(str)
        + "|"
        + output["sample_type"].astype(str)
        + "|"
        + output["control_type"].astype(str)
        + "|"
        + output["control_rank"].astype(str)
    )
    if output["sample_id"].duplicated().any():
        raise ValueError("Layer 2 sample identifiers are not unique.")
    return output


def future_max(values: Series, hours: int) -> Series:
    return values.rolling(hours, min_periods=hours).max().shift(-(hours - 1))


def future_min(values: Series, hours: int) -> Series:
    return values.rolling(hours, min_periods=hours).min().shift(-(hours - 1))


def future_sum(values: Series, hours: int) -> Series:
    return values.rolling(hours, min_periods=hours).sum().shift(-(hours - 1))


def safe_ratio(numerator: Series, denominator: Series) -> Series:
    return numerator.div(denominator.where(denominator.gt(0)))


def local_context(frame: DataFrame, atr_pre: Series) -> DataFrame:
    generic = g0.generic_level_frame(frame).shift(1)
    values = DataFrame(
        {
            name: pd.to_numeric(generic[column], errors="coerce")
            for name, (column, _) in LEVEL_COLUMNS.items()
        },
        index=frame.index,
    )
    distance = values.sub(frame["open"], axis=0).abs().div(atr_pre, axis=0)
    valid_history = distance.notna().any(axis=1)
    nearest_name = Series(pd.NA, index=distance.index, dtype="string")
    nearest_name.loc[valid_history] = distance.loc[valid_history].idxmin(axis=1)
    nearest_distance = distance.min(axis=1)
    near_primary = distance.le(0.25)
    near_wide = distance.le(0.50)
    family_by_name = {name: family for name, (_, family) in LEVEL_COLUMNS.items()}
    names_by_family: dict[str, list[str]] = {}
    for name, family in family_by_name.items():
        names_by_family.setdefault(family, []).append(name)
    family_count_primary = pd.concat(
        [near_primary[names].any(axis=1) for names in names_by_family.values()],
        axis=1,
    ).sum(axis=1)
    family_count_wide = pd.concat(
        [near_wide[names].any(axis=1) for names in names_by_family.values()],
        axis=1,
    ).sum(axis=1)
    state = np.select(
        [
            ~valid_history,
            family_count_primary.ge(2),
            near_primary.sum(axis=1).ge(1),
        ],
        ["insufficient_level_history", "cluster", "single_level"],
        default="far_from_frozen_levels",
    )
    return DataFrame(
        {
            "nearest_level": nearest_name,
            "nearest_level_family": nearest_name.map(family_by_name),
            "nearest_level_distance_atr": nearest_distance,
            "near_level_count_primary": near_primary.sum(axis=1),
            "near_family_count_primary": family_count_primary,
            "near_family_count_wide": family_count_wide,
            "local_level_state": state,
        },
        index=frame.index,
    )


def pair_metric_surface(pair: str) -> tuple[DataFrame, DataFrame]:
    frame = g0.load_ohlcv(g0.ohlcv_path(pair, "1h"))
    frame = frame.sort_values("date", kind="stable").drop_duplicates("date").reset_index(drop=True)
    true_range = pd.concat(
        [
            frame["high"].sub(frame["low"]),
            frame["high"].sub(frame["close"].shift(1)).abs(),
            frame["low"].sub(frame["close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr_pre = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean().shift(1)
    hourly_returns = frame["close"].pct_change()
    frame["pre_atr14"] = atr_pre
    frame["pre_return_30d"] = frame["close"].shift(1).div(frame["close"].shift(721)).sub(1)
    frame["pre_volatility_24h"] = hourly_returns.shift(1).rolling(24, min_periods=24).std()
    frame["pre_volatility_30d"] = hourly_returns.shift(1).rolling(
        720, min_periods=360
    ).std()
    local = local_context(frame, atr_pre)
    frame = pd.concat([frame, local], axis=1)
    response_columns: list[str] = []
    for horizon in HORIZONS:
        response_return = frame["close"].shift(-(horizon - 1)).div(frame["open"]).sub(1)
        response_range = future_max(frame["high"], horizon).sub(
            future_min(frame["low"], horizon)
        ).div(frame["open"])
        response_volume = future_sum(frame["volume"], horizon)
        components: list[Series] = []
        for name, values in (
            ("abs_return", response_return.abs()),
            ("range", response_range),
            ("volume", response_volume),
        ):
            baseline = values.shift(horizon).rolling(
                BASELINE_HOURS, min_periods=BASELINE_MINIMUM
            ).median()
            ratio = safe_ratio(values, baseline)
            frame[f"{name}_ratio_{horizon}h"] = ratio
            components.append(ratio)
        activity = pd.concat(components, axis=1).median(axis=1, skipna=False)
        threshold = activity.shift(horizon).rolling(
            BASELINE_HOURS, min_periods=BASELINE_MINIMUM
        ).quantile(0.95)
        frame[f"return_{horizon}h"] = response_return
        frame[f"activity_score_{horizon}h"] = activity
        frame[f"activity_threshold_{horizon}h"] = threshold
        frame[f"unusual_activity_{horizon}h"] = activity.gt(threshold)
        response_columns.extend(
            [
                f"abs_return_ratio_{horizon}h",
                f"range_ratio_{horizon}h",
                f"volume_ratio_{horizon}h",
                f"return_{horizon}h",
                f"activity_score_{horizon}h",
                f"activity_threshold_{horizon}h",
                f"unusual_activity_{horizon}h",
            ]
        )
    base_columns = [
        "date",
        "open",
        "pre_atr14",
        "pre_return_30d",
        "pre_volatility_24h",
        "pre_volatility_30d",
        "nearest_level",
        "nearest_level_family",
        "nearest_level_distance_atr",
        "near_level_count_primary",
        "near_family_count_primary",
        "near_family_count_wide",
        "local_level_state",
    ]
    return frame[base_columns + response_columns], frame


def extract_pair_samples(
    pair: str, samples: DataFrame
) -> tuple[DataFrame, DataFrame]:
    surface, raw = pair_metric_surface(pair)
    joined = samples.merge(
        surface,
        left_on="sample_anchor_utc",
        right_on="date",
        how="left",
        validate="many_to_one",
    )
    records: list[DataFrame] = []
    fixed = [
        "sample_id",
        "event_id",
        "event_source",
        "event_family",
        "whole_event_partition",
        "sample_type",
        "control_type",
        "control_rank",
        "sample_anchor_utc",
        "pre_return_30d",
        "pre_volatility_24h",
        "pre_volatility_30d",
        "pre_atr14",
        "nearest_level",
        "nearest_level_family",
        "nearest_level_distance_atr",
        "near_level_count_primary",
        "near_family_count_primary",
        "near_family_count_wide",
        "local_level_state",
    ]
    for horizon in HORIZONS:
        part = joined[fixed].copy()
        part["pair"] = pair
        part["horizon_hours"] = horizon
        part["abs_return_ratio"] = joined[f"abs_return_ratio_{horizon}h"]
        part["range_ratio"] = joined[f"range_ratio_{horizon}h"]
        part["volume_ratio"] = joined[f"volume_ratio_{horizon}h"]
        part["signed_return"] = joined[f"return_{horizon}h"]
        part["activity_score"] = joined[f"activity_score_{horizon}h"]
        part["activity_threshold"] = joined[f"activity_threshold_{horizon}h"]
        part["unusual_activity"] = joined[f"unusual_activity_{horizon}h"].where(
            joined[f"activity_score_{horizon}h"].notna()
        )
        records.append(part)
    metrics = pd.concat(records, ignore_index=True)
    ranges = extract_range_samples(pair, samples, raw)
    return metrics, ranges


def _first_true_position(values: np.ndarray) -> int | None:
    positions = np.flatnonzero(values)
    return int(positions[0]) if len(positions) else None


def boundary_reaction(
    future: DataFrame, lower: float, upper: float, atr: float
) -> tuple[bool | None, str]:
    upper_touch = _first_true_position(future["high"].to_numpy() >= upper)
    lower_touch = _first_true_position(future["low"].to_numpy() <= lower)
    if upper_touch is None and lower_touch is None:
        return None, "no_boundary_revisit"
    if upper_touch == lower_touch:
        return None, "both_boundaries_same_candle"
    side = "upper" if lower_touch is None or (
        upper_touch is not None and upper_touch < lower_touch
    ) else "lower"
    touch = upper_touch if side == "upper" else lower_touch
    window = future.iloc[touch : touch + 8]
    if side == "upper":
        inward = _first_true_position(window["low"].to_numpy() <= upper - 0.5 * atr)
        outward = _first_true_position(window["high"].to_numpy() >= upper + 0.5 * atr)
    else:
        inward = _first_true_position(window["high"].to_numpy() >= lower + 0.5 * atr)
        outward = _first_true_position(window["low"].to_numpy() <= lower - 0.5 * atr)
    if inward is None:
        return False, f"{side}_no_inward_half_atr"
    if outward is not None and outward <= inward:
        return False, f"{side}_outward_first_or_same_candle"
    return True, f"{side}_inward_half_atr_first"


def extract_range_samples(pair: str, samples: DataFrame, raw: DataFrame) -> DataFrame:
    indexed = raw.set_index("date", drop=False)
    records: list[dict[str, Any]] = []
    for sample in samples.itertuples(index=False):
        anchor = pd.Timestamp(sample.sample_anchor_utc)
        if anchor not in indexed.index:
            continue
        position = int(indexed.index.get_loc(anchor))
        formation = raw.iloc[position : position + 24]
        evaluation = raw.iloc[position + 24 : position + 168]
        if len(formation) != 24 or len(evaluation) != 144:
            continue
        atr = float(indexed.at[anchor, "pre_atr14"])
        if not np.isfinite(atr) or atr <= 0:
            continue
        lower = float(formation["low"].min())
        upper = float(formation["high"].max())
        inside = evaluation["close"].between(lower, upper, inclusive="both")
        final = evaluation.tail(24)["close"]
        permanent_failure = bool(final.gt(upper).all() or final.lt(lower).all())
        reaction, reaction_reason = boundary_reaction(evaluation, lower, upper, atr)
        records.append(
            {
                "sample_id": sample.sample_id,
                "event_id": sample.event_id,
                "event_source": sample.event_source,
                "whole_event_partition": sample.whole_event_partition,
                "sample_type": sample.sample_type,
                "control_type": sample.control_type,
                "control_rank": sample.control_rank,
                "pair": pair,
                "sample_anchor_utc": anchor,
                "range_width_atr": (upper - lower) / atr,
                "residence_fraction": float(inside.mean()),
                "permanent_failure": permanent_failure,
                "stable_range": bool(inside.mean() >= 0.60 and not permanent_failure),
                "boundary_reaction": reaction,
                "boundary_reaction_reason": reaction_reason,
            }
        )
    return DataFrame.from_records(records)


def cohort_map(freeze: dict[str, Any]) -> dict[str, str]:
    output = {
        "BTC/USDT:USDT": "btc",
        "ETH/USDT:USDT": "eth",
    }
    output.update({pair: "established_alts" for pair in freeze["cohorts"]["established_alts"]})
    output.update({pair: "memes" for pair in freeze["cohorts"]["top_ten_traded_memes"]})
    return output


def all_pairs(freeze: dict[str, Any]) -> list[str]:
    return list(
        dict.fromkeys(
            ["BTC/USDT:USDT", "ETH/USDT:USDT"]
            + list(freeze["cohorts"]["established_alts"])
            + list(freeze["cohorts"]["top_ten_traded_memes"])
        )
    )


def scope_metrics(pair_metrics: DataFrame, freeze: dict[str, Any]) -> DataFrame:
    identifiers = [
        "sample_id",
        "event_id",
        "event_source",
        "whole_event_partition",
        "sample_type",
        "control_type",
        "control_rank",
        "sample_anchor_utc",
        "horizon_hours",
    ]
    pieces: list[DataFrame] = []
    for pair, scope in (("BTC/USDT:USDT", "btc"), ("ETH/USDT:USDT", "eth")):
        part = pair_metrics.loc[pair_metrics["pair"].eq(pair), identifiers + [
            "abs_return_ratio",
            "range_ratio",
            "volume_ratio",
            "signed_return",
            "activity_score",
            "unusual_activity",
        ]].copy()
        part["scope"] = scope
        part["asset_count"] = part["activity_score"].notna().astype(int)
        pieces.append(part)
    for scope, pairs in (
        ("established_alts", freeze["cohorts"]["established_alts"]),
        ("memes", freeze["cohorts"]["top_ten_traded_memes"]),
    ):
        source = pair_metrics.loc[pair_metrics["pair"].isin(pairs)].copy()
        grouped = source.groupby(identifiers, dropna=False, sort=False)
        part = grouped.agg(
            abs_return_ratio=("abs_return_ratio", "median"),
            range_ratio=("range_ratio", "median"),
            volume_ratio=("volume_ratio", "median"),
            signed_return=("signed_return", "median"),
            activity_score=("activity_score", "median"),
            unusual_count=("unusual_activity", "sum"),
            asset_count=("activity_score", "count"),
        ).reset_index()
        required = len(pairs)
        part["unusual_activity"] = part["unusual_count"].ge(math.ceil(required / 2)).where(
            part["asset_count"].eq(required)
        )
        part.loc[part["asset_count"].ne(required), ["signed_return", "activity_score"]] = np.nan
        part["scope"] = scope
        pieces.append(part.drop(columns="unusual_count"))
    return pd.concat(pieces, ignore_index=True)


def paired_activity_rows(scopes: DataFrame) -> DataFrame:
    events = scopes.loc[scopes["sample_type"].eq("event")].drop(
        columns=["control_type", "control_rank"]
    )
    controls = scopes.loc[scopes["sample_type"].eq("control")].copy()
    keys = ["event_id", "event_source", "scope", "horizon_hours"]
    control_summary = controls.groupby(keys + ["control_type"], sort=False).agg(
        control_activity_score=("activity_score", "median"),
        control_abs_return_ratio=("abs_return_ratio", "median"),
        control_range_ratio=("range_ratio", "median"),
        control_volume_ratio=("volume_ratio", "median"),
        control_rows=("activity_score", "count"),
    ).reset_index()
    paired = events.merge(control_summary, on=keys, how="left", validate="one_to_many")
    paired["paired_uplift"] = paired["activity_score"] - paired["control_activity_score"]
    paired["paired_success"] = paired["paired_uplift"].gt(0).where(
        paired["paired_uplift"].notna()
    )
    paired["event_year"] = paired["sample_anchor_utc"].dt.year
    return paired


def _partition_rate(group: DataFrame, partition: str) -> float:
    values = group.loc[group["whole_event_partition"].eq(partition), "paired_success"]
    return float(values.mean()) if values.notna().any() else np.nan


def _minimum_leave_year_out(group: DataFrame) -> float:
    rates = [
        group.loc[group["event_year"].ne(year), "paired_success"].mean()
        for year in sorted(group["event_year"].dropna().unique())
    ]
    finite = [float(value) for value in rates if pd.notna(value)]
    return min(finite) if finite else np.nan


def summarize_activity(paired: DataFrame) -> DataFrame:
    records: list[dict[str, Any]] = []
    keys = ["event_source", "scope", "horizon_hours", "control_type"]
    for values, group in paired.groupby(keys, dropna=False, sort=False):
        valid = group.loc[group["paired_success"].notna()]
        success = float(valid["paired_success"].mean()) if len(valid) else np.nan
        development = _partition_rate(valid, "development_2021_2023")
        validation = _partition_rate(valid, "internal_validation_2024_2025")
        leave_year = _minimum_leave_year_out(valid)
        cross_period = (
            len(valid) >= MINIMUM_GENERAL_EVENTS
            and success >= 0.55
            and float(valid["paired_uplift"].median()) > 0
            and (pd.isna(development) or development >= 0.50)
            and (pd.isna(validation) or validation >= 0.50)
            and leave_year >= 0.50
        )
        records.append(
            {
                **dict(zip(keys, values, strict=True)),
                "event_count": len(valid),
                "paired_success_rate": success,
                "median_event_activity": valid["activity_score"].median(),
                "median_control_activity": valid["control_activity_score"].median(),
                "median_paired_uplift": valid["paired_uplift"].median(),
                "median_event_abs_return_ratio": valid["abs_return_ratio"].median(),
                "median_control_abs_return_ratio": valid[
                    "control_abs_return_ratio"
                ].median(),
                "median_event_range_ratio": valid["range_ratio"].median(),
                "median_control_range_ratio": valid["control_range_ratio"].median(),
                "median_event_volume_ratio": valid["volume_ratio"].median(),
                "median_control_volume_ratio": valid["control_volume_ratio"].median(),
                "unusual_event_rate": valid["unusual_activity"].mean(),
                "development_success_rate": development,
                "validation_success_rate": validation,
                "minimum_leave_one_year_out_rate": leave_year,
                "meets_55_floor": bool(cross_period),
                "meets_65_target": bool(cross_period and success >= 0.65),
            }
        )
    return DataFrame.from_records(records)


def classify_leadership(scopes: DataFrame) -> DataFrame:
    source = scopes.loc[scopes["scope"].isin(["btc", "eth", "established_alts"])].copy()
    records: list[dict[str, Any]] = []
    sample_columns = [
        "sample_id",
        "event_id",
        "event_source",
        "whole_event_partition",
        "sample_type",
        "control_type",
        "control_rank",
        "sample_anchor_utc",
    ]
    for _, group in source.groupby("sample_id", sort=False):
        metadata = group.iloc[0]
        detections: dict[str, int | None] = {}
        for scope in ("btc", "eth", "established_alts"):
            values = group.loc[group["scope"].eq(scope)].set_index("horizon_hours")
            detected = [
                horizon
                for horizon in HORIZONS
                if horizon in values.index and bool(values.at[horizon, "unusual_activity"])
            ]
            name = "broad_participation" if scope == "established_alts" else scope
            detections[name] = min(detected) if detected else None
        finite = {name: value for name, value in detections.items() if value is not None}
        if not finite:
            label = "no_candidate_crossed"
            first_hour = None
            leader_sign = np.nan
        else:
            first_hour = min(finite.values())
            earliest = [name for name, value in finite.items() if value == first_hour]
            label = earliest[0] if len(earliest) == 1 else "simultaneous"
            if label == "simultaneous":
                leader_sign = np.nan
            else:
                scope = "established_alts" if label == "broad_participation" else label
                value = group.loc[
                    group["scope"].eq(scope) & group["horizon_hours"].eq(first_hour),
                    "signed_return",
                ].iloc[0]
                leader_sign = float(np.sign(value)) if pd.notna(value) and value != 0 else np.nan
        record = {column: metadata[column] for column in sample_columns}
        record.update(
            {
                "leadership_label": label,
                "first_detection_hours": first_hour,
                "leader_sign": leader_sign,
                "btc_detection_hours": detections["btc"],
                "eth_detection_hours": detections["eth"],
                "broad_detection_hours": detections["broad_participation"],
            }
        )
        records.append(record)
    return DataFrame.from_records(records)


def summarize_leadership(leadership: DataFrame) -> DataFrame:
    events = leadership.loc[leadership["sample_type"].eq("event")]
    records: list[dict[str, Any]] = []
    for source, group in events.groupby("event_source", sort=False):
        counts = group["leadership_label"].value_counts()
        detected = group.loc[group["leadership_label"].ne("no_candidate_crossed")]
        dominant = counts.index[0] if len(counts) else "none"
        dominant_rate = float(counts.iloc[0] / len(group)) if len(group) else np.nan
        records.append(
            {
                "event_source": source,
                "event_count": len(group),
                "detected_event_count": len(detected),
                "btc_unique_leader_count": int(counts.get("btc", 0)),
                "eth_unique_leader_count": int(counts.get("eth", 0)),
                "broad_unique_leader_count": int(counts.get("broad_participation", 0)),
                "simultaneous_count": int(counts.get("simultaneous", 0)),
                "no_detection_count": int(counts.get("no_candidate_crossed", 0)),
                "most_common_state": str(dominant),
                "most_common_state_rate": dominant_rate,
            }
        )
    return DataFrame.from_records(records)


def transmission_rows(scopes: DataFrame, leadership: DataFrame) -> DataFrame:
    indexed = scopes.set_index(["sample_id", "scope", "horizon_hours"])
    records: list[dict[str, Any]] = []
    for lead in leadership.itertuples(index=False):
        if lead.leadership_label not in {"btc", "eth", "broad_participation"}:
            continue
        first = int(lead.first_detection_hours)
        later = next((value for value in HORIZONS if value > first), None)
        if later is None or not np.isfinite(lead.leader_sign):
            continue
        for follower in ("established_alts", "memes"):
            if lead.leadership_label == "broad_participation" and follower == "established_alts":
                continue
            try:
                first_row = indexed.loc[(lead.sample_id, follower, first)]
                later_row = indexed.loc[(lead.sample_id, follower, later)]
            except KeyError:
                continue
            if pd.isna(first_row["signed_return"]) or pd.isna(later_row["signed_return"]):
                continue
            if bool(first_row["unusual_activity"]):
                continue
            incremental = (1 + float(later_row["signed_return"])) / (
                1 + float(first_row["signed_return"])
            ) - 1
            records.append(
                {
                    "sample_id": lead.sample_id,
                    "event_id": lead.event_id,
                    "event_source": lead.event_source,
                    "whole_event_partition": lead.whole_event_partition,
                    "sample_type": lead.sample_type,
                    "control_type": lead.control_type,
                    "control_rank": lead.control_rank,
                    "leader": lead.leadership_label,
                    "leader_detection_hours": first,
                    "follower": follower,
                    "follower_end_hours": later,
                    "leader_sign": lead.leader_sign,
                    "follower_incremental_return": incremental,
                    "direction_aligned": bool(np.sign(incremental) == lead.leader_sign),
                }
            )
    return DataFrame.from_records(records)


def summarize_transmission(rows: DataFrame) -> DataFrame:
    if rows.empty:
        return DataFrame()
    records: list[dict[str, Any]] = []
    for values, group in rows.groupby(["event_source", "follower"], sort=False):
        event = group.loc[group["sample_type"].eq("event")]
        control = group.loc[
            group["sample_type"].eq("control") & group["control_type"].eq(PRIMARY_CONTROL)
        ]
        event_rate = event["direction_aligned"].mean()
        control_rate = control["direction_aligned"].mean()
        passes = (
            len(event) >= MINIMUM_GENERAL_EVENTS
            and event_rate >= 0.55
            and pd.notna(control_rate)
            and event_rate > control_rate
        )
        records.append(
            {
                "event_source": values[0],
                "follower": values[1],
                "event_count": len(event),
                "event_alignment_rate": event_rate,
                "matched_control_count": len(control),
                "matched_control_alignment_rate": control_rate,
                "alignment_uplift": event_rate - control_rate,
                "meets_55_floor_and_control": bool(passes),
                "meets_65_target_and_control": bool(passes and event_rate >= 0.65),
            }
        )
    return DataFrame.from_records(records)


def rolling_meme_fits(
    pair_metrics: DataFrame, samples: DataFrame, memes: Iterable[str]
) -> DataFrame:
    btc = g0.load_ohlcv(g0.ohlcv_path("BTC/USDT:USDT", "1h"))[["date", "close"]]
    btc = btc.sort_values("date").drop_duplicates("date")
    btc["btc_log_return"] = np.log(btc["close"]).diff()
    records: list[dict[str, Any]] = []
    unique_anchors = samples[["sample_anchor_utc"]].drop_duplicates()
    for pair in memes:
        coin = g0.load_ohlcv(g0.ohlcv_path(pair, "1h"))[["date", "close"]]
        coin = coin.sort_values("date").drop_duplicates("date")
        coin["coin_log_return"] = np.log(coin["close"]).diff()
        aligned = btc[["date", "btc_log_return"]].merge(
            coin[["date", "coin_log_return"]], on="date", how="inner", validate="one_to_one"
        ).set_index("date")
        for anchor in unique_anchors["sample_anchor_utc"]:
            history = aligned.loc[aligned.index < anchor].tail(24 * 90).dropna()
            if len(history) < 1080:
                continue
            design = np.column_stack(
                [np.ones(len(history)), history["btc_log_return"].to_numpy()]
            )
            target = history["coin_log_return"].to_numpy()
            alpha, beta = np.linalg.lstsq(design, target, rcond=None)[0]
            residual = target - design @ np.array([alpha, beta])
            residual_std = float(np.std(residual, ddof=2))
            records.append(
                {
                    "pair": pair,
                    "sample_anchor_utc": anchor,
                    "meme_alpha_90d": float(alpha),
                    "meme_beta_to_btc_90d": float(beta),
                    "meme_residual_std_90d": residual_std,
                    "meme_training_rows": len(history),
                }
            )
    fits = DataFrame.from_records(records)
    meme_rows = pair_metrics.loc[pair_metrics["pair"].isin(memes)].merge(
        fits, on=["pair", "sample_anchor_utc"], how="left", validate="many_to_one"
    )
    btc_returns = pair_metrics.loc[pair_metrics["pair"].eq("BTC/USDT:USDT"), [
        "sample_id",
        "horizon_hours",
        "signed_return",
    ]].rename(columns={"signed_return": "btc_signed_return"})
    meme_rows = meme_rows.merge(
        btc_returns, on=["sample_id", "horizon_hours"], how="left", validate="many_to_one"
    )
    actual_log = np.log1p(meme_rows["signed_return"])
    btc_log = np.log1p(meme_rows["btc_signed_return"])
    expected_log = (
        meme_rows["meme_alpha_90d"] * meme_rows["horizon_hours"]
        + meme_rows["meme_beta_to_btc_90d"] * btc_log
    )
    residual_log = actual_log - expected_log
    residual_scale = meme_rows["meme_residual_std_90d"] * np.sqrt(
        meme_rows["horizon_hours"]
    )
    meme_rows["residual_log_return"] = residual_log
    meme_rows["residual_z"] = residual_log.div(residual_scale.where(residual_scale.gt(0)))
    meme_rows["amplification_toward_btc_z"] = meme_rows["residual_z"] * np.sign(btc_log)
    meme_rows["same_direction_as_btc"] = np.sign(actual_log).eq(np.sign(btc_log))
    meme_rows["amplified_beyond_ordinary"] = (
        meme_rows["same_direction_as_btc"]
        & meme_rows["amplification_toward_btc_z"].gt(0)
    )
    return meme_rows


def summarize_meme_amplification(rows: DataFrame, required_coins: int) -> DataFrame:
    grouped = rows.groupby(
        [
            "event_id",
            "event_source",
            "whole_event_partition",
            "sample_type",
            "control_type",
            "horizon_hours",
        ],
        sort=False,
    ).agg(
        coin_count=("amplification_toward_btc_z", "count"),
        median_amplification_z=("amplification_toward_btc_z", "median"),
        coin_success_rate=("amplified_beyond_ordinary", "mean"),
    ).reset_index()
    common = grouped.loc[
        grouped["sample_type"].eq("event") & grouped["coin_count"].eq(required_coins)
    ]
    records: list[dict[str, Any]] = []
    for values, group in common.groupby(["event_source", "horizon_hours"], sort=False):
        success = group["median_amplification_z"].gt(0)
        rate = success.mean()
        enough = len(group) >= MINIMUM_MEME_COMMON_EVENTS
        records.append(
            {
                "event_source": values[0],
                "horizon_hours": values[1],
                "whole_event_count_with_all_ten_memes": len(group),
                "event_success_rate": rate,
                "median_event_amplification_z": group["median_amplification_z"].median(),
                "mean_coin_success_rate": group["coin_success_rate"].mean(),
                "meets_55_floor": bool(enough and rate >= 0.55),
                "coverage_status": "testable" if enough else "coverage_parked_under_10_events",
            }
        )
    return DataFrame.from_records(records)


def background_rows(pair_metrics: DataFrame) -> DataFrame:
    btc = pair_metrics.loc[
        pair_metrics["pair"].eq("BTC/USDT:USDT")
        & pair_metrics["horizon_hours"].isin([4, 24])
    ].copy()
    pivot = btc.pivot(
        index=[
            "sample_id",
            "event_id",
            "event_source",
            "whole_event_partition",
            "sample_type",
            "control_type",
            "control_rank",
            "pre_return_30d",
        ],
        columns="horizon_hours",
        values="signed_return",
    ).reset_index()
    pivot = pivot.rename(columns={4: "return_4h", 24: "return_24h"})
    pivot["background"] = np.where(
        pivot["pre_return_30d"].lt(0), "negative_30d_background", "positive_30d_background"
    )
    pivot["positive_initial_move"] = pivot["return_4h"].gt(0)
    pivot["positive_move_faded_by_half"] = (
        pivot["positive_initial_move"] & pivot["return_24h"].le(0.5 * pivot["return_4h"])
    )
    pivot["negative_initial_move"] = pivot["return_4h"].lt(0)
    pivot["negative_move_extended"] = (
        pivot["negative_initial_move"] & pivot["return_24h"].lt(pivot["return_4h"])
    )
    return pivot


def summarize_background(pivot: DataFrame) -> DataFrame:
    records: list[dict[str, Any]] = []
    for values, group in pivot.groupby(
        ["event_source", "sample_type", "control_type", "background"], sort=False
    ):
        positive = group.loc[group["positive_initial_move"]]
        negative = group.loc[group["negative_initial_move"]]
        records.append(
            {
                "event_source": values[0],
                "sample_type": values[1],
                "control_type": values[2],
                "background": values[3],
                "whole_event_count": len(group),
                "positive_initial_count": len(positive),
                "positive_fade_rate": positive["positive_move_faded_by_half"].mean(),
                "negative_initial_count": len(negative),
                "negative_extension_rate": negative["negative_move_extended"].mean(),
            }
        )
    return DataFrame.from_records(records)


def summarize_background_contrasts(pivot: DataFrame) -> DataFrame:
    records: list[dict[str, Any]] = []

    def fade_values(frame: DataFrame, background: str) -> Series:
        return frame.loc[
            frame["background"].eq(background) & frame["positive_initial_move"],
            "positive_move_faded_by_half",
        ]

    for source, source_rows in pivot.groupby("event_source", sort=False):
        event = source_rows.loc[source_rows["sample_type"].eq("event")]
        control = source_rows.loc[
            source_rows["sample_type"].eq("control")
            & source_rows["control_type"].eq(PRIMARY_CONTROL)
        ]
        event_negative = fade_values(event, "negative_30d_background")
        event_positive = fade_values(event, "positive_30d_background")
        control_negative = fade_values(control, "negative_30d_background")
        control_positive = fade_values(control, "positive_30d_background")
        event_gap = event_negative.mean() - event_positive.mean()
        control_gap = control_negative.mean() - control_positive.mean()
        partition_gaps: dict[str, float] = {}
        partition_minimum_counts: dict[str, int] = {}
        for partition in (
            "development_2021_2023",
            "internal_validation_2024_2025",
        ):
            part = event.loc[event["whole_event_partition"].eq(partition)]
            negative = fade_values(part, "negative_30d_background")
            positive = fade_values(part, "positive_30d_background")
            partition_minimum_counts[partition] = min(len(negative), len(positive))
            partition_gaps[partition] = (
                float(negative.mean() - positive.mean())
                if len(negative) and len(positive)
                else np.nan
            )
        passes = (
            len(event_negative) >= MINIMUM_GENERAL_EVENTS
            and len(event_positive) >= MINIMUM_GENERAL_EVENTS
            and event_negative.mean() >= 0.55
            and event_gap >= 0.10
            and pd.notna(control_gap)
            and event_gap > control_gap
            and partition_minimum_counts["development_2021_2023"] >= 3
            and partition_minimum_counts["internal_validation_2024_2025"] >= 3
            and partition_gaps["development_2021_2023"] > 0
            and partition_gaps["internal_validation_2024_2025"] > 0
        )
        records.append(
            {
                "event_source": source,
                "negative_background_positive_move_count": len(event_negative),
                "positive_background_positive_move_count": len(event_positive),
                "negative_background_fade_rate": event_negative.mean(),
                "positive_background_fade_rate": event_positive.mean(),
                "event_fade_rate_gap": event_gap,
                "matched_control_fade_rate_gap": control_gap,
                "event_minus_control_gap": event_gap - control_gap,
                "development_fade_rate_gap": partition_gaps["development_2021_2023"],
                "validation_fade_rate_gap": partition_gaps[
                    "internal_validation_2024_2025"
                ],
                "meets_55_floor_and_control": bool(passes),
                "meets_65_target_and_control": bool(
                    passes and event_negative.mean() >= 0.65
                ),
            }
        )
    return DataFrame.from_records(records)


def single_asset_partition_increment(
    target: DataFrame, far: DataFrame, partition: str
) -> float:
    target_part = target.loc[
        target["whole_event_partition"].eq(partition), "median_paired_uplift"
    ]
    far_part = far.loc[
        far["whole_event_partition"].eq(partition), "median_paired_uplift"
    ]
    if len(target_part) < 3 or len(far_part) < 3:
        return np.nan
    return float(target_part.median() - far_part.median())


def local_context_rows(
    pair_metrics: DataFrame, freeze: dict[str, Any]
) -> tuple[DataFrame, DataFrame]:
    events = pair_metrics.loc[pair_metrics["sample_type"].eq("event")].copy()
    controls = pair_metrics.loc[
        pair_metrics["sample_type"].eq("control")
        & pair_metrics["control_type"].eq(PRIMARY_CONTROL)
    ]
    keys = ["event_id", "event_source", "pair", "horizon_hours"]
    baseline = controls.groupby(keys, sort=False)["activity_score"].median().rename(
        "control_activity_score"
    ).reset_index()
    rows = events.merge(baseline, on=keys, how="left", validate="one_to_one")
    rows["paired_uplift"] = rows["activity_score"] - rows["control_activity_score"]
    mapping = cohort_map(freeze)
    rows["cohort"] = rows["pair"].map(mapping)
    rows["event_year"] = rows["sample_anchor_utc"].dt.year
    event_state = rows.groupby(
        [
            "event_id",
            "event_source",
            "whole_event_partition",
            "event_year",
            "cohort",
            "horizon_hours",
            "local_level_state",
        ],
        dropna=False,
        sort=False,
    ).agg(
        coin_count=("paired_uplift", "count"),
        median_paired_uplift=("paired_uplift", "median"),
    ).reset_index()
    records: list[dict[str, Any]] = []
    for values, group in event_state.groupby(
        ["event_source", "cohort", "horizon_hours"], sort=False
    ):
        source, cohort, horizon = values
        far = group.loc[group["local_level_state"].eq("far_from_frozen_levels")]
        for state in ("single_level", "cluster"):
            target = group.loc[group["local_level_state"].eq(state)]
            if target.empty or far.empty:
                continue
            if cohort in {"established_alts", "memes"}:
                pivot = group.pivot_table(
                    index=["event_id", "whole_event_partition", "event_year"],
                    columns="local_level_state",
                    values="median_paired_uplift",
                    aggfunc="median",
                )
                if state not in pivot or "far_from_frozen_levels" not in pivot:
                    continue
                difference = (
                    pivot[state] - pivot["far_from_frozen_levels"]
                ).dropna().rename("increment_beyond_far").reset_index()
                method = "within_same_event_across_cohort_members"
                positive_rate = difference["increment_beyond_far"].gt(0).mean()
                median_increment = difference["increment_beyond_far"].median()
                event_count = len(difference)
                development_values = difference.loc[
                    difference["whole_event_partition"].eq("development_2021_2023"),
                    "increment_beyond_far",
                ]
                validation_values = difference.loc[
                    difference["whole_event_partition"].eq(
                        "internal_validation_2024_2025"
                    ),
                    "increment_beyond_far",
                ]
                development_check = (
                    development_values.gt(0).mean() if len(development_values) else np.nan
                )
                validation_check = (
                    validation_values.gt(0).mean() if len(validation_values) else np.nan
                )
            else:
                method = "across_whole_events_for_single_asset"
                positive_rate = target["median_paired_uplift"].gt(0).mean()
                median_increment = (
                    target["median_paired_uplift"].median()
                    - far["median_paired_uplift"].median()
                )
                event_count = len(target)
                development_check = single_asset_partition_increment(
                    target, far, "development_2021_2023"
                )
                validation_check = single_asset_partition_increment(
                    target, far, "internal_validation_2024_2025"
                )
            chronological_pass = (
                pd.notna(development_check)
                and pd.notna(validation_check)
                and development_check > 0
                and validation_check > 0
                if method == "across_whole_events_for_single_asset"
                else pd.notna(development_check)
                and pd.notna(validation_check)
                and development_check >= 0.50
                and validation_check >= 0.50
            )
            passes = (
                event_count >= MINIMUM_GENERAL_EVENTS
                and positive_rate >= 0.55
                and median_increment > 0
                and chronological_pass
            )
            records.append(
                {
                    "event_source": source,
                    "cohort": cohort,
                    "horizon_hours": horizon,
                    "local_level_state": state,
                    "comparison_method": method,
                    "whole_event_count": event_count,
                    "positive_increment_rate": positive_rate,
                    "median_increment_beyond_far": median_increment,
                    "development_check": development_check,
                    "validation_check": validation_check,
                    "far_state_median_uplift": far["median_paired_uplift"].median(),
                    "meets_55_floor_and_beats_far": bool(passes),
                }
            )
    summary = DataFrame.from_records(records)
    return rows, summary


def range_scope_rows(ranges: DataFrame, freeze: dict[str, Any]) -> DataFrame:
    mapping = cohort_map(freeze)
    output = ranges.copy()
    output["scope"] = output["pair"].map(mapping)
    identifiers = [
        "sample_id",
        "event_id",
        "event_source",
        "whole_event_partition",
        "sample_type",
        "control_type",
        "control_rank",
        "scope",
    ]
    return output.groupby(identifiers, sort=False).agg(
        asset_count=("stable_range", "count"),
        stable_fraction=("stable_range", "mean"),
        median_residence_fraction=("residence_fraction", "median"),
        permanent_failure_fraction=("permanent_failure", "mean"),
        boundary_reaction_rate=("boundary_reaction", "mean"),
        boundary_reaction_count=("boundary_reaction", "count"),
    ).reset_index()


def summarize_ranges(scoped: DataFrame) -> DataFrame:
    events = scoped.loc[scoped["sample_type"].eq("event")].copy()
    controls = scoped.loc[
        scoped["sample_type"].eq("control") & scoped["control_type"].eq(PRIMARY_CONTROL)
    ]
    keys = ["event_id", "event_source", "scope"]
    baseline = controls.groupby(keys, sort=False).agg(
        control_stable_fraction=("stable_fraction", "median"),
        control_residence_fraction=("median_residence_fraction", "median"),
        control_failure_fraction=("permanent_failure_fraction", "median"),
    ).reset_index()
    paired = events.merge(baseline, on=keys, how="left", validate="one_to_one")
    paired["stable_fraction_uplift"] = (
        paired["stable_fraction"] - paired["control_stable_fraction"]
    )
    records: list[dict[str, Any]] = []
    for values, group in paired.groupby(["event_source", "scope"], sort=False):
        valid = group.loc[group["stable_fraction_uplift"].notna()]
        stable_mean = valid["stable_fraction"].mean()
        paired_rate = valid["stable_fraction_uplift"].gt(0).mean()
        passes = (
            len(valid) >= MINIMUM_GENERAL_EVENTS
            and stable_mean >= 0.55
            and paired_rate >= 0.55
            and valid["stable_fraction_uplift"].median() > 0
        )
        records.append(
            {
                "event_source": values[0],
                "scope": values[1],
                "whole_event_count": len(valid),
                "mean_event_stable_fraction": stable_mean,
                "mean_control_stable_fraction": valid["control_stable_fraction"].mean(),
                "median_stability_uplift": valid["stable_fraction_uplift"].median(),
                "paired_stability_success_rate": paired_rate,
                "median_event_residence_fraction": valid["median_residence_fraction"].median(),
                "mean_event_permanent_failure_fraction": valid[
                    "permanent_failure_fraction"
                ].mean(),
                "boundary_reaction_rate": valid["boundary_reaction_rate"].mean(),
                "meets_55_floor_and_control": bool(passes),
            }
        )
    return DataFrame.from_records(records)


def route_decisions(
    activity: DataFrame,
    leadership: DataFrame,
    transmission: DataFrame,
    amplification: DataFrame,
    background_contrasts: DataFrame,
    local: DataFrame,
    ranges: DataFrame,
) -> DataFrame:
    records: list[dict[str, str]] = []
    for source, route in (
        ("official_fomc", "official_event_to_market_activity"),
        ("gdelt_activity_spike", "gdelt_spike_to_market_activity"),
    ):
        cells = activity.loc[
            activity["event_source"].eq(source)
            & activity["control_type"].eq(PRIMARY_CONTROL)
            & activity["meets_55_floor"]
        ]
        verdict = "retained_individual_lead" if len(cells) else "parked_no_control_resistant_link"
        detail = (
            f"{len(cells)} scope/horizon cells met the frozen cross-period 55% rule."
        )
        records.append({"route": route, "verdict": verdict, "plain_result": detail})
    records.append(
        {
            "route": "event_to_signed_direction",
            "verdict": "coverage_parked",
            "plain_result": (
                "No historical expectation-and-surprise sign exists; direction was not guessed."
            ),
        }
    )
    dominant = leadership.sort_values("most_common_state_rate", ascending=False).head(1)
    if len(dominant):
        row = dominant.iloc[0]
        lead_verdict = (
            "retained_conditional_description"
            if row["most_common_state"] in {"btc", "eth", "broad_participation"}
            and row["most_common_state_rate"] >= 0.55
            else "simultaneous_or_inconsistent_no_leader"
        )
        lead_plain = (
            f"Most common state was {row['most_common_state']} in "
            f"{row['most_common_state_rate']:.1%} of {int(row['event_count'])} events."
        )
    else:
        lead_verdict = "insufficient"
        lead_plain = "No eligible leadership rows."
    records.append(
        {"route": "btc_eth_breadth_leadership", "verdict": lead_verdict, "plain_result": lead_plain}
    )
    for follower, route in (
        ("established_alts", "leader_to_established_alts"),
        ("memes", "leader_to_memes"),
    ):
        passed = transmission.loc[
            transmission["follower"].eq(follower)
            & transmission["meets_55_floor_and_control"]
        ] if not transmission.empty else DataFrame()
        records.append(
            {
                "route": route,
                "verdict": "retained_individual_lead" if len(passed) else "parked_no_complete_lead",
                "plain_result": f"{len(passed)} event-source cells beat their matched controls.",
            }
        )
    amp_testable = amplification.loc[amplification["coverage_status"].eq("testable")]
    amp_passed = amp_testable.loc[amp_testable["meets_55_floor"]]
    records.append(
        {
            "route": "meme_residual_amplification",
            "verdict": "retained_individual_lead" if len(amp_passed) else "coverage_parked",
            "plain_result": (
                "The full top-ten meme cohort has too few complete whole events."
                if amp_testable.empty
                else f"{len(amp_passed)} horizons met the 55% rule."
            ),
        }
    )
    background_passed = background_contrasts.loc[
        background_contrasts["meets_55_floor_and_control"]
    ]
    strongest_background = background_contrasts.sort_values(
        "event_minus_control_gap", ascending=False
    ).head(1)
    records.append(
        {
            "route": "slow_background_modification",
            "verdict": (
                "retained_conditional_lead"
                if len(background_passed)
                else "parked_no_clear_modifier"
            ),
            "plain_result": (
                "Negative background added "
                f"{strongest_background.iloc[0]['event_minus_control_gap']:.1%} "
                "fade-rate separation beyond matched controls."
                if len(strongest_background)
                else "Too few positive moves in both background groups."
            ),
        }
    )
    local_passed = local.loc[local["meets_55_floor_and_beats_far"]]
    records.append(
        {
            "route": "coin_local_level_and_cluster_description",
            "verdict": (
                "retained_individual_lead"
                if len(local_passed)
                else "parked_no_complete_local_link"
            ),
            "plain_result": f"{len(local_passed)} source/cohort/horizon/state cells passed.",
        }
    )
    range_passed = ranges.loc[ranges["meets_55_floor_and_control"]]
    records.append(
        {
            "route": "post_event_range_behaviour",
            "verdict": (
                "retained_individual_lead"
                if len(range_passed)
                else "parked_no_stable_range_link"
            ),
            "plain_result": f"{len(range_passed)} source/scope cells passed.",
        }
    )
    return DataFrame.from_records(records)


def render_report(decisions: DataFrame, result: dict[str, Any]) -> str:
    lines = [
        "# Event Hierarchy Layer 2 - Complete Individual-Link Review",
        "",
        f"- Status: `{result['status']}`",
        f"- Whole events tested: `{result['event_count']}`",
        f"- Distinct coins tested: `{result['pair_count']}`",
        "- Profit or trade returns used: **No**",
        "- Event direction guessed without a valid surprise sign: **No**",
        "",
        "| Individual question | Decision | Plain result |",
        "| --- | --- | --- |",
    ]
    for row in decisions.itertuples(index=False):
        lines.append(f"| `{row.route}` | `{row.verdict}` | {row.plain_result} |")
    lines.extend(
        [
            "",
            (
                "A retained item is only a lead for the next frozen pairwise batch. "
                "It is not a trade rule."
            ),
            "Unsupported or under-sized siblings were parked rather than silently dropped.",
            "",
        ]
    )
    return "\n".join(lines)


def execute(*, overwrite: bool = False) -> dict[str, Any]:
    result_path = OUTPUT_ROOT / "layer2_direct_result.json"
    if result_path.is_file() and not overwrite:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed_event_hierarchy_layer2_direct_review":
            raise ValueError("Existing Layer 2 direct result is not terminal.")
        return result
    freeze, events, _, samples = load_frozen_inputs()
    metric_parts: list[DataFrame] = []
    range_parts: list[DataFrame] = []
    for pair in all_pairs(freeze):
        metrics, ranges = extract_pair_samples(pair, samples)
        metric_parts.append(metrics)
        range_parts.append(ranges)
    pair_metrics = pd.concat(metric_parts, ignore_index=True)
    range_details = pd.concat(range_parts, ignore_index=True)
    scopes = scope_metrics(pair_metrics, freeze)
    paired = paired_activity_rows(scopes)
    activity_summary = summarize_activity(paired)
    leadership_details = classify_leadership(scopes)
    leadership_summary = summarize_leadership(leadership_details)
    transmission_details = transmission_rows(scopes, leadership_details)
    transmission_summary = summarize_transmission(transmission_details)
    amplification_details = rolling_meme_fits(
        pair_metrics, samples, freeze["cohorts"]["top_ten_traded_memes"]
    )
    amplification_summary = summarize_meme_amplification(
        amplification_details, len(freeze["cohorts"]["top_ten_traded_memes"])
    )
    background_detail = background_rows(pair_metrics)
    background_summary = summarize_background(background_detail)
    background_contrast_summary = summarize_background_contrasts(background_detail)
    local_details, local_summary = local_context_rows(pair_metrics, freeze)
    range_scopes = range_scope_rows(range_details, freeze)
    range_summary = summarize_ranges(range_scopes)
    decisions = route_decisions(
        activity_summary,
        leadership_summary,
        transmission_summary,
        amplification_summary,
        background_contrast_summary,
        local_summary,
        range_summary,
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    DETAIL_ROOT.mkdir(parents=True, exist_ok=True)
    detail_paths = {
        "pair_metrics": DETAIL_ROOT / "pair_sample_metrics.parquet",
        "paired_activity": DETAIL_ROOT / "paired_activity_rows.parquet",
        "range_details": DETAIL_ROOT / "post_event_range_rows.parquet",
        "transmission_details": DETAIL_ROOT / "transmission_rows.parquet",
        "amplification_details": DETAIL_ROOT / "meme_amplification_rows.parquet",
        "local_details": DETAIL_ROOT / "local_context_rows.parquet",
        "background_details": DETAIL_ROOT / "background_rows.parquet",
    }
    for key, frame in (
        ("pair_metrics", pair_metrics),
        ("paired_activity", paired),
        ("range_details", range_details),
        ("transmission_details", transmission_details),
        ("amplification_details", amplification_details),
        ("local_details", local_details),
        ("background_details", background_detail),
    ):
        g0.atomic_write_parquet(frame, detail_paths[key])
    summary_frames = {
        "activity_summary": activity_summary,
        "leadership_summary": leadership_summary,
        "transmission_summary": transmission_summary,
        "meme_amplification_summary": amplification_summary,
        "background_summary": background_summary,
        "background_contrast_summary": background_contrast_summary,
        "local_context_summary": local_summary,
        "range_summary": range_summary,
        "route_decisions": decisions,
    }
    summary_paths: dict[str, Path] = {}
    for name, frame in summary_frames.items():
        path = OUTPUT_ROOT / f"{name}.csv"
        g0.atomic_write_csv(frame, path)
        summary_paths[name] = path
    result = {
        "schema_version": 1,
        "status": "completed_event_hierarchy_layer2_direct_review",
        "created_at_utc": g0.utc_now(),
        "event_count": len(events),
        "pair_count": len(all_pairs(freeze)),
        "sample_count": len(samples),
        "profit_used": False,
        "event_direction_without_surprise_used": False,
        "retained_route_count": int(decisions["verdict"].str.startswith("retained").sum()),
        "parked_route_count": int((~decisions["verdict"].str.startswith("retained")).sum()),
        "freeze_contract": artifact(FREEZE_PATH),
        "summary_artifacts": {name: artifact(path) for name, path in summary_paths.items()},
        "detail_artifacts": {name: artifact(path) for name, path in detail_paths.items()},
    }
    report_path = OUTPUT_ROOT / "layer2_plain_review.md"
    report_path.write_text(render_report(decisions, result), encoding="utf-8", newline="\n")
    result["summary_artifacts"]["plain_review"] = artifact(report_path)
    g0.atomic_write_json(result, result_path)
    return result


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.execute:
        print(json.dumps({"status": "ready_not_executed", "outcomes_read": False}, indent=2))
        return 0
    print(json.dumps(execute(overwrite=args.overwrite), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
