"""Run the frozen, breadth-first Generation 11 direct relationship screen.

The screen deliberately answers all seven sibling questions before any descendant is
opened.  It compares market behaviour at causal calculated-area contacts with four
location controls and separately scores simple, trader-readable direction rules.  It
does not use profit, entries, exits, or strategy returns.
"""

from __future__ import annotations

# Keep native numerical libraries inside the project runtime allowance.
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
import gc
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation0 as g0,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation6_direct_screen as g6d,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation11_freeze as g11z,
)
from user_data.Custom_Launcher.research.context_features import (
    market_reaction_zone_generation11_preflight as g11p,
)


OUTPUT_ROOT = g11z.OUTPUT_ROOT
DIRECT_ROOT = OUTPUT_ROOT / "generation11_direct_screen"
LARGE_DIRECT_ROOT = g0.LARGE_ARTIFACT_ROOT / "generation11_direct_screen"
PREFLIGHT_RECORD = (
    g11p.PREFLIGHT_ROOT
    / g11p.DEFAULT_PREFLIGHT_ID
    / "g11_preflight_record.json"
)
DEFAULT_RUN_ID = "g11_broad_direct_20260822a"

HORIZONS = g11z.HORIZONS
VALIDATION_PERIODS = g11z.VALIDATION_PERIODS
CONTROLS = g11z.LEVEL_CONTROLS
MIN_EVENTS = 50
MIN_COINS = 5
MIN_SUBGROUP_COINS = 3
COOLDOWN_HOURS = 8
BOOTSTRAP_SAMPLES = 384

OUTCOME_METRICS = (
    "reaction_rate",
    "absolute_excursion_atr",
    "volume_ratio",
    "range_ratio",
    "dwell_fraction",
    "crossings",
)
BASELINE_METHODS = (
    "development_majority_path",
    "always_through_approach_baseline",
)

EXTRA_INPUT_COLUMNS = (
    "representation",
    "mechanism_group",
    "level_side",
)
OUTCOME_COLUMNS = tuple(
    dict.fromkeys(
        [
            "time_to_away_0_5atr",
            "time_to_through_0_5atr",
            *(
                f"{stem}_h{horizon}"
                for horizon in HORIZONS
                for stem in (
                    "abs_excursion_atr",
                    "away_excursion_atr",
                    "through_excursion_atr",
                    "volume_ratio",
                    "range_ratio",
                    "dwell_fraction",
                    "crossings",
                )
            ),
        ]
    )
)
READ_COLUMNS = tuple(
    dict.fromkeys((*g11p.INPUT_COLUMNS, *EXTRA_INPUT_COLUMNS, *OUTCOME_COLUMNS))
)

NORMAL_GROUPS: dict[str, tuple[str, ...]] = {
    "btc_separate": ("BTC/USDT:USDT",),
    "bnb_single_coin_audit": ("BNB/USDT:USDT",),
    "doge_bridge_single_coin_audit": ("DOGE/USDT:USDT",),
    "established_altcoins": (
        "ADA/USDT:USDT",
        "AVAX/USDT:USDT",
        "BNB/USDT:USDT",
        "ETH/USDT:USDT",
        "LINK/USDT:USDT",
        "SOL/USDT:USDT",
        "TRX/USDT:USDT",
        "XRP/USDT:USDT",
    ),
    "smart_contract_platforms": (
        "ADA/USDT:USDT",
        "AVAX/USDT:USDT",
        "BNB/USDT:USDT",
        "ETH/USDT:USDT",
        "SOL/USDT:USDT",
    ),
    "other_established_altcoins": (
        "LINK/USDT:USDT",
        "TRX/USDT:USDT",
        "XRP/USDT:USDT",
    ),
}


def artifact(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": g0.sha256_file(path),
    }


def validate_contracts() -> tuple[dict[str, Any], dict[str, Any], DataFrame]:
    frozen = g11z.validate_existing_freeze()
    if not PREFLIGHT_RECORD.is_file():
        raise FileNotFoundError(PREFLIGHT_RECORD)
    preflight = json.loads(PREFLIGHT_RECORD.read_text(encoding="utf-8"))
    if preflight.get("status") != "completed_generation11_outcome_blind_support_preflight":
        raise ValueError("Generation 11 outcome-blind preflight is incomplete.")
    if preflight.get("supported_routes") != len(g11z.ROUTES):
        raise ValueError("The complete seven-route portfolio did not pass preflight.")
    freeze_contract = preflight["source_contracts"]["generation11_freeze"]
    if g0.sha256_file(g11z.FREEZE_PATH) != freeze_contract["sha256"]:
        raise ValueError("Generation 11 freeze changed after preflight.")
    manifest = json.loads(g11z.G6_EVENT_MANIFEST.read_text(encoding="utf-8"))
    manifest_contract = preflight["source_contracts"]["generation6_event_manifest"]
    if g0.sha256_file(g11z.G6_EVENT_MANIFEST) != manifest_contract["sha256"]:
        raise ValueError("Generation 6 event manifest changed after preflight.")
    if manifest.get("status") != "completed_shared_causal_event_cache":
        raise ValueError("The causal event cache is incomplete.")
    if manifest["summary"].get("causal_timestamp_violations") != 0:
        raise ValueError("The causal event cache has timestamp violations.")
    if len(manifest.get("tasks", [])) != 20:
        raise ValueError("The normal-plus-meme event portfolio is incomplete.")
    for task in manifest["tasks"]:
        path = Path(task["event_path"])
        if not path.is_file() or g0.sha256_file(path) != task["event_sha256"]:
            raise ValueError(f"Generation 11 event source changed: {path}")
    threshold_contract = preflight["artifacts"]["thresholds"]
    threshold_path = Path(threshold_contract["path"])
    if g0.sha256_file(threshold_path) != threshold_contract["sha256"]:
        raise ValueError("Frozen Generation 11 thresholds changed after preflight.")
    thresholds = pd.read_csv(threshold_path)
    return frozen, manifest, thresholds


def numeric(frame: DataFrame, column: str) -> Series:
    return pd.to_numeric(frame[column], errors="coerce").replace(
        [np.inf, -np.inf], np.nan
    )


def signed_vote(frame: DataFrame, columns: Sequence[str]) -> Series:
    votes = np.zeros(len(frame), dtype=np.int16)
    available = np.zeros(len(frame), dtype=np.int16)
    for column in columns:
        values = numeric(frame, column).to_numpy(dtype=float)
        valid = np.isfinite(values)
        votes += np.where(valid, np.sign(values), 0).astype(np.int16)
        available += valid.astype(np.int16)
    result = np.sign(votes).astype(np.int8)
    result[available == 0] = 0
    return Series(result, index=frame.index, dtype="int8")


def direction_vote_to_path(vote: Series, approach_code: Series) -> Series:
    vote_values = pd.to_numeric(vote, errors="coerce").fillna(0).to_numpy(dtype=np.int8)
    approach = (
        pd.to_numeric(approach_code, errors="coerce")
        .fillna(0)
        .to_numpy(dtype=np.int8)
    )
    path = (vote_values * approach).astype(np.int8)
    path[(vote_values == 0) | (approach == 0)] = 0
    return Series(path, index=vote.index, dtype="int8")


def threshold_state(
    frame: DataFrame,
    thresholds: DataFrame,
    *,
    feature: str,
    labels: tuple[str, str, str],
) -> Series:
    mapping = g11p.threshold_map(thresholds, feature)
    output = Series("unavailable", index=frame.index, dtype="string")
    values = numeric(frame, feature)
    for cohort, (lower, upper) in mapping.items():
        scope = frame["cohort"].eq(cohort) & values.notna()
        output.loc[scope & values.le(lower)] = labels[0]
        output.loc[scope & values.gt(lower) & values.lt(upper)] = labels[1]
        output.loc[scope & values.ge(upper)] = labels[2]
    return output


def any_relationship(frame: DataFrame, suffix: str) -> Series:
    columns = [
        column
        for column in g11p.RELATIONSHIP_COLUMNS
        if column.endswith(suffix)
    ]
    return frame[columns].fillna(False).astype(bool).any(axis=1)


def enrich_predictor_states(frame: DataFrame, thresholds: DataFrame) -> DataFrame:
    output = frame.copy()
    output["activity_state"] = g11p.assign_activity_state(output, thresholds)
    output["trend_strength"] = threshold_state(
        output,
        thresholds,
        feature="state__adx14",
        labels=("weak", "ordinary", "strong"),
    )
    output["local_two_vote"] = signed_vote(
        output, ("state__ema20_slope", "state__return_slope")
    )
    # RSI is stored on its native 0-100 scale; center it before the five-input vote.
    output["state__rsi14_centered"] = numeric(output, "state__rsi14") - 50.0
    output["local_five_vote"] = signed_vote(
        output,
        (
            "state__ema20_slope",
            "state__return_slope",
            "state__return_acceleration",
            "state__macd_histogram",
            "state__rsi14_centered",
        ),
    )
    output["xm_cohort_breadth_centered"] = (
        numeric(output, "xm_cohort_breadth_positive") - 0.5
    )
    output["market_vote"] = signed_vote(
        output,
        (
            "xm_btc_return_1h",
            "xm_btc_return_4h",
            "xm_eth_return_1h",
            "xm_eth_return_4h",
            "xm_cohort_breadth_centered",
        ),
    )
    approach = numeric(output, "approach_code").fillna(0).astype("int8")
    local_path = direction_vote_to_path(output["local_five_vote"], approach)
    market_path = direction_vote_to_path(output["market_vote"], approach)
    output["trend_alignment"] = np.select(
        [local_path.eq(1), local_path.eq(-1)],
        ["points_through_area", "points_away_from_area"],
        default="unclear",
    )
    output["market_alignment"] = np.select(
        [market_path.eq(1), market_path.eq(-1)],
        ["wider_market_points_through", "wider_market_points_away"],
        default="wider_market_unclear",
    )
    different = any_relationship(output, "different_mechanism_agreement")
    cluster = any_relationship(output, "any_cross_timeframe_cluster")
    opposing = any_relationship(output, "opposing_side_overlap")
    output["has_opposing_overlap"] = opposing
    output["cluster_state"] = np.select(
        [opposing, different, cluster],
        [
            "opposing_level_overlap",
            "independent_indicator_agreement",
            "other_cross_timeframe_cluster",
        ],
        default="isolated_level",
    )
    return output


def add_horizon_outcomes(frame: DataFrame, horizon: int) -> DataFrame:
    output = frame.copy()
    output["absolute_excursion_atr"] = numeric(
        output, f"abs_excursion_atr_h{horizon}"
    )
    for metric in ("volume_ratio", "range_ratio", "dwell_fraction", "crossings"):
        output[metric] = numeric(output, f"{metric}_h{horizon}")
    width = numeric(output, "zone_half_width_atr")
    price_threshold = np.maximum(0.5, width.fillna(0.5).to_numpy(dtype=float))
    output["reaction_rate"] = (
        output["absolute_excursion_atr"].ge(price_threshold)
        & output["volume_ratio"].ge(1.25)
    ).astype("float64")
    return output


def cell_id(**parts: Series | str) -> Series:
    first = next(value for value in parts.values() if isinstance(value, Series))
    output = Series("", index=first.index, dtype="string")
    for index, (name, value) in enumerate(parts.items()):
        values = value.astype("string") if isinstance(value, Series) else str(value)
        prefix = "" if index == 0 else "|"
        output = output + prefix + name + "=" + values
    return output


def aggregate_pair_cells(
    frame: DataFrame,
    *,
    route: str,
    cells: Series,
    horizon: int,
    minimum_coins: int = MIN_COINS,
) -> DataFrame:
    source = frame.assign(cell=cells)
    keys = ["cohort", "pair", "period", "control", "cell"]
    grouped = source.groupby(keys, observed=True, dropna=False)[list(OUTCOME_METRICS)]
    values = grouped.median()
    # A binary outcome must retain its actual event rate; its median collapses every
    # above-50% cell to 1.0 and every below-50% cell to 0.0.
    values["reaction_rate"] = source.groupby(
        keys, observed=True, dropna=False
    )["reaction_rate"].mean()
    medians = values.stack(future_stack=True).rename("pair_median")
    counts = grouped.count().stack(future_stack=True).rename("events")
    summary = pd.concat([medians, counts], axis=1).reset_index()
    summary = summary.rename(columns={f"level_{len(keys)}": "metric"})
    if "metric" not in summary:
        summary = summary.rename(columns={summary.columns[len(keys)]: "metric"})
    summary = summary.loc[summary["events"].gt(0)].copy()
    summary["route"] = route
    summary["horizon_hours"] = horizon
    summary["minimum_coins"] = minimum_coins
    return summary


def group_memberships(cohort: str, pair: str, meme_pairs: Sequence[str]) -> list[str]:
    if cohort == "meme":
        return ["frozen_top_ten_memes"] if pair in meme_pairs else []
    memberships = [group for group, members in NORMAL_GROUPS.items() if pair in members]
    memberships.append("full_normal_cohort")
    return memberships


def pair_neutral_summaries(
    frame: DataFrame,
    thresholds: DataFrame,
    *,
    meme_pairs: Sequence[str],
) -> DataFrame:
    enriched = enrich_predictor_states(frame, thresholds)
    summaries: list[DataFrame] = []
    for horizon in HORIZONS:
        horizon_rows = add_horizon_outcomes(enriched, horizon)
        source_tf = horizon_rows["source_timeframe"].astype("string")
        summaries.append(
            aggregate_pair_cells(
                horizon_rows,
                route="activity_displacement",
                cells=cell_id(
                    source_timeframe=source_tf,
                    activity=horizon_rows["activity_state"],
                ),
                horizon=horizon,
            )
        )
        summaries.append(
            aggregate_pair_cells(
                horizon_rows,
                route="trend_timeframe_interaction",
                cells=cell_id(
                    source_timeframe=source_tf,
                    trend_alignment=horizon_rows["trend_alignment"],
                    trend_strength=horizon_rows["trend_strength"],
                ),
                horizon=horizon,
            )
        )
        summaries.append(
            aggregate_pair_cells(
                horizon_rows,
                route="crypto_market_alignment",
                cells=cell_id(
                    source_timeframe=source_tf,
                    local_alignment=horizon_rows["trend_alignment"],
                    market_alignment=horizon_rows["market_alignment"],
                ),
                horizon=horizon,
            )
        )
        summaries.append(
            aggregate_pair_cells(
                horizon_rows,
                route="cluster_obstacle_geometry",
                cells=cell_id(
                    source_timeframe=source_tf,
                    cluster_state=horizon_rows["cluster_state"],
                ),
                horizon=horizon,
            )
        )
        summaries.append(
            aggregate_pair_cells(
                horizon_rows,
                route="timeframe_horizon_persistence",
                cells=cell_id(
                    source_timeframe=source_tf,
                    level_family=horizon_rows["level_family"],
                ),
                horizon=horizon,
            )
        )

        external_frames: list[DataFrame] = []
        for source, ready_column, regime_column in (
            ("gdelt", "news_gdelt_ready", "news_gdelt_regime"),
            ("btc_orderbook", "ob_btc_model_ready", "ob_btc_regime"),
        ):
            ready = horizon_rows[ready_column].fillna(False).astype(bool)
            external = horizon_rows.loc[ready].copy()
            if external.empty:
                continue
            external["external_source"] = source
            external["external_regime"] = external[regime_column].astype("string")
            external_frames.append(external)
        for external in external_frames:
            summaries.append(
                aggregate_pair_cells(
                    external,
                    route="external_calm_and_stress",
                    cells=cell_id(
                        source=external["external_source"],
                        regime=external["external_regime"],
                        local_activity=external["activity_state"],
                    ),
                    horizon=horizon,
                )
            )

        cohort = str(horizon_rows["cohort"].iloc[0])
        pair = str(horizon_rows["pair"].iloc[0])
        for group in group_memberships(cohort, pair, meme_pairs):
            minimum = (
                MIN_SUBGROUP_COINS
                if group in {"other_established_altcoins"}
                else MIN_COINS
            )
            summaries.append(
                aggregate_pair_cells(
                    horizon_rows,
                    route="market_group_portability",
                    cells=cell_id(
                        group=group,
                        source_timeframe=source_tf,
                    ),
                    horizon=horizon,
                    minimum_coins=minimum,
                )
            )
    return pd.concat(summaries, ignore_index=True, sort=False)


def equal_coin_control_effects(pair_summaries: DataFrame) -> DataFrame:
    keys = [
        "route",
        "cohort",
        "period",
        "cell",
        "horizon_hours",
        "metric",
        "minimum_coins",
    ]
    actual = pair_summaries.loc[pair_summaries["control"].eq("actual")].drop(
        columns="control"
    )
    rows: list[DataFrame] = []
    for control in CONTROLS:
        comparison = pair_summaries.loc[
            pair_summaries["control"].eq(control)
        ].drop(columns="control")
        paired = actual.merge(
            comparison,
            on=[*keys, "pair"],
            how="inner",
            suffixes=("_actual", "_control"),
            validate="one_to_one",
        )
        if paired.empty:
            continue
        paired["pair_difference"] = (
            paired["pair_median_actual"] - paired["pair_median_control"]
        )
        effect = (
            paired.groupby(keys, observed=True, dropna=False)
            .agg(
                equal_coin_actual=("pair_median_actual", "mean"),
                equal_coin_control=("pair_median_control", "mean"),
                mean_pair_difference=("pair_difference", "mean"),
                median_pair_difference=("pair_difference", "median"),
                events_actual=("events_actual", "sum"),
                events_control=("events_control", "sum"),
                coins=("pair", "nunique"),
            )
            .reset_index()
        )
        effect["comparison_control"] = control
        effect["effect_sign"] = np.sign(effect["mean_pair_difference"]).astype(int)
        effect["supported"] = (
            effect["events_actual"].ge(MIN_EVENTS)
            & effect["events_control"].ge(MIN_EVENTS)
            & effect["coins"].ge(effect["minimum_coins"])
        )
        rows.append(effect)
    return pd.concat(rows, ignore_index=True) if rows else DataFrame()


def repeated_neutral_candidates(effects: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    group_keys = ["route", "cohort", "cell", "metric", "minimum_coins"]
    for key, cell in effects.groupby(group_keys, observed=True, dropna=False):
        route, cohort, cell_name, metric, minimum_coins = key
        periods = set(VALIDATION_PERIODS[str(cohort)])
        for band, horizons in g11z.HORIZON_BANDS.items():
            required = {
                (period, horizon, control)
                for period in periods
                for horizon in horizons
                for control in CONTROLS
            }
            checks = cell.loc[
                cell["period"].isin(periods)
                & cell["horizon_hours"].isin(horizons)
                & cell["comparison_control"].isin(CONTROLS)
            ].copy()
            present = set(
                zip(
                    checks["period"],
                    checks["horizon_hours"],
                    checks["comparison_control"],
                    strict=False,
                )
            )
            signs = checks.loc[checks["supported"], "effect_sign"]
            candidate = (
                present == required
                and len(checks) == len(required)
                and checks["supported"].all()
                and signs.ne(0).all()
                and signs.nunique() == 1
            )
            rows.append(
                {
                    "route": route,
                    "cohort": cohort,
                    "cell": cell_name,
                    "metric": metric,
                    "horizon_band": band,
                    "horizons": ",".join(str(value) for value in horizons),
                    "minimum_coins": int(minimum_coins),
                    "candidate": bool(candidate),
                    "behaviour_sign": int(signs.iloc[0]) if candidate else 0,
                    "checks_required": len(required),
                    "checks_present": len(checks),
                    "checks_supported": int(checks["supported"].sum()),
                    "interpretation": (
                        "exploratory_control_resistant_repeated_relationship"
                        if candidate
                        else "mixed_control_period_horizon_or_support_result"
                    ),
                }
            )
    return DataFrame.from_records(rows)


def causal_anchor_contacts(frame: DataFrame) -> DataFrame:
    actual = frame.loc[frame["control"].eq("actual")].copy()
    actual["event_time"] = pd.to_datetime(actual["event_time"], utc=True, errors="coerce")
    actual["_distance"] = numeric(actual, "contact_close_distance_atr")
    actual["_score"] = numeric(actual, "level_score")
    actual = actual.sort_values(
        ["pair", "period", "source_timeframe", "event_time", "_distance", "_score"],
        ascending=[True, True, True, True, True, False],
        na_position="last",
    ).drop_duplicates(
        ["pair", "period", "source_timeframe", "event_time"], keep="first"
    )
    actual = actual.loc[numeric(actual, "approach_code").ne(0)].copy()
    keep = Series(False, index=actual.index)
    cooldown = pd.Timedelta(hours=COOLDOWN_HOURS)
    for _, group in actual.groupby(
        ["pair", "period", "source_timeframe"], observed=True, sort=False
    ):
        last: pd.Timestamp | None = None
        for index, timestamp in zip(group.index, group["event_time"], strict=True):
            if last is None or timestamp - last >= cooldown:
                keep.loc[index] = True
                last = timestamp
    return actual.loc[keep].drop(columns=["_distance", "_score"]).reset_index(drop=True)


def path_resolution(frame: DataFrame, horizon: int) -> Series:
    away = numeric(frame, "time_to_away_0_5atr")
    through = numeric(frame, "time_to_through_0_5atr")
    away_hit = away.le(horizon)
    through_hit = through.le(horizon)
    result = np.zeros(len(frame), dtype=np.int8)
    result[away_hit & (~through_hit | away.lt(through))] = -1
    result[through_hit & (~away_hit | through.lt(away))] = 1
    # Same-candle ties and unresolved paths intentionally remain zero and fail joint success.
    return Series(result, index=frame.index, dtype="int8")


def direction_episodes(
    frame: DataFrame,
    thresholds: DataFrame,
) -> DataFrame:
    anchors = enrich_predictor_states(causal_anchor_contacts(frame), thresholds)
    approach = numeric(anchors, "approach_code").fillna(0).astype("int8")
    local_two_path = direction_vote_to_path(anchors["local_two_vote"], approach)
    local_five_path = direction_vote_to_path(anchors["local_five_vote"], approach)
    market_path = direction_vote_to_path(anchors["market_vote"], approach)
    consensus = Series(0, index=anchors.index, dtype="int8")
    agree = local_five_path.ne(0) & local_five_path.eq(market_path)
    consensus.loc[agree] = local_five_path.loc[agree]
    active_consensus = consensus.where(anchors["activity_state"].eq("active"), 0).astype(
        "int8"
    )
    obstacle = consensus.copy()
    obstacle.loc[anchors["has_opposing_overlap"]] = -1

    rows: list[DataFrame] = []
    retained = [
        "cohort",
        "pair",
        "event_time",
        "period",
        "source_timeframe",
        "level_family",
        "level_name",
        "activity_state",
        "trend_strength",
        "trend_alignment",
        "market_alignment",
        "cluster_state",
    ]
    for horizon in HORIZONS:
        outcome = add_horizon_outcomes(anchors, horizon)
        episode = outcome[retained].copy()
        episode["horizon_hours"] = horizon
        episode["reaction"] = outcome["reaction_rate"].astype(bool)
        episode["resolution_code"] = path_resolution(outcome, horizon)
        episode["pred__always_through_approach_baseline"] = 1
        episode["pred__local_ema_and_return_slope_vote"] = local_two_path
        episode["pred__local_five_indicator_vote"] = local_five_path
        episode["pred__btc_eth_and_cohort_direction_vote"] = market_path
        episode["pred__local_and_crypto_market_consensus"] = consensus
        episode["pred__active_market_consensus_only"] = active_consensus
        episode["pred__opposing_obstacle_rejection_else_consensus"] = obstacle
        rows.append(episode)
    return pd.concat(rows, ignore_index=True, sort=False)


def add_development_majority(episodes: DataFrame) -> DataFrame:
    output = episodes.copy()
    development = Series(
        np.where(
            output["cohort"].eq("normal"),
            output["period"].eq("development"),
            output["period"].eq("meme_development"),
        ),
        index=output.index,
    )
    resolved = output.loc[development & output["resolution_code"].ne(0)]
    mapping: dict[tuple[str, str, int], int] = {}
    for key, cell in resolved.groupby(
        ["cohort", "source_timeframe", "horizon_hours"], observed=True
    ):
        vote = int(np.sign(cell["resolution_code"].sum()))
        mapping[(str(key[0]), str(key[1]), int(key[2]))] = vote if vote else 1
    keys = zip(
        output["cohort"].astype(str),
        output["source_timeframe"].astype(str),
        output["horizon_hours"].astype(int),
        strict=True,
    )
    output["pred__development_majority_path"] = Series(
        [mapping.get(key, 0) for key in keys], index=output.index, dtype="int8"
    )
    return output


def equal_coin_rate(frame: DataFrame, column: str) -> float:
    if frame.empty:
        return np.nan
    return float(frame.groupby("pair", observed=True)[column].mean().mean())


def block_bootstrap(
    frame: DataFrame,
    *,
    value_columns: Sequence[str],
    seed_key: str,
    samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, tuple[float, float, float]]:
    if frame.empty:
        return {column: (np.nan, np.nan, np.nan) for column in value_columns}
    source = frame.copy()
    source["week"] = pd.to_datetime(source["event_time"], utc=True).dt.floor("7D")
    blocks = (
        source.groupby(["pair", "week"], observed=True)[list(value_columns)]
        .agg(["sum", "size"])
    )
    by_pair: dict[str, list[tuple[np.ndarray, int]]] = {}
    for pair, group in blocks.groupby(level=0, observed=True):
        entries: list[tuple[np.ndarray, int]] = []
        for _, row in group.droplevel(0).iterrows():
            totals = np.array([float(row[(column, "sum")]) for column in value_columns])
            count = int(row[(value_columns[0], "size")])
            entries.append((totals, count))
        by_pair[str(pair)] = entries
    point = {column: equal_coin_rate(source, column) for column in value_columns}
    if sum(len(entries) for entries in by_pair.values()) < 2:
        return {column: (value, np.nan, np.nan) for column, value in point.items()}
    seed = int.from_bytes(hashlib.sha256(seed_key.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    draws = np.empty((samples, len(value_columns)), dtype=float)
    for sample in range(samples):
        coin_rates: list[np.ndarray] = []
        for entries in by_pair.values():
            positions = rng.integers(0, len(entries), len(entries))
            total = np.zeros(len(value_columns), dtype=float)
            count = 0
            for position in positions:
                values, rows = entries[int(position)]
                total += values
                count += rows
            coin_rates.append(total / count)
        draws[sample] = np.mean(coin_rates, axis=0)
    return {
        column: (
            point[column],
            float(np.quantile(draws[:, index], 0.025)),
            float(np.quantile(draws[:, index], 0.975)),
        )
        for index, column in enumerate(value_columns)
    }


def direction_scopes(episodes: DataFrame, meme_pairs: Sequence[str]) -> dict[str, DataFrame]:
    scopes: dict[str, DataFrame] = {
        "normal_full_cohort": episodes.loc[episodes["cohort"].eq("normal")],
        "meme_frozen_top_ten": episodes.loc[episodes["cohort"].eq("meme")],
    }
    for group, members in NORMAL_GROUPS.items():
        scopes[f"normal_{group}"] = episodes.loc[
            episodes["cohort"].eq("normal") & episodes["pair"].isin(members)
        ]
    scopes["meme_frozen_top_ten"] = episodes.loc[
        episodes["cohort"].eq("meme") & episodes["pair"].isin(meme_pairs)
    ]
    return {key: value for key, value in scopes.items() if not value.empty}


def score_direction_methods(
    episodes: DataFrame,
    *,
    meme_pairs: Sequence[str],
) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for scope_id, scope in direction_scopes(episodes, meme_pairs).items():
        cohort = str(scope["cohort"].iloc[0])
        allowed_periods = VALIDATION_PERIODS[cohort]
        scope_type = (
            "full_cohort"
            if scope_id in {"normal_full_cohort", "meme_frozen_top_ten"}
            else "rational_subgroup"
            if scope["pair"].nunique() >= MIN_SUBGROUP_COINS
            else "single_coin_audit"
        )
        for (source_tf, horizon, period), cell in scope.loc[
            scope["period"].isin(allowed_periods)
        ].groupby(
            ["source_timeframe", "horizon_hours", "period"], observed=True
        ):
            eligible = len(cell)
            for method in g11z.DIRECTION_METHODS:
                prediction_column = f"pred__{method}"
                called = cell.loc[cell[prediction_column].ne(0)].copy()
                if called.empty:
                    continue
                called["method_success"] = (
                    called["reaction"]
                    & called[prediction_column].eq(called["resolution_code"])
                ).astype(float)
                for baseline in BASELINE_METHODS:
                    baseline_column = f"pred__{baseline}"
                    called[f"success__{baseline}"] = (
                        called["reaction"]
                        & called[baseline_column].eq(called["resolution_code"])
                    ).astype(float)
                baseline_rates = {
                    baseline: equal_coin_rate(called, f"success__{baseline}")
                    for baseline in BASELINE_METHODS
                }
                strongest = max(
                    baseline_rates,
                    key=lambda name: (
                        -np.inf
                        if not np.isfinite(baseline_rates[name])
                        else baseline_rates[name]
                    ),
                )
                called["strongest_comparator_success"] = called[
                    f"success__{strongest}"
                ]
                called["success_margin"] = (
                    called["method_success"] - called["strongest_comparator_success"]
                )
                boot = block_bootstrap(
                    called,
                    value_columns=("method_success", "success_margin"),
                    seed_key=(
                        f"g11|{scope_id}|{method}|{source_tf}|{horizon}|{period}"
                    ),
                )
                point, lower, upper = boot["method_success"]
                margin, margin_lower, margin_upper = boot["success_margin"]
                rows.append(
                    {
                        "scope_id": scope_id,
                        "scope_type": scope_type,
                        "cohort": cohort,
                        "method": method,
                        "source_timeframe": source_tf,
                        "horizon_hours": int(horizon),
                        "period": period,
                        "eligible_events": eligible,
                        "issued_calls": len(called),
                        "coverage": len(called) / eligible,
                        "coins": called["pair"].nunique(),
                        "reaction_rate_on_calls": equal_coin_rate(
                            called.assign(reaction_float=called["reaction"].astype(float)),
                            "reaction_float",
                        ),
                        "joint_success_rate": point,
                        "joint_success_lower_95": lower,
                        "joint_success_upper_95": upper,
                        "strongest_comparator": strongest,
                        "strongest_comparator_rate": baseline_rates[strongest],
                        "comparator_margin": margin,
                        "comparator_margin_lower_95": margin_lower,
                        "comparator_margin_upper_95": margin_upper,
                    }
                )
    return DataFrame.from_records(rows)


def direction_candidates(scores: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["scope_id", "scope_type", "cohort", "method", "source_timeframe", "horizon_hours"]
    for key, cell in scores.groupby(keys, observed=True, dropna=False):
        scope_id, scope_type, cohort, method, source_tf, horizon = key
        periods = set(VALIDATION_PERIODS[str(cohort)])
        checks = cell.loc[cell["period"].isin(periods)]
        complete = set(checks["period"]) == periods and len(checks) == len(periods)
        eligible_method = method not in BASELINE_METHODS and scope_type != "single_coin_audit"
        point_pass = (
            complete
            and eligible_method
            and checks["joint_success_rate"].ge(0.55).all()
            and checks["issued_calls"].sum() >= 100
            and checks["coins"].ge(MIN_COINS).all()
            and checks["coverage"].ge(0.20).all()
            and checks["comparator_margin"].ge(0.02).all()
        )
        strict_pass = (
            point_pass
            and checks["joint_success_lower_95"].gt(0.50).all()
            and checks["comparator_margin_lower_95"].gt(0.0).all()
        )
        rows.append(
            {
                "scope_id": scope_id,
                "scope_type": scope_type,
                "cohort": cohort,
                "method": method,
                "source_timeframe": source_tf,
                "horizon_hours": int(horizon),
                "periods_complete": bool(complete),
                "issued_calls_total": int(checks["issued_calls"].sum()),
                "minimum_period_joint_success": (
                    float(checks["joint_success_rate"].min()) if complete else np.nan
                ),
                "minimum_period_coverage": (
                    float(checks["coverage"].min()) if complete else np.nan
                ),
                "minimum_period_coins": (
                    int(checks["coins"].min()) if complete else 0
                ),
                "minimum_comparator_margin": (
                    float(checks["comparator_margin"].min()) if complete else np.nan
                ),
                "point_candidate_55pct": bool(point_pass),
                "strict_candidate": bool(strict_pass),
                "status": (
                    "strict_exploratory_lead_needs_fresh_confirmation"
                    if strict_pass
                    else "point_only_exploratory_lead_needs_fresh_confirmation"
                    if point_pass
                    else "reference_baseline_not_a_combined_data_candidate"
                    if method in BASELINE_METHODS
                    else "did_not_clear_frozen_joint_gate"
                ),
            }
        )
    return DataFrame.from_records(rows)


def route_decisions(
    neutral_candidates: DataFrame,
    direction_candidate_frame: DataFrame,
) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for route in g11z.ROUTES:
        neutral = neutral_candidates.loc[
            neutral_candidates["route"].eq(route.route_id)
            & neutral_candidates["candidate"]
        ]
        point_direction = (
            direction_candidate_frame.loc[
                direction_candidate_frame["point_candidate_55pct"]
            ]
            if route.route_id
            in {"crypto_market_alignment", "market_group_portability"}
            else direction_candidate_frame.iloc[0:0]
        )
        rows.append(
            {
                "route_id": route.route_id,
                "family": route.family,
                "status": "completed_exploratory_direct_screen",
                "neutral_repeated_leads": len(neutral),
                "direction_point_leads": len(point_direction),
                "direction_strict_leads": int(point_direction["strict_candidate"].sum()),
                "next_step": (
                    "consider_in_joint_review_without_branching_yet"
                    if len(neutral) or len(point_direction)
                    else "park_direct_form_after_joint_review"
                ),
            }
        )
    return DataFrame.from_records(rows)


def load_pair_rows(path: Path, *, controls: Sequence[str], periods: Sequence[str]) -> DataFrame:
    frame = pd.read_parquet(
        path,
        columns=list(READ_COLUMNS),
        filters=[("control", "in", list(controls)), ("period", "in", list(periods))],
    )
    frame["event_time"] = pd.to_datetime(frame["event_time"], utc=True, errors="coerce")
    if frame["event_time"].isna().any():
        raise ValueError(f"Invalid event timestamps in {path}")
    return g6d.causal_deduplicate(
        frame,
        keys=(
            "cohort",
            "pair",
            "event_time",
            "source_timeframe",
            "level_family",
            "control",
        ),
    )


def run_direct_screen(run_id: str, *, overwrite: bool = False) -> dict[str, Any]:
    frozen, manifest, thresholds = validate_contracts()
    run_dir = DIRECT_ROOT / run_id
    result_path = run_dir / "g11_direct_screen_result.json"
    if result_path.is_file() and not overwrite:
        return json.loads(result_path.read_text(encoding="utf-8"))
    run_dir.mkdir(parents=True, exist_ok=True)
    large_run_dir = LARGE_DIRECT_ROOT / run_id
    large_run_dir.mkdir(parents=True, exist_ok=True)

    meme_pairs = tuple(
        str(task["pair"]) for task in manifest["tasks"] if task["cohort"] == "meme"
    )
    validation_names = tuple(
        period for values in VALIDATION_PERIODS.values() for period in values
    )
    pair_summaries: list[DataFrame] = []
    episode_frames: list[DataFrame] = []
    progress: list[dict[str, Any]] = []
    for task_number, task in enumerate(manifest["tasks"], start=1):
        path = Path(task["event_path"])
        neutral = load_pair_rows(
            path,
            controls=("actual", *CONTROLS),
            periods=validation_names,
        )
        pair_summary = pair_neutral_summaries(
            neutral, thresholds, meme_pairs=meme_pairs
        )
        pair_summaries.append(pair_summary)
        neutral_rows = len(neutral)
        del neutral, pair_summary
        gc.collect()

        cohort = str(task["cohort"])
        direction_periods = (
            (g11p.development_period(cohort), *VALIDATION_PERIODS[cohort])
        )
        direction_source = load_pair_rows(
            path,
            controls=("actual",),
            periods=direction_periods,
        )
        pair_episodes = direction_episodes(direction_source, thresholds)
        episode_frames.append(pair_episodes)
        progress.append(
            {
                "task": task_number,
                "cohort": cohort,
                "pair": task["pair"],
                "neutral_rows": neutral_rows,
                "direction_anchor_horizon_rows": len(pair_episodes),
            }
        )
        print(
            f"[{task_number:02d}/{len(manifest['tasks']):02d}] "
            f"{cohort} {task['pair']}: {neutral_rows:,} controlled rows, "
            f"{len(pair_episodes):,} direction-horizon rows",
            flush=True,
        )
        del direction_source, pair_episodes
        gc.collect()

    pair_summary_frame = pd.concat(pair_summaries, ignore_index=True, sort=False)
    pair_summary_path = large_run_dir / "g11_pair_condition_medians.parquet"
    g0.atomic_write_parquet(pair_summary_frame, pair_summary_path)
    effects = equal_coin_control_effects(pair_summary_frame)
    neutral_candidates = repeated_neutral_candidates(effects)
    del pair_summaries, pair_summary_frame
    gc.collect()

    episodes = add_development_majority(
        pd.concat(episode_frames, ignore_index=True, sort=False)
    )
    episode_path = large_run_dir / "g11_direction_episodes.parquet"
    g0.atomic_write_parquet(episodes, episode_path)
    direction_scores = score_direction_methods(episodes, meme_pairs=meme_pairs)
    direction_candidate_frame = direction_candidates(direction_scores)
    decisions = route_decisions(neutral_candidates, direction_candidate_frame)

    paths = {
        "neutral_effects": run_dir / "g11_direction_neutral_control_effects.csv",
        "neutral_candidates": run_dir / "g11_direction_neutral_candidates.csv",
        "direction_scores": run_dir / "g11_direction_method_scores.csv",
        "direction_candidates": run_dir / "g11_direction_candidates.csv",
        "route_decisions": run_dir / "g11_route_decisions.csv",
        "progress": run_dir / "g11_pair_processing_progress.csv",
    }
    frames = {
        "neutral_effects": effects,
        "neutral_candidates": neutral_candidates,
        "direction_scores": direction_scores,
        "direction_candidates": direction_candidate_frame,
        "route_decisions": decisions,
        "progress": DataFrame.from_records(progress),
    }
    for key, path in paths.items():
        g0.atomic_write_csv(frames[key], path)

    result = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at_utc": g0.utc_now(),
        "status": "completed_all_seven_generation11_direct_routes",
        "evidence_label": frozen["evidence_label"],
        "method": {
            "profit_used": False,
            "entry_or_exit_rules_built": False,
            "cohorts": list(g11z.COHORTS),
            "source_timeframes": list(g11z.SOURCE_TIMEFRAMES),
            "horizons_hours": list(HORIZONS),
            "controls": list(CONTROLS),
            "direction_methods": list(g11z.DIRECTION_METHODS),
            "independent_contact_cooldown_hours": COOLDOWN_HOURS,
            "reaction_definition": frozen["reaction_definition"],
            "direction_definition": frozen["direction_definition"],
            "direction_rate_weighting": "mean of per-coin rates",
            "uncertainty": (
                "deterministic within-coin seven-day block bootstrap; 384 samples"
            ),
            "fresh_confirmation_required": True,
        },
        "source_contracts": {
            "generation11_freeze": artifact(g11z.FREEZE_PATH),
            "generation11_preflight": artifact(PREFLIGHT_RECORD),
            "generation6_event_manifest": artifact(g11z.G6_EVENT_MANIFEST),
        },
        "artifacts": {
            **{key: artifact(path) for key, path in paths.items()},
            "pair_condition_medians": artifact(pair_summary_path),
            "direction_episodes": artifact(episode_path),
        },
        "summary": {
            "routes_completed": len(decisions),
            "event_source_tasks_completed": len(progress),
            "direction_episode_horizon_rows": len(episodes),
            "neutral_effect_rows": len(effects),
            "neutral_repeated_leads": int(neutral_candidates["candidate"].sum()),
            "direction_score_rows": len(direction_scores),
            "direction_point_leads_55pct": int(
                direction_candidate_frame["point_candidate_55pct"].sum()
            ),
            "direction_strict_leads": int(
                direction_candidate_frame["strict_candidate"].sum()
            ),
            "profit_used": False,
            "strategy_promotion": False,
        },
        "next_action": (
            "Run one joint review across all seven completed siblings. Only then freeze "
            "a balanced FreqAI descendant batch from supported, non-duplicative leads."
        ),
    }
    g0.atomic_write_json(result, result_path)
    return {**result, "result_path": str(result_path.resolve())}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all seven frozen Generation 11 broad direct screens."
    )
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    result = run_direct_screen(args.run_id, overwrite=args.overwrite)
    print(json.dumps(result["summary"], indent=2, sort_keys=True), flush=True)
    print(result.get("result_path", str((DIRECT_ROOT / args.run_id).resolve())), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
