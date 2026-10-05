from __future__ import annotations

import argparse
import json
import math
import re
import sys
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features.structure_orderbook_confluence_freqai_report import (  # noqa: E402
    DEFAULT_MODELS_DIR,
    _actual_frame,
    _load_predictions,
    _metrics,
    _score_predictions,
)
from user_data.Custom_Launcher.research.context_features.structure_orderbook_confluence_tests import (  # noqa: E402
    DEFAULT_ORDERBOOK,
    DEFAULT_STRUCTURAL,
    event_mask,
    load_frame,
)


DEFAULT_LEDGER = Path(__file__).resolve().parents[3] / "research_news_data" / "context_features" / "reports" / "freqai_research_results_ledger.csv"
DEFAULT_TRADER_CONFLUENCE = (
    Path(__file__).resolve().parents[3]
    / "research_news_data"
    / "context_features"
    / "confluence_cache"
    / "trader_confluence_1h_latest.parquet"
)
GENERAL_EXIT_STAGE1_RUNTIME = (
    Path(__file__).resolve().parents[2]
    / "launcher_v2"
    / "runtime"
    / "entry_sieve_target_quality_investigation_stage1"
)
GENERAL_EXIT_RAW_DATA = Path(__file__).resolve().parents[3] / "data" / "binance" / "futures"
GENERAL_EXIT_PAIR_FILES = {
    "ETH/USDT:USDT": GENERAL_EXIT_RAW_DATA / "ETH_USDT_USDT-1h-futures.feather",
    "SOL/USDT:USDT": GENERAL_EXIT_RAW_DATA / "SOL_USDT_USDT-1h-futures.feather",
}
GENERAL_EXIT_REACTION_EVENTS = (
    Path(__file__).resolve().parents[3]
    / "research_news_data"
    / "context_features"
    / "reports"
    / "general_exit_level_reaction"
    / "level_reaction_events.parquet"
)
GENERAL_EXIT_REACTION_PLACEBO_EVENTS = (
    Path(__file__).resolve().parents[3]
    / "research_news_data"
    / "context_features"
    / "reports"
    / "general_exit_level_reaction"
    / "level_reaction_placebo_events.parquet"
)
GENERAL_EXIT_COHORTS = {
    "bos_continuation_long": {
        "source": "overtrade_bos_bull_continuation_long_1h_vp_market_guard",
        "side": "long",
    },
    "d1_vp_bos_retest_short": {
        "source": "mtf_confluence_d1_vp_bos_4h_retest_short",
        "side": "short",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Score one queued FreqAI experiment and append to the research ledger.")
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--structural", type=Path, default=DEFAULT_STRUCTURAL)
    parser.add_argument("--orderbook", type=Path, default=DEFAULT_ORDERBOOK)
    args = parser.parse_args()

    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    experiment = next((item for item in queue.get("experiments", []) if item.get("id") == args.experiment_id), None)
    if experiment is None:
        raise ValueError(f"Experiment not found: {args.experiment_id}")
    rows = score_experiment(
        experiment,
        ledger_path=args.ledger,
        models_dir=args.models_dir,
        structural=args.structural,
        orderbook=args.orderbook,
    )
    print(json.dumps({"rows": len(rows), "ledger": str(args.ledger)}, indent=2))
    return 0


def score_experiment(
    experiment: dict[str, Any],
    *,
    ledger_path: Path,
    models_dir: Path,
    structural: Path,
    orderbook: Path,
) -> list[dict[str, Any]]:
    profile = dict(experiment.get("profile") or {})
    identifier = str(experiment["id"])
    freqai_identifier = str(experiment.get("freqai_identifier") or identifier)
    if str(profile.get("family") or "") == "general_exit_target_quality":
        rows = _score_general_exit_experiment(
            experiment,
            ledger_path=ledger_path,
            model_dir=models_dir / freqai_identifier,
        )
        return rows
    if str(profile.get("family") or "") in {
        "general_exit_level_reaction",
        "general_exit_readiness",
    }:
        return _score_general_exit_reaction_experiment(
            experiment,
            ledger_path=ledger_path,
            model_dir=models_dir / freqai_identifier,
        )
    predictions = _load_predictions(models_dir / freqai_identifier)
    if predictions.empty:
        rows = [_ledger_row(experiment, "missing_predictions")]
        _append_ledger(ledger_path, rows)
        return rows
    event_id = str(profile.get("event_id") or "vah_rejection")
    actual_column = str(profile.get("actual_column") or "breakout_failure_next_6h")
    prediction_column = str(profile.get("prediction_column") or "&-so_vah_rejection_breakout_failure_next_6h")
    actuals = _actuals_for_experiment(structural, orderbook, event_id=event_id, actual_column=actual_column)
    if str(profile.get("family") or "").startswith("trader_confluence_delta"):
        actuals = _append_trader_confluence_scope_columns(actuals)
    scored = _score_predictions(
        identifier,
        str(profile.get("profile_id") or ""),
        predictions,
        actuals,
        event_id=event_id,
        actual_column=actual_column,
        prediction_column=prediction_column,
    )
    if str(profile.get("family") or "").startswith("trader_confluence_delta"):
        scored.extend(
            _score_trader_confluence_scopes(
                predictions,
                actuals,
                actual_column=actual_column,
                prediction_column=prediction_column,
            )
        )
    rows = [_ledger_row(experiment, "scored", score=row) for row in scored]
    rows.extend(_control_comparison_rows(experiment, rows, ledger_path))
    _append_ledger(ledger_path, rows)
    return rows


def _actuals_for_experiment(structural: Path, orderbook: Path, *, event_id: str, actual_column: str) -> pd.DataFrame:
    try:
        actuals = _actual_frame(structural, orderbook, event_id=event_id, actual_column=actual_column)
    except KeyError:
        frame = load_frame(structural, orderbook)
        frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
        frame["event_active"] = event_mask(frame, event_id).astype(float)
        if "orderbook_present" not in frame:
            frame["orderbook_present"] = 0.0
        actuals = frame[["date", "event_active", "orderbook_present"]].dropna(subset=["date"])
    if actual_column not in actuals:
        actuals = _append_trader_confluence_actual_column(actuals, actual_column)
    if actual_column not in actuals:
        raise KeyError(f"{actual_column!r} is not available in structural/orderbook or trader-confluence actual frames")
    return actuals.dropna(subset=["date"])


def _append_trader_confluence_actual_column(actuals: pd.DataFrame, actual_column: str) -> pd.DataFrame:
    if not DEFAULT_TRADER_CONFLUENCE.exists():
        return actuals
    confluence = pd.read_parquet(DEFAULT_TRADER_CONFLUENCE, columns=["date", actual_column])
    if "date" not in confluence or actual_column not in confluence:
        return actuals
    confluence["date"] = pd.to_datetime(confluence["date"], utc=True, errors="coerce")
    confluence = confluence.dropna(subset=["date"]).drop_duplicates("date", keep="last")
    return actuals.merge(confluence[["date", actual_column]], on="date", how="left")


def _append_trader_confluence_scope_columns(actuals: pd.DataFrame) -> pd.DataFrame:
    if not DEFAULT_TRADER_CONFLUENCE.exists():
        return actuals
    confluence = pd.read_parquet(DEFAULT_TRADER_CONFLUENCE)
    if "date" not in confluence:
        return actuals
    confluence["date"] = pd.to_datetime(confluence["date"], utc=True, errors="coerce")
    scope_columns = [
        column
        for column in confluence.columns
        if column.startswith("conf_delta_") and column.endswith(("_setup", "_trigger"))
    ]
    if not scope_columns:
        return actuals
    scoped = confluence[["date", *scope_columns]].dropna(subset=["date"]).drop_duplicates("date", keep="last")
    return actuals.merge(scoped, on="date", how="left")


def _score_trader_confluence_scopes(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    *,
    actual_column: str,
    prediction_column: str,
) -> list[dict[str, Any]]:
    merged = predictions.merge(actuals, on="date", how="left")
    if "do_predict" in merged:
        merged = merged[pd.to_numeric(merged["do_predict"], errors="coerce").eq(1.0)].copy()
    if prediction_column not in merged:
        return []
    orderbook_source = merged.get("orderbook_present")
    if orderbook_source is None:
        orderbook_source = pd.Series(0.0, index=merged.index)
    orderbook_rows = pd.to_numeric(orderbook_source, errors="coerce").fillna(0.0).gt(0.0)
    rows: list[dict[str, Any]] = []
    scope_columns = [
        column
        for column in merged.columns
        if column.startswith("conf_delta_") and column.endswith(("_setup", "_trigger"))
    ]
    for column in sorted(scope_columns):
        base = column.removeprefix("conf_")
        active = pd.to_numeric(merged[column], errors="coerce").fillna(0.0).gt(0.0)
        rows.append(_score_scope(merged[active], f"{base}_only", actual_column, prediction_column))
        rows.append(_score_scope(merged[active & orderbook_rows], f"{base}_orderbook_present", actual_column, prediction_column))
    return rows


def _score_scope(scoped: pd.DataFrame, scope: str, actual_column: str, prediction_column: str) -> dict[str, Any]:
    valid = scoped[[prediction_column, actual_column]].apply(pd.to_numeric, errors="coerce").dropna()
    row: dict[str, Any] = {
        "scope": scope,
        "prediction_column": prediction_column,
        "status": "scored",
        "rows": int(len(valid)),
    }
    if valid.empty or valid[actual_column].nunique() < 2 or valid[prediction_column].nunique() < 2:
        row.update(
            {
                "actual_event_rate": float(valid[actual_column].mean()) if not valid.empty else None,
                "roc_auc": None,
                "average_precision": None,
                "top_quintile_actual_rate": None,
                "bottom_quintile_actual_rate": None,
            }
        )
        return row
    row.update(_metrics(valid, prediction_column, actual_column=actual_column))
    return row


def _ledger_row(experiment: dict[str, Any], status: str, *, score: dict[str, Any] | None = None) -> dict[str, Any]:
    profile = dict(experiment.get("profile") or {})
    score = score or {}
    return {
        "experiment_id": experiment.get("id"),
        "freqai_identifier": experiment.get("freqai_identifier"),
        "status": status if score.get("status") in {None, "scored"} else score.get("status"),
        "profile_id": profile.get("profile_id"),
        "family": profile.get("family"),
        "control_profile_id": profile.get("control_profile_id"),
        "window": experiment.get("window"),
        "timerange": experiment.get("timerange"),
        "scope": score.get("scope"),
        "pair": score.get("pair"),
        "target": score.get("target"),
        "orientation": score.get("orientation"),
        "rows": score.get("rows"),
        "actual_event_rate": score.get("actual_event_rate"),
        "roc_auc": score.get("roc_auc"),
        "average_precision": score.get("average_precision"),
        "prediction_actual_corr": score.get("prediction_actual_corr"),
        "top_quintile_actual_rate": score.get("top_quintile_actual_rate"),
        "bottom_quintile_actual_rate": score.get("bottom_quintile_actual_rate"),
        "top_bottom_lift": score.get("top_bottom_lift"),
        "normalized_lift": score.get("normalized_lift"),
        "spearman_rho": score.get("spearman_rho"),
        "spearman_p": score.get("spearman_p"),
        "rank_test_p": score.get("rank_test_p"),
        "quarter_count": score.get("quarter_count"),
        "quarter_positive_fraction": score.get("quarter_positive_fraction"),
        "objective": profile.get("objective"),
        "pass_rule": profile.get("pass_rule"),
    }


def _append_ledger(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    for column in frame.columns:
        frame[column] = frame[column].map(_clean_value)
    if path.exists():
        existing = pd.read_csv(path)
        frame = pd.concat([existing, frame], ignore_index=True)
    frame.to_csv(path, index=False)


def _control_comparison_rows(
    experiment: dict[str, Any],
    candidate_rows: list[dict[str, Any]],
    ledger_path: Path,
) -> list[dict[str, Any]]:
    profile = dict(experiment.get("profile") or {})
    control_profile_id = profile.get("control_profile_id")
    if not control_profile_id or not ledger_path.exists():
        return []
    existing = pd.read_csv(ledger_path)
    control = existing[
        (existing.get("profile_id") == control_profile_id)
        & (existing.get("window") == experiment.get("window"))
        & (existing.get("timerange") == experiment.get("timerange"))
        & (existing.get("status") == "scored")
    ].copy()
    if control.empty:
        return []
    comparisons: list[dict[str, Any]] = []
    general_exit = str(profile.get("family") or "") in {
        "general_exit_target_quality",
        "general_exit_level_reaction",
        "general_exit_readiness",
    }
    if general_exit:
        control_by_scope = {
            (
                str(getattr(row, "scope", "")),
                str(getattr(row, "pair", "")),
                str(getattr(row, "target", "")),
                str(getattr(row, "orientation", "")),
            ): row
            for row in control.itertuples(index=False)
        }
    else:
        control_by_scope = {str(row.scope): row for row in control.itertuples(index=False)}
    for candidate in candidate_rows:
        if candidate.get("status") != "scored":
            continue
        scope = str(candidate.get("scope") or "")
        control_key: Any = scope
        if general_exit:
            control_key = (
                scope,
                str(candidate.get("pair") or ""),
                str(candidate.get("target") or ""),
                str(candidate.get("orientation") or ""),
            )
        control_row = control_by_scope.get(control_key)
        if control_row is None:
            continue
        comparison = {
            **candidate,
            "status": "control_comparison",
            "control_profile_id": control_profile_id,
        }
        for metric in (
            "roc_auc",
            "average_precision",
            "prediction_actual_corr",
            "top_quintile_actual_rate",
            "bottom_quintile_actual_rate",
            "top_bottom_lift",
            "normalized_lift",
            "spearman_rho",
            "quarter_positive_fraction",
        ):
            comparison[f"{metric}_delta_vs_control"] = _numeric_or_none(candidate.get(metric)) - _numeric_or_none(getattr(control_row, metric, None))
        comparisons.append(comparison)
    return comparisons


def _score_general_exit_experiment(
    experiment: dict[str, Any], *, ledger_path: Path, model_dir: Path
) -> list[dict[str, Any]]:
    predictions = _load_general_exit_predictions(model_dir)
    if predictions.empty:
        rows = [_ledger_row(experiment, "missing_predictions")]
        _append_ledger(ledger_path, rows)
        return rows

    cohort_trades = _load_general_exit_cohort_trades()
    scored: list[dict[str, Any]] = []
    merged_by_pair: dict[str, DataFrame] = {}
    for pair, pair_predictions in predictions.groupby("pair", sort=True):
        actuals = _general_exit_actuals(pair)
        merged = pair_predictions.merge(actuals, on="date", how="left")
        if "do_predict" in merged:
            merged = merged[pd.to_numeric(merged["do_predict"], errors="coerce").eq(1.0)]
        merged_by_pair[pair] = merged
        scored.extend(_general_exit_scopes_for_pair(pair, merged, cohort_trades))

    if merged_by_pair:
        pooled = pd.concat(merged_by_pair.values(), ignore_index=True)
        scored.extend(_general_exit_scopes_for_pair("ALL", pooled, cohort_trades))

    rows = [_ledger_row(experiment, "scored", score=row) for row in scored]
    rows.extend(_control_comparison_rows(experiment, rows, ledger_path))
    _append_ledger(ledger_path, rows)
    return rows


def _score_general_exit_reaction_experiment(
    experiment: dict[str, Any], *, ledger_path: Path, model_dir: Path
) -> list[dict[str, Any]]:
    predictions = _load_general_exit_predictions(model_dir)
    if predictions.empty:
        rows = [_ledger_row(experiment, "missing_predictions")]
        _append_ledger(ledger_path, rows)
        return rows
    if not GENERAL_EXIT_REACTION_EVENTS.exists():
        rows = [_ledger_row(experiment, "missing_reaction_events")]
        _append_ledger(ledger_path, rows)
        return rows

    family = str((experiment.get("profile") or {}).get("family") or "")
    include_reaction_magnitude = family == "general_exit_readiness"
    target_kinds = [
        "down_vs_up_advantage",
        "future_volume_ratio",
        "future_volume_pressure",
    ]
    if include_reaction_magnitude:
        target_kinds.append("future_reaction_magnitude")

    events = pd.read_parquet(GENERAL_EXIT_REACTION_EVENTS)
    events["date"] = pd.to_datetime(events["date"], utc=True, errors="coerce")
    event_window = {
        "exit_dev": "development",
        "exit_holdout": "holdout",
    }.get(str(experiment.get("window") or ""), str(experiment.get("window") or ""))
    events = events[
        events["first_touch_after_4h"].fillna(False)
        & events["window"].eq(event_window)
    ].copy()
    placebo_events = DataFrame()
    if include_reaction_magnitude and GENERAL_EXIT_REACTION_PLACEBO_EVENTS.exists():
        placebo_events = pd.read_parquet(GENERAL_EXIT_REACTION_PLACEBO_EVENTS)
        placebo_events["date"] = pd.to_datetime(
            placebo_events["date"], utc=True, errors="coerce"
        )
        placebo_events = placebo_events[
            placebo_events["first_touch_after_4h"].fillna(False)
            & placebo_events["window"].eq(event_window)
        ].copy()
    pair_tokens = {"ETH/USDT:USDT": "ETH", "SOL/USDT:USDT": "SOL"}
    scored: list[dict[str, Any]] = []
    merged_by_pair: dict[str, DataFrame] = {}
    for pair, pair_predictions in predictions.groupby("pair", sort=True):
        actuals = _general_exit_reaction_actuals(
            pair, include_reaction_magnitude=include_reaction_magnitude
        )
        merged = pair_predictions.merge(actuals, on="date", how="left")
        if "do_predict" in merged:
            merged = merged[pd.to_numeric(merged["do_predict"], errors="coerce").eq(1.0)]
        merged_by_pair[pair] = merged
        pair_events = events[events["pair"].eq(pair_tokens[pair])]
        scored.extend(
            _general_exit_reaction_scopes(
                pair,
                merged,
                pair_events,
                target_kinds=target_kinds,
            )
        )
        if not placebo_events.empty:
            pair_placebos = placebo_events[
                placebo_events["pair"].eq(pair_tokens[pair])
            ]
            for shift_hours, shifted in pair_placebos.groupby(
                "placebo_shift_hours", sort=True
            ):
                scored.extend(
                    _general_exit_reaction_scopes(
                        pair,
                        merged,
                        shifted,
                        target_kinds=target_kinds,
                        scope_prefix=f"stale_{int(shift_hours)}h_",
                        include_all_rows=False,
                    )
                )

    if merged_by_pair:
        pooled = pd.concat(merged_by_pair.values(), ignore_index=True)
        scored.extend(
            _general_exit_reaction_scopes(
                "ALL", pooled, events, target_kinds=target_kinds
            )
        )
        if not placebo_events.empty:
            for shift_hours, shifted in placebo_events.groupby(
                "placebo_shift_hours", sort=True
            ):
                scored.extend(
                    _general_exit_reaction_scopes(
                        "ALL",
                        pooled,
                        shifted,
                        target_kinds=target_kinds,
                        scope_prefix=f"stale_{int(shift_hours)}h_",
                        include_all_rows=False,
                    )
                )

    rows = [_ledger_row(experiment, "scored", score=row) for row in scored]
    rows.extend(_control_comparison_rows(experiment, rows, ledger_path))
    _append_ledger(ledger_path, rows)
    return rows


def _general_exit_reaction_actuals(
    pair: str, *, include_reaction_magnitude: bool = False
) -> DataFrame:
    if include_reaction_magnitude:
        from user_data.strategies.GeneralExitReadinessFreqAIResearchStrategy import (
            build_exit_readiness_targets as target_builder,
        )
    else:
        from user_data.strategies.GeneralExitLevelReactionFreqAIResearchStrategy import (
            build_level_reaction_targets as target_builder,
        )

    raw = pd.read_feather(GENERAL_EXIT_PAIR_FILES[pair])
    raw["date"] = pd.to_datetime(raw["date"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    labelled = target_builder(raw)
    kinds = [
        "down_vs_up_advantage",
        "future_volume_ratio",
        "future_volume_pressure",
    ]
    if include_reaction_magnitude:
        kinds.append("future_reaction_magnitude")
    targets = [
        f"{kind}_{horizon}h"
        for kind in kinds
        for horizon in (1, 2, 3, 4)
    ]
    return DataFrame(
        {
            "date": labelled["date"],
            **{f"actual_{target}": labelled[f"&-{target}"] for target in targets},
        }
    )


def _general_exit_reaction_scopes(
    pair: str,
    merged: DataFrame,
    events: DataFrame,
    *,
    target_kinds: list[str] | tuple[str, ...] = (
        "down_vs_up_advantage",
        "future_volume_ratio",
        "future_volume_pressure",
    ),
    scope_prefix: str = "",
    include_all_rows: bool = True,
) -> list[dict[str, Any]]:
    if merged.empty:
        return []
    scopes: list[tuple[str, Series, str]] = []
    if include_all_rows:
        scopes.append(("all_rows", pd.Series(True, index=merged.index), "raw"))
    event_specs = (
        ("all_named_level_touches", pd.Series(True, index=events.index)),
        ("single_named_level_touches", events["event_kind"].eq("single_level")),
        (
            "same_timeframe_cross_type_cluster_touches",
            events["event_kind"].eq("same_timeframe_cross_type_cluster"),
        ),
        (
            "same_type_cross_timeframe_cluster_touches",
            events["event_kind"].eq("same_type_cross_timeframe_cluster"),
        ),
        (
            "mixed_type_and_timeframe_cluster_touches",
            events["event_kind"].eq("mixed_type_and_timeframe_cluster"),
        ),
        (
            "target_coin_and_btc_same_side_cluster_touches",
            pd.to_numeric(events["btc_same_side_cluster_size"], errors="coerce").ge(2.0),
        ),
    )
    for side, orientation in (("resistance", "long"), ("support", "short")):
        side_events = events["side"].eq(side)
        for name, event_mask in event_specs:
            selected = events.loc[
                side_events & event_mask.fillna(False), ["pair", "date"]
            ].dropna(subset=["date"])
            if selected.empty:
                continue
            if pair == "ALL":
                event_pair_map = {
                    "ETH": "ETH/USDT:USDT",
                    "SOL": "SOL/USDT:USDT",
                }
                selected = selected.assign(pair=selected["pair"].map(event_pair_map))
                post_keys = pd.MultiIndex.from_frame(selected[["pair", "date"]])
                pre_keys = pd.MultiIndex.from_arrays(
                    [selected["pair"], selected["date"] - pd.Timedelta(hours=1)]
                )
                merged_keys = pd.MultiIndex.from_frame(merged[["pair", "date"]])
                post_touch = pd.Series(merged_keys.isin(post_keys), index=merged.index)
                pre_touch = pd.Series(merged_keys.isin(pre_keys), index=merged.index)
            else:
                dates = selected["date"].unique()
                post_touch = merged["date"].isin(dates)
                pre_touch_dates = pd.DatetimeIndex(dates) - pd.Timedelta(hours=1)
                pre_touch = merged["date"].isin(pre_touch_dates)
            scopes.append(
                (f"{scope_prefix}post_touch_{side}_{name}", post_touch, orientation)
            )
            scopes.append(
                (
                    f"{scope_prefix}one_candle_before_{side}_{name}",
                    pre_touch,
                    orientation,
                )
            )

    rows: list[dict[str, Any]] = []
    for scope, mask, orientation in scopes:
        rows.extend(
            _score_general_exit_reaction_targets(
                pair,
                scope,
                orientation,
                merged[mask.fillna(False)].copy(),
                target_kinds=target_kinds,
            )
        )
    return rows


def _score_general_exit_reaction_targets(
    pair: str,
    scope: str,
    orientation: str,
    scoped: DataFrame,
    *,
    target_kinds: list[str] | tuple[str, ...] = (
        "down_vs_up_advantage",
        "future_volume_ratio",
        "future_volume_pressure",
    ),
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for kind in target_kinds:
        for horizon in (1, 2, 3, 4):
            target = f"{kind}_{horizon}h"
            prediction_column = f"&-{target}"
            actual_column = f"actual_{target}"
            if prediction_column not in scoped or actual_column not in scoped:
                rows.append(
                    {
                        "status": "missing_prediction_column",
                        "scope": scope,
                        "pair": pair,
                        "target": target,
                        "orientation": orientation,
                    }
                )
                continue
            valid = scoped[["date", prediction_column, actual_column]].copy()
            valid[prediction_column] = pd.to_numeric(
                valid[prediction_column], errors="coerce"
            )
            valid[actual_column] = pd.to_numeric(valid[actual_column], errors="coerce")
            valid = valid.dropna(subset=[prediction_column, actual_column])
            if orientation == "short" and kind in {
                "down_vs_up_advantage",
                "future_volume_pressure",
            }:
                valid[prediction_column] *= -1.0
                valid[actual_column] *= -1.0
            row = _general_exit_metrics(
                valid,
                prediction_column=prediction_column,
                actual_column=actual_column,
                is_binary=False,
            )
            row.update(
                {
                    "scope": scope,
                    "pair": pair,
                    "target": target,
                    "orientation": orientation,
                }
            )
            rows.append(row)
    return rows


def _load_general_exit_predictions(model_dir: Path) -> DataFrame:
    prediction_dir = model_dir / "backtesting_predictions"
    frames: list[DataFrame] = []
    pair_by_token = {"eth": "ETH/USDT:USDT", "sol": "SOL/USDT:USDT"}
    for path in sorted(prediction_dir.glob("*_prediction.feather")):
        match = re.search(r"(?:^|_)cb_([a-z0-9]+)_", path.stem.lower())
        if match is None:
            match = re.search(r"^([a-z0-9]+)_", path.stem.lower())
        pair = pair_by_token.get(match.group(1) if match else "")
        if pair is None:
            continue
        frame = pd.read_feather(path)
        frame["pair"] = pair
        frame["prediction_file"] = path.name
        frames.append(frame)
    if not frames:
        return DataFrame()
    predictions = pd.concat(frames, ignore_index=True)
    predictions["date"] = pd.to_datetime(predictions["date"], utc=True, errors="coerce")
    return (
        predictions.dropna(subset=["date"])
        .drop_duplicates(subset=["pair", "date"], keep="last")
        .sort_values(["pair", "date"])
        .reset_index(drop=True)
    )


def _general_exit_actuals(pair: str) -> DataFrame:
    from user_data.strategies.GeneralExitTargetQualityFreqAIResearchStrategy import (
        GeneralExitTargetQualityFreqAIResearchStrategy,
        build_general_exit_targets,
    )

    path = GENERAL_EXIT_PAIR_FILES[pair]
    raw = pd.read_feather(path)
    raw["date"] = pd.to_datetime(raw["date"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    labelled = build_general_exit_targets(raw)
    aligned = GeneralExitTargetQualityFreqAIResearchStrategy._aligned_cache(raw, pair)
    close = pd.to_numeric(raw["close"], errors="coerce").replace(0.0, np.nan)
    states: dict[str, dict[str, Any]] = {}
    for timeframe in ("1h", "4h", "1d"):
        _, state = GeneralExitTargetQualityFreqAIResearchStrategy._level_features(
            aligned, close, timeframe
        )
        states[timeframe] = state
    diagnostics = GeneralExitTargetQualityFreqAIResearchStrategy._aggregate_level_state(states)
    actuals = DataFrame(
        {
            "date": labelled["date"],
            "decision_date": labelled["date"] + pd.Timedelta(hours=1),
            **{
                f"actual_down_vs_up_advantage_{horizon}h": labelled[
                    f"&-down_vs_up_advantage_{horizon}h"
                ]
                for horizon in (2, 4, 8, 24, 48)
            },
            **{f"diag_{name}": values for name, values in diagnostics.items()},
        }
    )
    return actuals


def _general_exit_scopes_for_pair(
    pair: str, merged: DataFrame, cohort_trades: DataFrame
) -> list[dict[str, Any]]:
    if merged.empty:
        return []
    near_long = pd.to_numeric(
        merged["diag_nearest_resistance_pct"], errors="coerce"
    ).between(0.0, 0.01)
    near_short = pd.to_numeric(
        merged["diag_nearest_support_pct"], errors="coerce"
    ).between(0.0, 0.01)
    confluence_long = near_long & pd.to_numeric(
        merged["diag_resistance_confluence_50bp"], errors="coerce"
    ).ge(2.0)
    confluence_short = near_short & pd.to_numeric(
        merged["diag_support_confluence_50bp"], errors="coerce"
    ).ge(2.0)

    scopes: list[tuple[str, Series, str]] = [
        ("all_rows", pd.Series(True, index=merged.index), "down"),
        ("near_resistance", near_long, "long"),
        ("near_support", near_short, "short"),
        ("confluent_resistance", confluence_long, "long"),
        ("confluent_support", confluence_short, "short"),
    ]
    if pair != "ALL":
        for cohort_name, spec in GENERAL_EXIT_COHORTS.items():
            side = str(spec["side"])
            cohort_mask = _cohort_window_mask(
                merged,
                cohort_trades[
                    cohort_trades["cohort"].eq(cohort_name)
                    & cohort_trades["pair"].eq(pair)
                ],
            )
            cohort_mask &= near_long if side == "long" else near_short
            scopes.append((f"cohort_{cohort_name}", cohort_mask, side))

    rows: list[dict[str, Any]] = []
    for scope_name, mask, orientation in scopes:
        scoped = merged[mask.fillna(False)].copy()
        rows.extend(_score_general_exit_targets(pair, scope_name, orientation, scoped))
    return rows


def _score_general_exit_targets(
    pair: str, scope: str, orientation: str, scoped: DataFrame
) -> list[dict[str, Any]]:
    targets = tuple(
        (f"down_vs_up_advantage_{horizon}h", False)
        for horizon in (2, 4, 8, 24, 48)
    )
    rows: list[dict[str, Any]] = []
    for target, is_binary in targets:
        prediction_column = f"&-{target}"
        actual_column = f"actual_{target}"
        if prediction_column not in scoped or actual_column not in scoped:
            rows.append(
                {
                    "status": "missing_prediction_column",
                    "scope": scope,
                    "pair": pair,
                    "target": target,
                    "orientation": orientation,
                }
            )
            continue
        valid = scoped[["date", prediction_column, actual_column]].copy()
        valid[prediction_column] = pd.to_numeric(valid[prediction_column], errors="coerce")
        valid[actual_column] = pd.to_numeric(valid[actual_column], errors="coerce")
        valid = valid.dropna(subset=[prediction_column, actual_column])
        if orientation == "short":
            valid[prediction_column] *= -1.0
            valid[actual_column] = (
                1.0 - valid[actual_column]
                if is_binary
                else -valid[actual_column]
            )
        row = _general_exit_metrics(
            valid,
            prediction_column=prediction_column,
            actual_column=actual_column,
            is_binary=is_binary,
        )
        row.update(
            {
                "scope": scope,
                "pair": pair,
                "target": target,
                "orientation": orientation,
            }
        )
        rows.append(row)
    return rows


def _general_exit_metrics(
    valid: DataFrame, *, prediction_column: str, actual_column: str, is_binary: bool
) -> dict[str, Any]:
    row: dict[str, Any] = {"status": "scored", "rows": int(len(valid))}
    if len(valid) < 30 or valid[prediction_column].nunique() < 2 or valid[actual_column].nunique() < 2:
        row["status"] = "insufficient_rows"
        return row

    ranked = valid.sort_values(prediction_column)
    bucket_size = max(1, int(len(ranked) * 0.2))
    bottom = ranked.head(bucket_size)
    top = ranked.tail(bucket_size)
    top_mean = float(top[actual_column].mean())
    bottom_mean = float(bottom[actual_column].mean())
    lift = top_mean - bottom_mean
    actual_std = float(valid[actual_column].std(ddof=0))
    rho, rho_p = spearmanr(valid[prediction_column], valid[actual_column])
    row.update(
        {
            "actual_event_rate": float(valid[actual_column].mean()),
            "prediction_actual_corr": float(valid[prediction_column].corr(valid[actual_column])),
            "top_quintile_actual_rate": top_mean,
            "bottom_quintile_actual_rate": bottom_mean,
            "top_bottom_lift": lift,
            "normalized_lift": lift / actual_std if actual_std > 0 else None,
            "spearman_rho": float(rho),
            "spearman_p": float(rho_p),
        }
    )
    if is_binary:
        actual_binary = valid[actual_column].astype(int)
        row["roc_auc"] = float(roc_auc_score(actual_binary, valid[prediction_column]))
        row["average_precision"] = float(
            average_precision_score(actual_binary, valid[prediction_column])
        )
        positive = valid.loc[actual_binary.eq(1), prediction_column]
        negative = valid.loc[actual_binary.eq(0), prediction_column]
        row["rank_test_p"] = float(
            mannwhitneyu(positive, negative, alternative="greater").pvalue
        )

    quarter_scores: list[float] = []
    quarter_key = valid["date"].dt.tz_localize(None).dt.to_period("Q")
    for _, quarter in valid.groupby(quarter_key):
        if len(quarter) < 20 or quarter[actual_column].nunique() < 2:
            continue
        if is_binary:
            quarter_scores.append(
                float(roc_auc_score(quarter[actual_column].astype(int), quarter[prediction_column]))
                - 0.5
            )
        else:
            quarter_rho, _ = spearmanr(quarter[prediction_column], quarter[actual_column])
            if np.isfinite(quarter_rho):
                quarter_scores.append(float(quarter_rho))
    row["quarter_count"] = len(quarter_scores)
    row["quarter_positive_fraction"] = (
        float(np.mean(np.asarray(quarter_scores) > 0.0)) if quarter_scores else None
    )
    return row


def _load_general_exit_cohort_trades() -> DataFrame:
    rows: list[dict[str, Any]] = []
    for cohort, spec in GENERAL_EXIT_COHORTS.items():
        backtest = _find_stage1_backtest(str(spec["source"]))
        if backtest is None:
            continue
        with zipfile.ZipFile(backtest) as archive:
            result_name = next(
                name
                for name in archive.namelist()
                if name.endswith(".json") and "_config" not in name
            )
            payload = json.loads(archive.read(result_name))
        _, result = next(iter(payload.get("strategy", {}).items()))
        for trade in result.get("trades", []):
            pair = str(trade.get("pair") or "")
            if pair not in GENERAL_EXIT_PAIR_FILES:
                continue
            open_date = pd.to_datetime(trade.get("open_date"), utc=True, errors="coerce")
            if pd.isna(open_date):
                continue
            rows.append(
                {
                    "cohort": cohort,
                    "pair": pair,
                    "side": str(spec["side"]),
                    "open_date": open_date,
                    "window_end": open_date + pd.Timedelta(hours=336),
                }
            )
    return DataFrame(rows)


def _find_stage1_backtest(source: str) -> Path | None:
    result_dir = GENERAL_EXIT_STAGE1_RUNTIME / "results"
    candidates: list[tuple[float, Path]] = []
    expected = f"sieve3_V2_tq_va_edge_from_{source}"
    for result_file in result_dir.glob("*.jsonl"):
        for line in result_file.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("status") != "ok" or row.get("strategy") != expected:
                continue
            path = Path(str(row.get("backtest_file") or ""))
            if path.exists():
                candidates.append((result_file.stat().st_mtime, path))
    return max(candidates, default=(0.0, None), key=lambda item: item[0])[1]


def _cohort_window_mask(frame: DataFrame, trades: DataFrame) -> Series:
    mask = pd.Series(False, index=frame.index, dtype="bool")
    if trades.empty:
        return mask
    decision_date = pd.to_datetime(frame["decision_date"], utc=True, errors="coerce")
    for trade in trades.itertuples(index=False):
        mask |= decision_date.between(trade.open_date, trade.window_end, inclusive="both")
    return mask


def _numeric_or_none(value: Any) -> float:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return float("nan")
    return float(parsed)


def _clean_value(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
