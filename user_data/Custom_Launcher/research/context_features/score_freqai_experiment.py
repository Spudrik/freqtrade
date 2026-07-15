from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd


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
        "rows": score.get("rows"),
        "actual_event_rate": score.get("actual_event_rate"),
        "roc_auc": score.get("roc_auc"),
        "average_precision": score.get("average_precision"),
        "prediction_actual_corr": score.get("prediction_actual_corr"),
        "top_quintile_actual_rate": score.get("top_quintile_actual_rate"),
        "bottom_quintile_actual_rate": score.get("bottom_quintile_actual_rate"),
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
    control_by_scope = {str(row.scope): row for row in control.itertuples(index=False)}
    for candidate in candidate_rows:
        if candidate.get("status") != "scored":
            continue
        scope = str(candidate.get("scope") or "")
        control_row = control_by_scope.get(scope)
        if control_row is None:
            continue
        comparison = {
            **candidate,
            "status": "control_comparison",
            "control_profile_id": control_profile_id,
        }
        for metric in ("roc_auc", "average_precision", "prediction_actual_corr", "top_quintile_actual_rate", "bottom_quintile_actual_rate"):
            comparison[f"{metric}_delta_vs_control"] = _numeric_or_none(candidate.get(metric)) - _numeric_or_none(getattr(control_row, metric, None))
        comparisons.append(comparison)
    return comparisons


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
