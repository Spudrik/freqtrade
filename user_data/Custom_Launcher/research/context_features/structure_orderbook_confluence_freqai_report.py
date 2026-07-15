from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from user_data.Custom_Launcher.research.context_features.structure_orderbook_confluence_tests import (  # noqa: E402
    DEFAULT_ORDERBOOK,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_STRUCTURAL,
    event_mask,
    load_frame,
)


USER_DATA_DIR = Path(__file__).resolve().parents[3]
DEFAULT_MODELS_DIR = USER_DATA_DIR / "models"
PREDICTION_COLUMN = "&-so_vah_rejection_breakout_failure_next_6h"
ACTUAL_COLUMN = "breakout_failure_next_6h"
EVENT_ID = "vah_rejection"


def main() -> int:
    parser = argparse.ArgumentParser(description="Score structure/orderbook confluence FreqAI prediction files.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--structural", type=Path, default=DEFAULT_STRUCTURAL)
    parser.add_argument("--orderbook", type=Path, default=DEFAULT_ORDERBOOK)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--tag", default="structure_orderbook_freqai")
    parser.add_argument("--event-id", default="")
    parser.add_argument("--actual-column", default="")
    parser.add_argument("--prediction-column", default="")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    event_id = args.event_id or str(manifest.get("event_id") or EVENT_ID)
    actual_column = args.actual_column or str(manifest.get("actual_column") or ACTUAL_COLUMN)
    prediction_column = args.prediction_column or str(manifest.get("prediction_column") or PREDICTION_COLUMN)
    actuals = _actual_frame(args.structural, args.orderbook, event_id=event_id, actual_column=actual_column)
    rows: list[dict[str, Any]] = []
    for command in manifest.get("commands", []):
        identifier = str(command.get("identifier") or "")
        model = str(command.get("model") or "")
        if not identifier:
            continue
        predictions = _load_predictions(args.models_dir / identifier)
        if predictions.empty:
            rows.append({"identifier": identifier, "model": model, "status": "missing_predictions"})
            continue
        rows.extend(
            _score_predictions(
                identifier,
                model,
                predictions,
                actuals,
                event_id=event_id,
                actual_column=actual_column,
                prediction_column=prediction_column,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    safe_tag = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in args.tag) or "latest"
    output = args.output_dir / f"structure_orderbook_confluence_freqai_{safe_tag}_metrics.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print(json.dumps({"metrics": str(output), "rows": len(rows)}, indent=2))
    return 0


def _actual_frame(structural: Path, orderbook: Path, *, event_id: str, actual_column: str) -> pd.DataFrame:
    frame = load_frame(structural, orderbook)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame["event_active"] = event_mask(frame, event_id).astype(float)
    if "orderbook_present" not in frame:
        frame["orderbook_present"] = 0.0
    return frame[["date", "event_active", "orderbook_present", actual_column]].dropna(subset=["date"])


def _load_predictions(model_dir: Path) -> pd.DataFrame:
    prediction_dir = model_dir / "backtesting_predictions"
    files = sorted(prediction_dir.glob("*_prediction.feather"))
    if not files:
        return pd.DataFrame()
    frames = []
    for path in files:
        frame = pd.read_feather(path)
        frame["prediction_file"] = path.name
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)
    predictions["date"] = pd.to_datetime(predictions["date"], utc=True, errors="coerce")
    predictions = predictions.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
    return predictions.sort_values("date").reset_index(drop=True)


def _score_predictions(
    identifier: str,
    model: str,
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    *,
    event_id: str,
    actual_column: str,
    prediction_column: str,
) -> list[dict[str, Any]]:
    merged = predictions.merge(actuals, on="date", how="left")
    if "do_predict" in merged:
        merged = merged[pd.to_numeric(merged["do_predict"], errors="coerce").eq(1.0)].copy()
    rows = []
    event_rows = pd.to_numeric(merged["event_active"], errors="coerce").fillna(0.0).gt(0.0)
    orderbook_rows = pd.to_numeric(merged["orderbook_present"], errors="coerce").fillna(0.0).gt(0.0)
    for scope_name, scoped in (
        ("all_predicted_rows", merged),
        (f"{event_id}_only", merged[event_rows]),
        (f"{event_id}_orderbook_present", merged[event_rows & orderbook_rows]),
    ):
        if prediction_column not in scoped:
            rows.append(
                {
                    "identifier": identifier,
                    "model": model,
                    "scope": scope_name,
                    "prediction_column": prediction_column,
                    "status": "missing_prediction_column",
                }
            )
            continue
        valid = scoped[[prediction_column, actual_column]].apply(pd.to_numeric, errors="coerce").dropna()
        row: dict[str, Any] = {
            "identifier": identifier,
            "model": model,
            "scope": scope_name,
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
        else:
            row.update(_metrics(valid, prediction_column, actual_column=actual_column))
        rows.append(row)
    return rows


def _metrics(valid: pd.DataFrame, prediction_column: str, *, actual_column: str) -> dict[str, float]:
    ranked = valid.sort_values(prediction_column)
    bucket_size = max(1, int(len(ranked) * 0.2))
    bottom = ranked.head(bucket_size)
    top = ranked.tail(bucket_size)
    binary_actual = set(valid[actual_column].dropna().astype(float).unique()).issubset({0.0, 1.0})
    roc_auc = None
    average_precision = None
    if binary_actual:
        roc_auc = float(roc_auc_score(valid[actual_column].astype(int), valid[prediction_column]))
        average_precision = float(average_precision_score(valid[actual_column].astype(int), valid[prediction_column]))
    return {
        "actual_event_rate": float(valid[actual_column].mean()),
        "roc_auc": roc_auc,
        "average_precision": average_precision,
        "prediction_actual_corr": float(valid[prediction_column].corr(valid[actual_column])),
        "top_quintile_actual_rate": float(top[actual_column].mean()),
        "bottom_quintile_actual_rate": float(bottom[actual_column].mean()),
    }


if __name__ == "__main__":
    raise SystemExit(main())
