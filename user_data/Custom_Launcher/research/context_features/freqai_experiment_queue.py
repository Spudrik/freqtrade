from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.Custom_Launcher.research.context_features.feature_profile_registry import (  # noqa: E402
    available_profiles,
    missing_profile_requirements,
)


USER_DATA_DIR = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = USER_DATA_DIR / "configs" / "config_context_freqai_research.example.json"
DEFAULT_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
DEFAULT_QUEUE_ROOT = USER_DATA_DIR / "research_news_data" / "context_features" / "freqai_queue"
DEFAULT_LEDGER = USER_DATA_DIR / "research_news_data" / "context_features" / "reports" / "freqai_research_results_ledger.csv"
DEFAULT_TRADER_CONFLUENCE = USER_DATA_DIR / "research_news_data" / "context_features" / "confluence_cache" / "trader_confluence_1h_latest.parquet"
WINDOWS = {
    "broad": "20250701-20260218",
    "jul_aug": "20250701-20250815",
    "oct": "20251001-20251115",
    "jan_feb": "20260101-20260218",
    "ob_jul_aug": "20250726-20250812",
    "ob_oct": "20251016-20251101",
    "ob_jan_feb": "20260130-20260216",
    "spot_full": "20250701-20260520",
    "spot_q4_2025": "20251001-20251231",
    "spot_q1_2026": "20260101-20260331",
    "spot_recent": "20260301-20260520",
    "spot_aug_2025": "20250801-20250831",
    "spot_sep_2025": "20250901-20250930",
    "spot_oct_2025": "20251001-20251031",
    "spot_nov_2025": "20251101-20251130",
    "spot_dec_2025": "20251201-20251231",
    "spot_jan_2026": "20260101-20260131",
    "spot_feb_2026": "20260201-20260228",
    "spot_mar_2026": "20260301-20260331",
    "spot_apr_2026": "20260401-20260430",
    "exit_dev": "20240401-20250331",
    "exit_holdout": "20250401-20260401",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a guarded FreqAI research experiment queue.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--python-exe", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--queue-root", type=Path, default=DEFAULT_QUEUE_ROOT)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument(
        "--profiles",
        default="structure_vp_tree,structure_vp_orderbook_tree,structure_vp_ridge,structure_vp_orderbook_ridge",
    )
    parser.add_argument("--windows", default="broad,jul_aug,oct,jan_feb,ob_jul_aug,ob_oct,ob_jan_feb")
    parser.add_argument("--train-days", type=int, default=120)
    parser.add_argument("--backtest-days", type=int, default=14)
    parser.add_argument("--plot-feature-importances", type=int, default=0)
    args = parser.parse_args()

    queue_path = create_queue(
        config_path=args.config,
        python_exe=args.python_exe,
        queue_root=args.queue_root,
        ledger_path=args.ledger,
        profile_ids=_split(args.profiles),
        window_ids=_split(args.windows),
        train_days=int(args.train_days),
        backtest_days=int(args.backtest_days),
        plot_feature_importances=int(args.plot_feature_importances),
    )
    print(json.dumps({"queue": str(queue_path)}, indent=2))
    return 0


def create_queue(
    *,
    config_path: Path,
    python_exe: Path,
    queue_root: Path,
    ledger_path: Path,
    profile_ids: list[str],
    window_ids: list[str],
    train_days: int,
    backtest_days: int,
    plot_feature_importances: int,
) -> Path:
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not python_exe.exists():
        raise FileNotFoundError(python_exe)
    profiles = available_profiles()
    missing = missing_profile_requirements()
    if profile_ids == ["all"]:
        profile_ids = sorted(profiles)
    unknown = [profile_id for profile_id in profile_ids if profile_id not in profiles]
    if unknown:
        raise ValueError(f"Unavailable profile(s): {unknown}; missing requirements: {missing}")
    invalid_windows = [window_id for window_id in window_ids if window_id not in WINDOWS]
    if invalid_windows:
        raise ValueError(f"Unknown window(s): {invalid_windows}")

    created_at = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_root = queue_root / f"queue_{created_at}"
    run_root.mkdir(parents=True, exist_ok=True)
    experiment_specs = _eligible_experiment_specs(profiles, profile_ids, window_ids, created_at)
    if not experiment_specs:
        raise ValueError("No experiments matched the selected profiles/windows after allowed_windows filtering.")
    selected_profiles = [spec["profile"] for spec in experiment_specs]
    validation_errors = _feature_validation_errors(selected_profiles)
    preflight = _preflight_audit(experiment_specs)
    if validation_errors:
        preflight.setdefault("errors", []).append({"scope": "feature_validation", "detail": validation_errors})
    preflight_path = run_root / "preflight_audit.json"
    preflight_path.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if preflight.get("errors"):
        raise ValueError(f"Feature cache preflight failed; see {preflight_path}: {preflight['errors'][:5]}")
    base_config = json.loads(config_path.read_text(encoding="utf-8"))
    experiments: list[dict[str, Any]] = []
    for index, spec in enumerate(experiment_specs, start=1):
        window_id = str(spec["window_id"])
        timerange = str(spec["timerange"])
        profile = spec["profile"]
        experiment_id = str(spec["experiment_id"])
        fs_slug = f"exp_{index:04d}"
        freqai_identifier = f"{run_root.name}_{fs_slug}"
        experiment_dir = run_root / fs_slug
        experiment_dir.mkdir(parents=True, exist_ok=True)
        experiment_train_days = profile.default_train_days or train_days
        experiment_backtest_days = profile.default_backtest_days or backtest_days
        cfg = _run_config(
            base_config,
            identifier=freqai_identifier,
            train_days=experiment_train_days,
            backtest_days=experiment_backtest_days,
            plot_feature_importances=plot_feature_importances,
            training_parameters=profile.model_training_parameters,
        )
        config_out = experiment_dir / "config.json"
        config_out.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        command = [
            str(python_exe),
            "-m",
            "freqtrade",
            "backtesting",
            "--userdir",
            str(USER_DATA_DIR),
            "--config",
            str(config_out),
            "--strategy",
            profile.strategy,
            "--freqaimodel",
            profile.freqaimodel,
            "--timerange",
            timerange,
            "--export",
            "signals",
            "--export-directory",
            str(experiment_dir),
        ]
        experiments.append(
            {
                "id": experiment_id,
                "fs_slug": fs_slug,
                "freqai_identifier": freqai_identifier,
                "status": "pending",
                "profile": profile.to_dict(),
                "window": window_id,
                "timerange": timerange,
                "train_days": experiment_train_days,
                "backtest_days": experiment_backtest_days,
                "run_dir": str(experiment_dir),
                "config_path": str(config_out),
                "command": command,
                "created_at": datetime.now().astimezone().isoformat(),
            }
        )
    queue = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "pending",
        "queue_root": str(run_root),
        "ledger_path": str(ledger_path),
        "preflight_audit_path": str(preflight_path),
        "guardrails": [
            "Do not run if another freqtrade/FreqAI job is active.",
            "Score candidate profiles against their stated control where possible.",
            "Treat event-specific hypotheses on event rows, and orderbook hypotheses on orderbook-present rows.",
            "Do not judge by profit.",
        ],
        "experiments": experiments,
    }
    queue_path = run_root / "freqai_experiment_queue.json"
    queue_path.write_text(json.dumps(queue, indent=2) + "\n", encoding="utf-8")
    return queue_path


def _run_config(
    base: dict[str, Any],
    *,
    identifier: str,
    train_days: int,
    backtest_days: int,
    plot_feature_importances: int,
    training_parameters: dict[str, Any],
) -> dict[str, Any]:
    config = json.loads(json.dumps(base))
    config["bot_name"] = "freqai_research_queue"
    freqai = config.setdefault("freqai", {})
    freqai["identifier"] = identifier
    freqai["train_period_days"] = train_days
    freqai["backtest_period_days"] = backtest_days
    feature_parameters = freqai.setdefault("feature_parameters", {})
    feature_parameters["label_period_candles"] = 6
    feature_parameters["indicator_periods_candles"] = [3, 6, 12, 24]
    feature_parameters["plot_feature_importances"] = plot_feature_importances
    freqai["model_training_parameters"] = dict(training_parameters)
    return config


def _split(raw: str) -> list[str]:
    return [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]


def _eligible_experiment_specs(
    profiles: dict[str, Any],
    profile_ids: list[str],
    window_ids: list[str],
    created_at: str,
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for window_id in window_ids:
        timerange = WINDOWS[window_id]
        for profile_id in profile_ids:
            profile = profiles[profile_id]
            if profile.allowed_windows and window_id not in profile.allowed_windows:
                continue
            specs.append(
                {
                    "profile": profile,
                    "window_id": window_id,
                    "timerange": timerange,
                    "experiment_id": f"{profile.profile_id}-{window_id}-{timerange}-{created_at}",
                }
            )
    return specs


def _feature_validation_errors(profiles: list[Any]) -> dict[str, list[str]]:
    checked: dict[Path, list[str]] = {}
    for profile in profiles:
        for path in profile.required_files:
            if path in checked:
                continue
            errors = []
            if path.suffix.lower() in {".sqlite", ".db", ".sqlite3"}:
                checked[path] = _sqlite_feature_validation_errors(path)
                continue
            try:
                frame = pd.read_parquet(path)
            except Exception as exc:
                checked[path] = [f"failed to read parquet: {exc}"]
                continue
            if "date" not in frame.columns:
                errors.append("missing date column")
            else:
                dates = pd.to_datetime(frame["date"], utc=True, errors="coerce")
                if dates.isna().any():
                    errors.append("contains null/unparseable dates")
                if dates.duplicated().any():
                    errors.append("contains duplicate dates")
            numeric = frame.select_dtypes(include=["number", "bool"])
            all_null = [str(column) for column in numeric.columns if frame[column].isna().all()]
            if all_null:
                errors.append(f"all-null numeric columns: {all_null[:8]}")
            checked[path] = errors
    return {str(path): errors for path, errors in checked.items() if errors}


def _preflight_audit(experiment_specs: list[dict[str, Any]]) -> dict[str, Any]:
    windows = {
        str(spec["window_id"]): _timerange_bounds(str(spec["timerange"]))
        for spec in experiment_specs
    }
    required_paths: dict[Path, dict[str, set[str]]] = {}
    for spec in experiment_specs:
        profile = spec["profile"]
        window_id = str(spec["window_id"])
        for path in profile.required_files:
            entry = required_paths.setdefault(path, {"profiles": set(), "windows": set()})
            entry["profiles"].add(str(profile.profile_id))
            entry["windows"].add(window_id)
    audit: dict[str, Any] = {
        "schema_version": 1,
        "generated_at": datetime.now().astimezone().isoformat(),
        "min_orderbook_coverage_ratio": 0.50,
        "windows": {
            window_id: {
                "timerange": WINDOWS[window_id],
                "start": start.isoformat(),
                "end": end.isoformat(),
            }
            for window_id, (start, end) in windows.items()
        },
        "files": [],
        "errors": [],
        "warnings": [],
    }
    for path, usage in sorted(required_paths.items(), key=lambda item: str(item[0])):
        path_windows = {
            window_id: windows[window_id]
            for window_id in sorted(usage["windows"])
        }
        if path.suffix.lower() in {".sqlite", ".db", ".sqlite3"}:
            file_audit = _audit_sqlite_context_file(path, path_windows)
        else:
            file_audit = _audit_parquet_feature_file(path, path_windows)
        file_audit["profiles"] = sorted(usage["profiles"])
        file_audit["windows_checked"] = sorted(usage["windows"])
        audit["files"].append(file_audit)
        for error in file_audit.get("errors", []):
            audit["errors"].append({"path": str(path), "detail": error})
        for warning in file_audit.get("warnings", []):
            audit["warnings"].append({"path": str(path), "detail": warning})
    balance = _target_balance_audit(experiment_specs, windows)
    audit["target_balance"] = balance
    for error in balance.get("errors", []):
        audit["errors"].append({"scope": "target_balance", "detail": error})
    for warning in balance.get("warnings", []):
        audit["warnings"].append({"scope": "target_balance", "detail": warning})
    return audit


def _timerange_bounds(timerange: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start_raw, end_raw = timerange.split("-", 1)
    start = pd.Timestamp(datetime.strptime(start_raw, "%Y%m%d"), tz="UTC")
    end = pd.Timestamp(datetime.strptime(end_raw, "%Y%m%d"), tz="UTC")
    return start, end


def _audit_parquet_feature_file(path: Path, windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]]) -> dict[str, Any]:
    audit: dict[str, Any] = {"path": str(path), "kind": "parquet", "errors": [], "warnings": []}
    if not path.exists():
        audit["errors"].append("missing file")
        return audit
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        audit["errors"].append(f"failed to read parquet: {exc}")
        return audit
    audit["rows"] = int(len(frame))
    audit["columns"] = int(len(frame.columns))
    if "date" not in frame:
        audit["errors"].append("missing date column")
        return audit
    dates = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    valid_dates = dates.dropna()
    audit["null_dates"] = int(dates.isna().sum())
    audit["duplicate_dates"] = int(dates.duplicated().sum())
    if valid_dates.empty:
        audit["errors"].append("no valid dates")
        return audit
    min_date = valid_dates.min()
    max_date = valid_dates.max()
    audit["min_date"] = min_date.isoformat()
    audit["max_date"] = max_date.isoformat()
    sorted_dates = valid_dates.sort_values()
    gaps = sorted_dates.diff().dropna()
    audit["gaps_over_1h"] = int(gaps.gt(pd.Timedelta(hours=1)).sum())
    source_timestamp_column = next(
        (column for column in ("source_max_ts", "max_source_available_at") if column in frame),
        None,
    )
    source_max = None
    if source_timestamp_column is not None:
        source_max = pd.to_datetime(frame[source_timestamp_column], utc=True, errors="coerce")
        audit["source_timestamp_column"] = source_timestamp_column
        audit["source_timestamp_null_rows"] = int(source_max.isna().sum())
        audit["source_after_date_rows"] = int(source_max.gt(dates).sum())
        if audit["source_after_date_rows"]:
            audit["errors"].append(
                f"{source_timestamp_column} after date rows: {audit['source_after_date_rows']}"
            )
        if audit["source_timestamp_null_rows"]:
            audit["warnings"].append(
                f"{source_timestamp_column} null rows: {audit['source_timestamp_null_rows']}"
            )
    if "obts_coverage_ratio" in frame:
        coverage = pd.to_numeric(frame["obts_coverage_ratio"], errors="coerce")
        if "obts_feature_present" in frame:
            base_present = pd.to_numeric(frame["obts_feature_present"], errors="coerce").fillna(0.0).gt(0.0)
        else:
            base_present = coverage.notna()
        usable_orderbook = base_present & coverage.fillna(0.0).ge(0.50)
        audit["orderbook_coverage"] = {
            "nonnull_rows": int(coverage.notna().sum()),
            "min": _float_or_none(coverage.min()),
            "median": _float_or_none(coverage.median()),
            "below_0_50_rows": int(coverage.fillna(0.0).lt(0.50).sum()),
            "usable_rows": int(usable_orderbook.sum()),
        }
        if audit["orderbook_coverage"]["below_0_50_rows"]:
            audit["warnings"].append(f"orderbook rows below 0.50 coverage: {audit['orderbook_coverage']['below_0_50_rows']}")
    else:
        usable_orderbook = None
    audit["window_coverage"] = {}
    for window_id, (start, end) in windows.items():
        window_mask = dates.ge(start) & dates.le(end)
        covers = bool(min_date <= start and max_date >= end)
        window_audit = {
            "covers": covers,
            "starts_before_or_at_window": bool(min_date <= start),
            "ends_after_or_at_window": bool(max_date >= end),
            "rows": int(window_mask.sum()),
        }
        if source_max is not None:
            window_source = source_max[window_mask]
            source_age_hours = (dates[window_mask].reset_index(drop=True) - window_source.reset_index(drop=True)).dt.total_seconds() / 3600.0
            stale_source_rows = int(source_age_hours.gt(48.0).sum())
            null_source_rows = int(window_source.isna().sum())
            window_audit["source_timestamp_null_rows"] = null_source_rows
            window_audit["source_age_max_hours"] = _float_or_none(source_age_hours.max())
            window_audit["source_age_over_48h_rows"] = stale_source_rows
            if null_source_rows:
                audit["warnings"].append(f"{source_timestamp_column} null rows in window {window_id}: {null_source_rows}")
            if stale_source_rows:
                audit["warnings"].append(f"{source_timestamp_column} older than 48h in window {window_id}: {stale_source_rows}")
        if usable_orderbook is not None:
            usable_rows = int((window_mask & usable_orderbook).sum())
            window_audit["usable_orderbook_rows"] = usable_rows
            window_audit["usable_orderbook_ratio"] = _float_or_none(usable_rows / max(1, int(window_mask.sum())))
            if usable_rows <= 0:
                audit["errors"].append(f"no usable orderbook rows in window {window_id}")
            elif usable_rows < 100:
                audit["warnings"].append(f"low usable orderbook rows in window {window_id}: {usable_rows}")
        audit["window_coverage"][window_id] = window_audit
        if not covers:
            audit["errors"].append(f"does not cover window {window_id} ({start.date()} to {end.date()})")
    return audit


def _audit_sqlite_context_file(path: Path, windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]]) -> dict[str, Any]:
    audit: dict[str, Any] = {"path": str(path), "kind": "sqlite_context", "errors": [], "warnings": []}
    if not path.exists():
        audit["errors"].append("missing file")
        return audit
    try:
        with sqlite3.connect(path) as conn:
            summary = pd.read_sql_query(
                "SELECT COUNT(*) AS rows, MIN(date) AS min_date, MAX(date) AS max_date FROM context_features_1h",
                conn,
            )
            columns = pd.read_sql_query("PRAGMA table_info(context_features_1h)", conn)
            has_available = {"max_source_available_at", "date"}.issubset(set(columns["name"].astype(str)))
            violation_rows = 0
            if has_available:
                violation_rows = int(
                    pd.read_sql_query(
                        "SELECT COUNT(*) AS rows FROM context_features_1h WHERE max_source_available_at > date",
                        conn,
                    ).loc[0, "rows"]
                )
    except Exception as exc:
        audit["errors"].append(f"failed to inspect sqlite: {exc}")
        return audit
    rows = int(summary.loc[0, "rows"])
    audit["rows"] = rows
    if rows <= 0:
        audit["errors"].append("context_features_1h is empty")
        return audit
    min_date = pd.to_datetime(summary.loc[0, "min_date"], utc=True, errors="coerce")
    max_date = pd.to_datetime(summary.loc[0, "max_date"], utc=True, errors="coerce")
    audit["min_date"] = min_date.isoformat() if pd.notna(min_date) else None
    audit["max_date"] = max_date.isoformat() if pd.notna(max_date) else None
    audit["max_source_available_after_date_rows"] = violation_rows
    if violation_rows:
        audit["errors"].append(f"max_source_available_at after date rows: {violation_rows}")
    for window_id, (start, end) in windows.items():
        covers = bool(pd.notna(min_date) and pd.notna(max_date) and min_date <= start and max_date >= end)
        audit.setdefault("window_coverage", {})[window_id] = {
            "covers": covers,
            "starts_before_or_at_window": bool(pd.notna(min_date) and min_date <= start),
            "ends_after_or_at_window": bool(pd.notna(max_date) and max_date >= end),
        }
        if not covers:
            audit["errors"].append(f"does not cover window {window_id} ({start.date()} to {end.date()})")
    return audit


def _target_balance_audit(experiment_specs: list[dict[str, Any]], windows: dict[str, tuple[pd.Timestamp, pd.Timestamp]]) -> dict[str, Any]:
    audit: dict[str, Any] = {"rows": [], "errors": [], "warnings": []}
    try:
        from user_data.Custom_Launcher.research.context_features.structure_orderbook_confluence_tests import (
            DEFAULT_ORDERBOOK,
            DEFAULT_STRUCTURAL,
            event_mask,
            load_frame,
        )

        frame = load_frame(DEFAULT_STRUCTURAL, DEFAULT_ORDERBOOK)
    except Exception as exc:
        audit["warnings"].append(f"target balance audit skipped: {exc}")
        return audit
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    missing_actual_columns = sorted(
        {
            str(getattr(spec["profile"], "actual_column", "") or "")
            for spec in experiment_specs
            if str(getattr(spec["profile"], "actual_column", "") or "") and str(getattr(spec["profile"], "actual_column", "") or "") not in frame
        }
    )
    if missing_actual_columns:
        frame = _append_trader_confluence_actual_columns(frame, missing_actual_columns, audit)
    for spec in experiment_specs:
        profile = spec["profile"]
        event_id = str(getattr(profile, "event_id", "") or "all_rows")
        actual_column = str(getattr(profile, "actual_column", "") or "")
        if not actual_column or actual_column not in frame:
            audit["warnings"].append(f"{profile.profile_id}/{spec['window_id']}: missing actual column {actual_column}")
            continue
        start, end = windows[str(spec["window_id"])]
        scoped = frame[frame["date"].ge(start) & frame["date"].le(end)].copy()
        if scoped.empty:
            audit["errors"].append(f"{profile.profile_id}/{spec['window_id']}: no rows in timerange")
            continue
        if event_id in {"all", "all_rows", "none"}:
            event_rows = pd.Series(True, index=scoped.index)
        else:
            event_rows = event_mask(scoped, event_id)
        actual = pd.to_numeric(scoped.loc[event_rows, actual_column], errors="coerce").dropna()
        row = {
            "profile_id": str(profile.profile_id),
            "window": str(spec["window_id"]),
            "event_id": event_id,
            "actual_column": actual_column,
            "timerange_rows": int(len(scoped)),
            "event_rows": int(event_rows.sum()),
            "label_rows": int(len(actual)),
            "event_rate": _float_or_none(actual.mean()) if len(actual) else None,
            "positive_rows": int(actual.eq(1.0).sum()) if len(actual) else 0,
            "negative_rows": int(actual.eq(0.0).sum()) if len(actual) else 0,
        }
        audit["rows"].append(row)
        if int(row["label_rows"]) < 30:
            audit["warnings"].append(f"{profile.profile_id}/{spec['window_id']}: low label rows {row['label_rows']}")
        if actual.nunique(dropna=True) < 2:
            audit["warnings"].append(f"{profile.profile_id}/{spec['window_id']}: target has fewer than two classes")
        if event_id != "all_rows" and int(row["event_rows"]) < 30:
            audit["warnings"].append(f"{profile.profile_id}/{spec['window_id']}: low event rows {row['event_rows']}")
    return audit


def _append_trader_confluence_actual_columns(frame: pd.DataFrame, columns: list[str], audit: dict[str, Any]) -> pd.DataFrame:
    if not DEFAULT_TRADER_CONFLUENCE.exists():
        audit["warnings"].append(f"trader confluence actual-column source missing: {DEFAULT_TRADER_CONFLUENCE}")
        return frame
    try:
        confluence = pd.read_parquet(DEFAULT_TRADER_CONFLUENCE, columns=["date", *columns])
    except Exception as exc:
        audit["warnings"].append(f"failed to load trader confluence actual columns {columns}: {exc}")
        return frame
    if "date" not in confluence:
        audit["warnings"].append("trader confluence actual-column source has no date column")
        return frame
    confluence["date"] = pd.to_datetime(confluence["date"], utc=True, errors="coerce")
    merge_columns = ["date", *[column for column in columns if column in confluence]]
    if len(merge_columns) <= 1:
        audit["warnings"].append(f"trader confluence actual-column source did not contain requested columns: {columns}")
        return frame
    scoped = confluence[merge_columns].dropna(subset=["date"]).drop_duplicates("date", keep="last")
    return frame.merge(scoped, on="date", how="left")


def _float_or_none(value: Any) -> float | None:
    parsed = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(parsed):
        return None
    return float(parsed)


def _sqlite_feature_validation_errors(path: Path) -> list[str]:
    errors: list[str] = []
    try:
        with sqlite3.connect(path) as conn:
            tables = pd.read_sql_query(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'context_features_1h'",
                conn,
            )
            if tables.empty:
                return ["missing context_features_1h table"]
            summary = pd.read_sql_query(
                "SELECT COUNT(*) AS rows, MIN(date) AS min_date, MAX(date) AS max_date FROM context_features_1h",
                conn,
            )
            if int(summary.loc[0, "rows"]) <= 0:
                errors.append("context_features_1h is empty")
            if pd.isna(summary.loc[0, "min_date"]) or pd.isna(summary.loc[0, "max_date"]):
                errors.append("context_features_1h has missing min/max date")
    except Exception as exc:
        errors.append(f"failed to validate sqlite: {exc}")
    return errors


if __name__ == "__main__":
    raise SystemExit(main())
