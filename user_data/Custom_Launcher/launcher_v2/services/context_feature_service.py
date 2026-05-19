from __future__ import annotations

from pathlib import Path
from typing import Any

from .collector_service import open_path
from research.context_features.builder import (
    ContextFeaturePaths,
    build_context_features,
    create_btc_return_report,
    default_paths,
    export_features,
    read_feature_status,
    validate_no_lookahead,
)


class ContextFeatureService:
    def __init__(self, app_dir: Path) -> None:
        self.app_dir = Path(app_dir)

    def paths(self, state: dict[str, Any]) -> ContextFeaturePaths:
        defaults = default_paths(self.app_dir)
        return ContextFeaturePaths(
            app_dir=defaults.app_dir,
            news_db=_path_or_default(state.get("news_db"), defaults.news_db),
            web_db=_path_or_default(state.get("web_db"), defaults.web_db),
            global_db=_path_or_default(state.get("global_db"), defaults.global_db),
            feature_db=_path_or_default(state.get("feature_db"), defaults.feature_db),
            export_dir=_path_or_default(state.get("export_dir"), defaults.export_dir),
            report_dir=_path_or_default(state.get("report_dir"), defaults.report_dir),
            btc_ohlcv_path=_path_or_default(state.get("btc_ohlcv_path"), defaults.btc_ohlcv_path),
        )

    def status(self, state: dict[str, Any]) -> dict[str, Any]:
        return read_feature_status(self.paths(state).feature_db)

    def dry_run(self, state: dict[str, Any]) -> dict[str, Any]:
        summary = build_context_features(self.paths(state), mode="dry-run", overlap_days=_overlap_days(state))
        return summary.to_dict()

    def update(self, state: dict[str, Any]) -> dict[str, Any]:
        summary = build_context_features(self.paths(state), mode="update", overlap_days=_overlap_days(state))
        return summary.to_dict()

    def full_rebuild(self, state: dict[str, Any]) -> dict[str, Any]:
        summary = build_context_features(self.paths(state), mode="full-rebuild", overlap_days=_overlap_days(state))
        return summary.to_dict()

    def export(self, state: dict[str, Any]) -> tuple[Path, int]:
        paths = self.paths(state)
        return export_features(paths.feature_db, paths.export_dir, parquet=True, csv=False)

    def report(self, state: dict[str, Any]) -> Path:
        paths = self.paths(state)
        return create_btc_return_report(paths.feature_db, paths.btc_ohlcv_path, paths.report_dir)

    def validate(self, state: dict[str, Any]) -> dict[str, Any]:
        return validate_no_lookahead(self.paths(state).feature_db)

    def open_feature_folder(self, state: dict[str, Any]) -> None:
        path = self.paths(state).feature_db.parent
        path.mkdir(parents=True, exist_ok=True)
        open_path(path)

    def open_export_folder(self, state: dict[str, Any]) -> None:
        path = self.paths(state).export_dir
        path.mkdir(parents=True, exist_ok=True)
        open_path(path)

    def open_report_folder(self, state: dict[str, Any]) -> None:
        path = self.paths(state).report_dir
        path.mkdir(parents=True, exist_ok=True)
        open_path(path)


def _path_or_default(value: Any, default: Path) -> Path:
    text = str(value or "").strip()
    return Path(text) if text else default


def _overlap_days(state: dict[str, Any]) -> int:
    try:
        return max(1, int(float(str(state.get("overlap_days") or "7"))))
    except Exception:
        return 7
