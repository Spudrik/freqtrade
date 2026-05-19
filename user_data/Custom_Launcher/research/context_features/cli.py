from __future__ import annotations

from pathlib import Path
import argparse
import json

from .builder import (
    build_context_features,
    create_btc_return_report,
    default_paths,
    export_features,
    read_feature_status,
    validate_no_lookahead,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build research-only hourly context features.")
    parser.add_argument("--app-dir", type=Path, default=None)
    parser.add_argument("--news-db", type=Path, default=None)
    parser.add_argument("--web-db", type=Path, default=None)
    parser.add_argument("--global-db", type=Path, default=None)
    parser.add_argument("--feature-db", type=Path, default=None)
    parser.add_argument("--export-dir", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--btc-ohlcv-path", type=Path, default=None)
    parser.add_argument("--mode", choices=["dry-run", "update", "full-rebuild"], default="dry-run")
    parser.add_argument("--overlap-days", type=int, default=7)
    parser.add_argument("--export-parquet", action="store_true")
    parser.add_argument("--export-csv", action="store_true")
    parser.add_argument("--make-report", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--export-only", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args()

    paths = default_paths(args.app_dir)
    paths = type(paths)(
        app_dir=paths.app_dir,
        news_db=args.news_db or paths.news_db,
        web_db=args.web_db or paths.web_db,
        global_db=args.global_db or paths.global_db,
        feature_db=args.feature_db or paths.feature_db,
        export_dir=args.export_dir or paths.export_dir,
        report_dir=args.report_dir or paths.report_dir,
        btc_ohlcv_path=args.btc_ohlcv_path or paths.btc_ohlcv_path,
    )
    if args.status:
        print(json.dumps(read_feature_status(paths.feature_db), indent=2, sort_keys=True))
        return 0
    if args.validate:
        print(json.dumps(validate_no_lookahead(paths.feature_db), indent=2, sort_keys=True))
        return 0
    if args.export_only:
        path, count = export_features(paths.feature_db, paths.export_dir, parquet=args.export_parquet or not args.export_csv, csv=args.export_csv)
        print(json.dumps({"export_path": str(path), "rows": count}, indent=2, sort_keys=True))
        return 0
    if args.report_only:
        path = create_btc_return_report(paths.feature_db, paths.btc_ohlcv_path, paths.report_dir)
        print(json.dumps({"report_path": str(path)}, indent=2, sort_keys=True))
        return 0
    summary = build_context_features(
        paths,
        mode=args.mode,
        overlap_days=args.overlap_days,
        export_parquet=args.export_parquet,
        export_csv=args.export_csv,
        make_report=args.make_report,
    )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
