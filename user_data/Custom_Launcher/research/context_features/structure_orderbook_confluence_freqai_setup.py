from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


APP_DIR = Path(__file__).resolve().parents[2]
USER_DATA_DIR = APP_DIR.parent
REPO_ROOT = USER_DATA_DIR.parent
DEFAULT_CONFIG = USER_DATA_DIR / "configs" / "config_context_freqai_research.example.json"
DEFAULT_RUN_ROOT = USER_DATA_DIR / "research_news_data" / "context_features" / "freqai_runs"
DEFAULT_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
DEFAULT_STRUCTURAL = USER_DATA_DIR / "research_news_data" / "context_features" / "structural_cache" / "btc_structural_features_1h_latest.parquet"
DEFAULT_ORDERBOOK = USER_DATA_DIR / "orderbook_data" / "historical_bybit" / "features" / "orderbook_trader_state_1h_latest.parquet"
DIRECT_TEST_REPORT = USER_DATA_DIR / "research_news_data" / "context_features" / "reports" / "structure_orderbook_confluence_structure_orderbook_v2_summary.csv"


WINDOWS = {
    "broad": "20250701-20260218",
}

MODELS = {
    "structure_vp_tree": {
        "strategy": "ContextStructureVahRejectionFreqAIResearchStrategy",
        "freqaimodel": "LightGBMRegressorMultiTarget",
        "training_parameters": {
            "n_estimators": 90,
            "learning_rate": 0.04,
            "num_leaves": 7,
            "max_depth": 3,
            "min_child_samples": 18,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": 1,
            "verbosity": -1,
        },
    },
    "structure_vp_orderbook_tree": {
        "strategy": "ContextStructureOrderbookVahRejectionFreqAIResearchStrategy",
        "freqaimodel": "LightGBMRegressorMultiTarget",
        "training_parameters": {
            "n_estimators": 90,
            "learning_rate": 0.04,
            "num_leaves": 7,
            "max_depth": 3,
            "min_child_samples": 18,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": 1,
            "verbosity": -1,
        },
    },
    "structure_vp_ridge": {
        "strategy": "ContextStructureVahRejectionFreqAIResearchStrategy",
        "freqaimodel": "SKLearnRidgeRegressorMultiTarget",
        "training_parameters": {},
    },
    "structure_vp_orderbook_ridge": {
        "strategy": "ContextStructureOrderbookVahRejectionFreqAIResearchStrategy",
        "freqaimodel": "SKLearnRidgeRegressorMultiTarget",
        "training_parameters": {},
    },
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare structure/orderbook confluence FreqAI runs.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--python-exe", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--windows", default="broad")
    parser.add_argument("--models", default="structure_vp_tree,structure_vp_orderbook_tree")
    parser.add_argument("--train-days", type=int, default=120)
    parser.add_argument("--backtest-days", type=int, default=14)
    parser.add_argument("--plot-feature-importances", type=int, default=0)
    parser.add_argument("--candidate-id", default="vah_rejection")
    parser.add_argument("--event-id", default="vah_rejection")
    parser.add_argument("--actual-column", default="breakout_failure_next_6h")
    parser.add_argument("--prediction-column", default="&-so_vah_rejection_breakout_failure_next_6h")
    parser.add_argument(
        "--objective",
        default=(
            "Validate a direct-test candidate by comparing a structure/VP control against "
            "a structure/VP/orderbook candidate."
        ),
    )
    args = parser.parse_args()

    _require_file(args.config, "base FreqAI config")
    _require_file(DEFAULT_STRUCTURAL, "structural feature parquet")
    _require_file(DEFAULT_ORDERBOOK, "orderbook trader-state parquet")

    windows = _selected(args.windows, WINDOWS)
    models = _selected(args.models, MODELS)
    args.run_root.mkdir(parents=True, exist_ok=True)

    base_config = json.loads(args.config.read_text(encoding="utf-8"))
    commands: list[dict[str, Any]] = []
    created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_dir = args.run_root / f"structure_orderbook_confluence_setup_{created_at}"
    setup_dir.mkdir(parents=True, exist_ok=True)

    for window_name in windows:
        timerange = WINDOWS[window_name]
        for model_name in models:
            model = MODELS[model_name]
            safe_candidate = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "-" for ch in args.candidate_id)
            identifier = f"structure-ob-{safe_candidate}-{model_name}-{window_name}-{timerange}-v1"
            run_dir = args.run_root / f"freqai_structure_orderbook_{model_name}_{window_name}_{created_at}"
            run_dir.mkdir(parents=True, exist_ok=True)
            config = _run_config(base_config, identifier, args.train_days, args.backtest_days, args.plot_feature_importances, model)
            config_path = run_dir / f"{identifier}.json"
            config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
            command = [
                str(args.python_exe),
                "-m",
                "freqtrade",
                "backtesting",
                "--userdir",
                str(USER_DATA_DIR),
                "--config",
                str(config_path),
                "--strategy",
                str(model["strategy"]),
                "--freqaimodel",
                str(model["freqaimodel"]),
                "--timerange",
                timerange,
                "--export",
                "signals",
                "--export-directory",
                str(run_dir),
            ]
            commands.append(
                {
                    "window": window_name,
                    "timerange": timerange,
                    "model": model_name,
                    "identifier": identifier,
                    "run_dir": str(run_dir),
                    "config_path": str(config_path),
                    "command": command,
                }
            )

    manifest_path = setup_dir / "structure_orderbook_confluence_freqai_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "status": "setup-only",
                "created_at": datetime.now().astimezone().isoformat(),
                "objective": args.objective,
                "candidate_id": args.candidate_id,
                "event_id": args.event_id,
                "actual_column": args.actual_column,
                "prediction_column": args.prediction_column,
                "direct_test_gate": {
                    "source_report": str(DIRECT_TEST_REPORT),
                    "required_candidate": f"{args.event_id} -> {args.actual_column}",
                    "direct_test_result": "candidate should pass direct-test gate before broad FreqAI validation",
                },
                "data_inputs": {
                    "structural_features": str(DEFAULT_STRUCTURAL),
                    "orderbook_features": str(DEFAULT_ORDERBOOK),
                },
                "commands": commands,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    ps1_path = setup_dir / "run_structure_orderbook_confluence_freqai.ps1"
    ps1_path.write_text(_powershell_script(commands), encoding="utf-8")
    print(json.dumps({"manifest_path": str(manifest_path), "run_script": str(ps1_path), "commands": len(commands)}, indent=2))
    return 0


def _selected(raw: str, allowed: dict[str, Any]) -> list[str]:
    selected = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
    invalid = [part for part in selected if part not in allowed]
    if invalid:
        raise ValueError(f"Unknown option(s): {', '.join(invalid)}")
    return selected


def _require_file(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def _run_config(
    base: dict[str, Any],
    identifier: str,
    train_days: int,
    backtest_days: int,
    plot_importances: int,
    model: dict[str, Any],
) -> dict[str, Any]:
    config = json.loads(json.dumps(base))
    config["bot_name"] = "structure_orderbook_confluence_freqai_research"
    freqai = config.setdefault("freqai", {})
    freqai["identifier"] = identifier
    freqai["train_period_days"] = train_days
    freqai["backtest_period_days"] = backtest_days
    feature_parameters = freqai.setdefault("feature_parameters", {})
    feature_parameters["label_period_candles"] = 6
    feature_parameters["indicator_periods_candles"] = [3, 6, 12, 24]
    feature_parameters["plot_feature_importances"] = plot_importances
    freqai["model_training_parameters"] = dict(model.get("training_parameters") or {})
    return config


def _powershell_script(commands: list[dict[str, Any]]) -> str:
    lines = [
        "$ErrorActionPreference = 'Stop'",
        "Set-Location 'C:\\FreqTradeStuff'",
    ]
    for item in commands:
        command = " ".join(_ps_quote(part) for part in item["command"])
        lines.append(f"# {item['model']} {item['window']} {item['timerange']}")
        lines.append(f"& {command}")
    return "\n".join(lines) + "\n"


def _ps_quote(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


if __name__ == "__main__":
    raise SystemExit(main())
