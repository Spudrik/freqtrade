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
DEFAULT_BEHAVIOUR_FEATURE_CACHE = USER_DATA_DIR / "orderbook_data" / "live" / "exports" / "orderbook_features_1h_behaviour_latest.parquet"
DEFAULT_TRADER_STATE_FEATURE_CACHE = USER_DATA_DIR / "orderbook_data" / "historical_bybit" / "features" / "orderbook_trader_state_1h_latest.parquet"
DEFAULT_OHLCV = USER_DATA_DIR / "data" / "binance" / "BTC_USDT-1h.feather"
DIRECT_TEST_SCRIPT = Path(__file__).with_name("orderbook_behaviour_direct_tests.py")

WINDOWS = {
    "quiet": "20250725-20250811",
    "volatile": "20251015-20251031",
    "event": "20260129-20260216",
}

MODELS = {
    "price_tree": {
        "strategy": "ContextBybitOrderbookBehaviourPriceOnlyFreqAIResearchStrategy",
        "freqaimodel": "LightGBMRegressorMultiTarget",
        "training_parameters": {
            "n_estimators": 150,
            "learning_rate": 0.04,
            "num_leaves": 15,
            "max_depth": 5,
            "min_child_samples": 20,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": 1,
            "verbosity": -1,
        },
    },
    "price_event_tree": {
        "strategy": "ContextBybitOrderbookTraderEventPriceOnlyFreqAIResearchStrategy",
        "freqaimodel": "LightGBMRegressorMultiTarget",
        "training_parameters": {
            "n_estimators": 150,
            "learning_rate": 0.04,
            "num_leaves": 15,
            "max_depth": 5,
            "min_child_samples": 20,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": 1,
            "verbosity": -1,
        },
    },
    "behaviour_tree": {
        "strategy": "ContextBybitOrderbookBehaviourParquetFreqAIResearchStrategy",
        "freqaimodel": "LightGBMRegressorMultiTarget",
        "training_parameters": {
            "n_estimators": 150,
            "learning_rate": 0.04,
            "num_leaves": 15,
            "max_depth": 5,
            "min_child_samples": 20,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": 1,
            "verbosity": -1,
        },
    },
    "price_ridge": {
        "strategy": "ContextBybitOrderbookBehaviourPriceOnlyFreqAIResearchStrategy",
        "freqaimodel": "SKLearnRidgeRegressorMultiTarget",
        "training_parameters": {},
    },
    "price_event_ridge": {
        "strategy": "ContextBybitOrderbookTraderEventPriceOnlyFreqAIResearchStrategy",
        "freqaimodel": "SKLearnRidgeRegressorMultiTarget",
        "training_parameters": {},
    },
    "behaviour_ridge": {
        "strategy": "ContextBybitOrderbookBehaviourParquetFreqAIResearchStrategy",
        "freqaimodel": "SKLearnRidgeRegressorMultiTarget",
        "training_parameters": {},
    },
    "zone_ridge": {
        "strategy": "ContextBybitOrderbookZoneCompressionParquetFreqAIResearchStrategy",
        "freqaimodel": "SKLearnRidgeRegressorMultiTarget",
        "training_parameters": {},
    },
    "rejection_ridge": {
        "strategy": "ContextBybitOrderbookRejectionStateParquetFreqAIResearchStrategy",
        "freqaimodel": "SKLearnRidgeRegressorMultiTarget",
        "training_parameters": {},
    },
    "alpha_ridge": {
        "strategy": "ContextBybitOrderbookObjectiveAlphaParquetFreqAIResearchStrategy",
        "freqaimodel": "SKLearnRidgeRegressorMultiTarget",
        "training_parameters": {},
    },
    "price_xgboost": {
        "strategy": "ContextBybitOrderbookBehaviourPriceOnlyFreqAIResearchStrategy",
        "freqaimodel": "XGBoostRegressorMultiTarget",
        "training_parameters": {
            "n_estimators": 120,
            "learning_rate": 0.04,
            "max_depth": 4,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": 1,
            "verbosity": 0,
            "tree_method": "hist",
        },
    },
    "behaviour_xgboost": {
        "strategy": "ContextBybitOrderbookBehaviourParquetFreqAIResearchStrategy",
        "freqaimodel": "XGBoostRegressorMultiTarget",
        "training_parameters": {
            "n_estimators": 120,
            "learning_rate": 0.04,
            "max_depth": 4,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": 1,
            "verbosity": 0,
            "tree_method": "hist",
        },
    },
    "zone_tree": {
        "strategy": "ContextBybitOrderbookZoneCompressionParquetFreqAIResearchStrategy",
        "freqaimodel": "LightGBMRegressorMultiTarget",
        "training_parameters": {
            "n_estimators": 90,
            "learning_rate": 0.04,
            "num_leaves": 7,
            "max_depth": 3,
            "min_child_samples": 20,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": 1,
            "verbosity": -1,
        },
    },
    "rejection_tree": {
        "strategy": "ContextBybitOrderbookRejectionStateParquetFreqAIResearchStrategy",
        "freqaimodel": "LightGBMRegressorMultiTarget",
        "training_parameters": {
            "n_estimators": 90,
            "learning_rate": 0.04,
            "num_leaves": 7,
            "max_depth": 3,
            "min_child_samples": 20,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": 1,
            "verbosity": -1,
        },
    },
    "alpha_tree": {
        "strategy": "ContextBybitOrderbookObjectiveAlphaParquetFreqAIResearchStrategy",
        "freqaimodel": "LightGBMRegressorMultiTarget",
        "training_parameters": {
            "n_estimators": 90,
            "learning_rate": 0.04,
            "num_leaves": 7,
            "max_depth": 3,
            "min_child_samples": 20,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": 1,
            "verbosity": -1,
        },
    },
    "trader_ridge": {
        "strategy": "ContextBybitOrderbookTraderStateParquetFreqAIResearchStrategy",
        "freqaimodel": "SKLearnRidgeRegressorMultiTarget",
        "training_parameters": {},
    },
    "trader_tree": {
        "strategy": "ContextBybitOrderbookTraderStateParquetFreqAIResearchStrategy",
        "freqaimodel": "LightGBMRegressorMultiTarget",
        "training_parameters": {
            "n_estimators": 120,
            "learning_rate": 0.04,
            "num_leaves": 11,
            "max_depth": 4,
            "min_child_samples": 16,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": 1,
            "verbosity": -1,
        },
    },
    "trader_refined_ridge": {
        "strategy": "ContextBybitOrderbookTraderRefinedParquetFreqAIResearchStrategy",
        "freqaimodel": "SKLearnRidgeRegressorMultiTarget",
        "training_parameters": {},
    },
    "trader_refined_tree": {
        "strategy": "ContextBybitOrderbookTraderRefinedParquetFreqAIResearchStrategy",
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
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare orderbook behaviour FreqAI runs without executing them.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--python-exe", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--windows", default="quiet,volatile,event")
    parser.add_argument("--models", default="price_ridge,behaviour_ridge,price_tree,behaviour_tree,price_xgboost,behaviour_xgboost")
    parser.add_argument("--train-days", type=int, default=14)
    parser.add_argument("--backtest-days", type=int, default=1)
    parser.add_argument("--plot-feature-importances", type=int, default=0)
    args = parser.parse_args()

    windows = _selected(args.windows, WINDOWS)
    models = _selected(args.models, MODELS)
    args.run_root.mkdir(parents=True, exist_ok=True)

    base_config = json.loads(args.config.read_text(encoding="utf-8"))
    commands: list[dict[str, Any]] = []
    created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
    setup_dir = args.run_root / f"orderbook_behaviour_setup_{created_at}"
    setup_dir.mkdir(parents=True, exist_ok=True)

    for window_name in windows:
        timerange = WINDOWS[window_name]
        for model_name in models:
            model = MODELS[model_name]
            identifier = f"bybit-behaviour-{model_name}-{window_name}-{timerange}-v1"
            run_dir = args.run_root / f"freqai_context_behaviour_{model_name}_{window_name}_{created_at}"
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

    manifest_path = setup_dir / "orderbook_behaviour_freqai_manifest.json"
    direct_test_setup_command = [
        str(args.python_exe),
        str(DIRECT_TEST_SCRIPT),
        "--features",
        str(DEFAULT_BEHAVIOUR_FEATURE_CACHE),
        "--ohlcv",
        str(DEFAULT_OHLCV),
    ]
    direct_test_execute_command = [
        *direct_test_setup_command,
        "--execute",
        "--snapshot-confirmed",
    ]
    manifest_path.write_text(
        json.dumps(
            {
                "status": "setup-only",
                "created_at": datetime.now().astimezone().isoformat(),
                "objective": "Compare price-only controls against behaviour models for Ridge, LightGBM, and XGBoost on short-horizon orderbook behaviour features.",
                "behaviour_feature_cache": str(DEFAULT_BEHAVIOUR_FEATURE_CACHE),
                "trader_state_feature_cache": str(DEFAULT_TRADER_STATE_FEATURE_CACHE),
                "direct_tests": {
                    "setup_command": direct_test_setup_command,
                    "execute_command_after_snapshot_confirmation": direct_test_execute_command,
                },
                "commands": commands,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    ps1_path = setup_dir / "run_orderbook_behaviour_freqai.ps1"
    ps1_path.write_text(_powershell_script(commands), encoding="utf-8")
    print(json.dumps({"manifest_path": str(manifest_path), "run_script": str(ps1_path), "commands": len(commands)}, indent=2))
    return 0


def _selected(raw: str, allowed: dict[str, Any]) -> list[str]:
    selected = [part.strip() for part in raw.replace(";", ",").split(",") if part.strip()]
    invalid = [part for part in selected if part not in allowed]
    if invalid:
        raise ValueError(f"Unknown option(s): {', '.join(invalid)}")
    return selected


def _run_config(
    base: dict[str, Any],
    identifier: str,
    train_days: int,
    backtest_days: int,
    plot_importances: int,
    model: dict[str, Any],
) -> dict[str, Any]:
    config = json.loads(json.dumps(base))
    config["bot_name"] = "context_orderbook_behaviour_freqai_research"
    freqai = config.setdefault("freqai", {})
    freqai["identifier"] = identifier
    freqai["train_period_days"] = train_days
    freqai["backtest_period_days"] = backtest_days
    feature_parameters = freqai.setdefault("feature_parameters", {})
    feature_parameters["label_period_candles"] = 24
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
