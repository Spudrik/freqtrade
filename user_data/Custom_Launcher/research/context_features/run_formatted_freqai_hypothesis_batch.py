from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[4]
USER_DATA_DIR = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from user_data.strategies.FormattedContextHypothesisFreqAIStrategy import (  # noqa: E402
    FormattedContextHypothesisFreqAIBaseStrategy,
    SNAPSHOT_PATH,
)


DEFAULT_PYTHON = REPO_ROOT / ".venv" / "Scripts" / "python.exe"
DEFAULT_REPORT_ROOT = (
    USER_DATA_DIR
    / "research_news_data"
    / "context_features"
    / "reports"
    / "formatted_freqai_hypothesis"
)
DEFAULT_MODELS_DIR = USER_DATA_DIR / "models"
DEFAULT_TIMERANGE = "20260508-20260605"
PAIR = "BTC/USDT:USDT"
PAIR_CANONICAL = "BTC/USDT"


def _canonical_pair(pair: str | None) -> str:
    return str(pair or PAIR).upper().split(":", 1)[0]


PROFILES = {
    "price_control": {
        "strategy": "FormattedFreqAIPriceOnlyHypothesisStrategy",
        "hypothesis_group": "H046-H060 price/volume control",
        "theory": "Price, volume, compression, and breakout pressure alone provide the baseline ranking power.",
    },
    "context_news": {
        "strategy": "FormattedFreqAIContextHypothesisStrategy",
        "hypothesis_group": "H001-H015 context escalation/relief",
        "theory": "Safe source news/context escalation or relief changes the odds of reward-before-danger paths.",
    },
    "context_compact_risk": {
        "strategy": "FormattedFreqAIContextCompactRiskHypothesisStrategy",
        "hypothesis_group": "H001-H015 context compact risk/state refinement",
        "theory": "The small trader-state context features may carry the drawdown-risk signal without the full transform set.",
    },
    "context_topic_risk": {
        "strategy": "FormattedFreqAIContextTopicRiskHypothesisStrategy",
        "hypothesis_group": "H001-H015 context topic-risk refinement",
        "theory": "Risk, conflict, macro, regulation, liquidity, and security context should identify hours where downside danger rises.",
    },
    "context_crypto_native": {
        "strategy": "FormattedFreqAIContextCryptoNativeHypothesisStrategy",
        "hypothesis_group": "H001-H015 crypto-native context refinement",
        "theory": "Crypto-specific context such as BTC/ETH, ETF, stablecoin, liquidation, exchange, security, and regulation terms should refine BTC path risk.",
    },
    "context_activity": {
        "strategy": "FormattedFreqAIContextActivityHypothesisStrategy",
        "hypothesis_group": "H001-H015 context activity/coverage refinement",
        "theory": "News activity, source volume, and coverage intensity may be acting as a risk-regime proxy rather than topic content.",
    },
    "orderbook_pressure": {
        "strategy": "FormattedFreqAIOrderbookHypothesisStrategy",
        "hypothesis_group": "H016-H045 orderbook pressure/walls/liquidity",
        "theory": "Orderbook pressure, wall persistence/removal, and liquidity vacuum features improve path ranking.",
    },
    "cross_confluence": {
        "strategy": "FormattedFreqAICrossConfluenceHypothesisStrategy",
        "hypothesis_group": "H061-H080 multi-source confluence",
        "theory": "News/context, orderbook, market breadth, and price agreeing together should improve ranking versus each source alone.",
    },
    "market_breadth": {
        "strategy": "FormattedFreqAIMarketBreadthHypothesisStrategy",
        "hypothesis_group": "H081-H092 market breadth and BTC-alt regime",
        "theory": "BTC path odds change when broader crypto breadth confirms or contradicts local price action.",
    },
    "market_compact": {
        "strategy": "FormattedFreqAIMarketCompactHypothesisStrategy",
        "hypothesis_group": "H081-H092 compact market breadth refinement",
        "theory": "The small market-breadth confluence layer may explain the downside/breakdown signal without all raw breadth columns.",
    },
    "market_raw_breadth": {
        "strategy": "FormattedFreqAIMarketRawBreadthHypothesisStrategy",
        "hypothesis_group": "H081-H092 raw market breadth refinement",
        "theory": "Raw alt-market returns, breadth, and pressure features may identify broad risk-on/off moves before BTC path changes.",
    },
    "market_volatility": {
        "strategy": "FormattedFreqAIMarketVolatilityHypothesisStrategy",
        "hypothesis_group": "H081-H092 market volatility/compression refinement",
        "theory": "Broad market volatility, range, and compression features may identify expansion or breakdown risk.",
    },
    "live_media": {
        "strategy": "FormattedFreqAILiveMediaHypothesisStrategy",
        "hypothesis_group": "Live media news/web/global core",
        "theory": "Live news, web, and global-source features may improve short-horizon BTC path ranking versus price-only.",
    },
    "live_media_topic_risk": {
        "strategy": "FormattedFreqAILiveMediaTopicRiskHypothesisStrategy",
        "hypothesis_group": "Live media topic-risk",
        "theory": "Live risk, macro, crypto-native, security, regulation, and source-confluence signals may identify fast downside or upside follow-through.",
    },
    "live_media_activity": {
        "strategy": "FormattedFreqAILiveMediaActivityHypothesisStrategy",
        "hypothesis_group": "Live media activity/attention",
        "theory": "Fast changes in article/source activity and attention may explain 1h, 2h, and 4h BTC movement better than price-only.",
    },
    "live_media_news_activity": {
        "strategy": "FormattedFreqAILiveNewsActivityHypothesisStrategy",
        "hypothesis_group": "Live media source-family ablation: news",
        "theory": "News-only activity/attention features should be tested separately so we know whether the signal comes from news rather than web/global coverage.",
    },
    "live_media_web_activity": {
        "strategy": "FormattedFreqAILiveWebActivityHypothesisStrategy",
        "hypothesis_group": "Live media source-family ablation: web",
        "theory": "Web-only activity/attention features should be tested separately so we know whether the signal comes from broad web coverage rather than news/global metrics.",
    },
    "live_media_global_macro": {
        "strategy": "FormattedFreqAILiveGlobalMacroHypothesisStrategy",
        "hypothesis_group": "Live media source-family ablation: global",
        "theory": "Global market/macro context features should be tested separately so we know whether the signal is just broad market regime information.",
    },
    "live_media_trader_shapes": {
        "strategy": "FormattedFreqAILiveMediaTraderShapesHypothesisStrategy",
        "hypothesis_group": "Live media tactical trader-shape layer",
        "theory": "Fast source agreement, topic acceleration, fresh-topic crosses, persistence, and price-confirmed media states improve 1h, 2h, and 4h path ranking.",
    },
    "live_media_topic_risk_trader_shapes": {
        "strategy": "FormattedFreqAILiveMediaTopicRiskTraderShapesHypothesisStrategy",
        "hypothesis_group": "Live media topic-risk tactical trader-shape layer",
        "theory": "Risk, macro, crypto, and relief topics become more useful when encoded as fast trader-readable shapes rather than raw counts.",
    },
    "live_media_trader_shapes_vp": {
        "strategy": "FormattedFreqAILiveMediaTraderShapesVpHypothesisStrategy",
        "hypothesis_group": "Live media tactical shapes plus volume profile",
        "theory": "News/source acceleration should become more useful when FreqAI also sees VP value-area state, POC/VAH/VAL distance, node evidence, and news-plus-VP confluence.",
    },
    "live_media_plus_orderbook": {
        "strategy": "FormattedFreqAILiveMediaPlusOrderbookHypothesisStrategy",
        "hypothesis_group": "Live media plus Binance orderbook",
        "theory": "Live media becomes more useful when Binance orderbook pressure, walls, spread, and liquidity fragility agree with the news state.",
    },
    "live_media_plus_loose_orderbook": {
        "strategy": "FormattedFreqAILiveMediaPlusLooseOrderbookHypothesisStrategy",
        "hypothesis_group": "Live media plus loose/tick-present Binance orderbook",
        "theory": "Loose orderbook availability may still carry useful wall, spread, pressure, and liquidity state when strict-ready orderbook windows are too sparse.",
    },
}

TARGET_HORIZONS = (1, 2, 3, 4, 6, 24)
STRUCTURE_HORIZONS = (1, 2, 4, 6)
TARGETS = [
    *[f"&-future_return_{horizon}h" for horizon in TARGET_HORIZONS],
    *[f"&-future_max_upside_{horizon}h" for horizon in TARGET_HORIZONS],
    *[f"&-future_max_drawdown_{horizon}h" for horizon in TARGET_HORIZONS],
    *[f"&-long_reward_before_danger_{horizon}h" for horizon in TARGET_HORIZONS],
    *[f"&-short_reward_before_danger_{horizon}h" for horizon in TARGET_HORIZONS],
    *[f"&-large_upside_next_{horizon}h" for horizon in TARGET_HORIZONS],
    *[f"&-large_drawdown_next_{horizon}h" for horizon in TARGET_HORIZONS],
    *[f"&-breakout_success_from_current_setup_{horizon}h" for horizon in STRUCTURE_HORIZONS],
    *[f"&-breakdown_success_from_current_setup_{horizon}h" for horizon in STRUCTURE_HORIZONS],
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a bounded FreqAI batch against the formatted aligned snapshot.")
    parser.add_argument("--python-exe", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    parser.add_argument("--timerange", default=DEFAULT_TIMERANGE)
    parser.add_argument("--pair", default=PAIR)
    parser.add_argument("--profiles", default="price_control,context_news,orderbook_pressure,cross_confluence,market_breadth")
    parser.add_argument("--train-days", type=int, default=14)
    parser.add_argument("--backtest-days", type=int, default=7)
    parser.add_argument("--plot-feature-importances", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.python_exe.exists():
        raise FileNotFoundError(args.python_exe)
    if not SNAPSHOT_PATH.exists():
        raise FileNotFoundError(SNAPSHOT_PATH)
    pair = str(args.pair)

    selected_profiles = [item.strip() for item in args.profiles.split(",") if item.strip()]
    unknown = [item for item in selected_profiles if item not in PROFILES]
    if unknown:
        raise ValueError(f"Unknown profile(s): {unknown}")

    run_id = f"formatted_freqai_hypothesis_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    run_dir = args.report_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    commands = []
    for profile_id in selected_profiles:
        profile = PROFILES[profile_id]
        identifier = f"{run_id}_{profile_id}"
        config_path = run_dir / f"{profile_id}_config.json"
        config = _config(
            identifier=identifier,
            train_days=args.train_days,
            backtest_days=args.backtest_days,
            plot_feature_importances=args.plot_feature_importances,
            pair=pair,
        )
        config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        export_dir = run_dir / profile_id
        export_dir.mkdir(parents=True, exist_ok=True)
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
            str(profile["strategy"]),
            "--freqaimodel",
            "LightGBMRegressorMultiTarget",
            "--timerange",
            args.timerange,
            "--export",
            "signals",
            "--export-directory",
            str(export_dir),
        ]
        commands.append(
            {
                "profile_id": profile_id,
                "identifier": identifier,
                "strategy": profile["strategy"],
                "hypothesis_group": profile["hypothesis_group"],
                "theory": profile["theory"],
                "config_path": str(config_path),
                "export_dir": str(export_dir),
                "command": command,
                "status": "pending",
            }
        )

    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "created_at": datetime.now().astimezone().isoformat(),
        "objective": "Use FreqAI to test whether formatted live context/orderbook/market features rank trader-readable BTC path labels.",
        "snapshot": str(SNAPSHOT_PATH),
        "timerange": args.timerange,
        "pair": pair,
        "targets": TARGETS,
        "commands": commands,
    }
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.dry_run:
        print(json.dumps({"run_dir": str(run_dir), "manifest": str(manifest_path), "commands": len(commands)}, indent=2))
        return 0

    for item in commands:
        stdout_path = Path(str(item["export_dir"])) / "freqai_stdout.log"
        stderr_path = Path(str(item["export_dir"])) / "freqai_stderr.log"
        item["started_at"] = datetime.now().astimezone().isoformat()
        item["status"] = "running"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
            result = subprocess.run(
                [str(part) for part in item["command"]],
                cwd=str(REPO_ROOT),
                stdout=stdout,
                stderr=stderr,
                text=True,
                check=False,
            )
        item["finished_at"] = datetime.now().astimezone().isoformat()
        item["returncode"] = int(result.returncode)
        item["stdout_log"] = str(stdout_path)
        item["stderr_log"] = str(stderr_path)
        item["status"] = "completed" if result.returncode == 0 else "failed"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if result.returncode != 0:
            print(json.dumps({"status": "failed", "profile_id": item["profile_id"], "stderr": str(stderr_path)}, indent=2))
            return int(result.returncode)

    summary_path = run_dir / "formatted_freqai_hypothesis_metrics.csv"
    summary = score_manifest(manifest, args.models_dir)
    summary.to_csv(summary_path, index=False)
    print(json.dumps({"status": "completed", "run_dir": str(run_dir), "manifest": str(manifest_path), "metrics": str(summary_path)}, indent=2))
    return 0


def _config(*, identifier: str, train_days: int, backtest_days: int, plot_feature_importances: int, pair: str) -> dict[str, Any]:
    return {
        "$schema": "https://schema.freqtrade.io/schema.json",
        "dry_run": True,
        "trading_mode": "futures",
        "margin_mode": "isolated",
        "max_open_trades": 1,
        "stake_currency": "USDT",
        "stake_amount": 100,
        "tradable_balance_ratio": 1,
        "fiat_display_currency": "USD",
        "timeframe": "1h",
        "dry_run_wallet": 1000,
        "cancel_open_orders_on_exit": True,
        "exchange": {
            "name": "binance",
            "key": "",
            "secret": "",
            "sandbox": False,
            "skip_pair_validation": True,
            "ccxt_config": {},
            "ccxt_async_config": {},
            "pair_whitelist": [pair],
            "pair_blacklist": [],
        },
        "entry_pricing": {
            "price_side": "other",
            "use_order_book": True,
            "order_book_top": 1,
            "price_last_balance": 1.0,
            "check_depth_of_market": {"enabled": True, "bids_to_ask_delta": 1.0},
        },
        "exit_pricing": {
            "price_side": "other",
            "use_order_book": True,
            "order_book_top": 1,
            "price_last_balance": 1.0,
        },
        "pairlists": [{"method": "StaticPairList"}],
        "freqai": {
            "enabled": True,
            "activate_tensorboard": False,
            "purge_old_models": 2,
            "train_period_days": train_days,
            "backtest_period_days": backtest_days,
            "live_retrain_hours": 0,
            "identifier": identifier,
            "feature_parameters": {
                "include_timeframes": ["1h"],
                "include_corr_pairlist": [],
                "label_period_candles": 24,
                "include_shifted_candles": 0,
                "DI_threshold": 0,
                "weight_factor": 0.9,
                "principal_component_analysis": False,
                "use_SVM_to_remove_outliers": False,
                "indicator_periods_candles": [6],
                "plot_feature_importances": int(plot_feature_importances),
            },
            "data_split_parameters": {"test_size": 0.25, "random_state": 1, "shuffle": False},
            "model_training_parameters": {
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
        "bot_name": "formatted_freqai_hypothesis_research",
        "force_entry_enable": True,
        "initial_state": "stopped",
        "internals": {"process_throttle_secs": 5},
    }


def score_manifest(manifest: dict[str, Any], models_dir: Path) -> pd.DataFrame:
    actuals = _actual_labels(_canonical_pair(str(manifest.get("pair") or PAIR)))
    actuals = actuals.rename(columns={target: f"{target}_actual" for target in TARGETS if target in actuals})
    rows: list[dict[str, Any]] = []
    for item in manifest["commands"]:
        predictions = _load_predictions(models_dir / str(item["identifier"]))
        if predictions.empty:
            rows.append({**_row_base(item), "target": "", "scope": "all_predicted_rows", "status": "missing_predictions"})
            continue
        merged = predictions.merge(actuals, on="date", how="left")
        if "do_predict" in merged:
            merged = merged[pd.to_numeric(merged["do_predict"], errors="coerce").eq(1.0)].copy()
        scopes = [("all_predicted_rows", pd.Series(True, index=merged.index))]
        if "score_live_media_ready" in merged:
            scopes.append((
                "live_media_ready",
                pd.to_numeric(merged["score_live_media_ready"], errors="coerce").fillna(0.0).eq(1.0),
            ))
        if "score_live_media_orderbook_ready" in merged:
            scopes.append((
                "live_media_orderbook_ready",
                pd.to_numeric(merged["score_live_media_orderbook_ready"], errors="coerce").fillna(0.0).eq(1.0),
            ))
        if "score_live_media_orderbook_loose_ready" in merged:
            scopes.append((
                "live_media_orderbook_loose_ready",
                pd.to_numeric(merged["score_live_media_orderbook_loose_ready"], errors="coerce").fillna(0.0).eq(1.0),
            ))
        for scope, scope_mask in scopes:
            scoped = merged.loc[scope_mask].copy()
            for target in TARGETS:
                rows.append(_score_target(item, scoped, target, scope))
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    control_columns = [
        "target",
        "scope",
        "roc_auc",
        "average_precision",
        "prediction_actual_corr",
        "top_quintile_actual_rate",
        "bottom_quintile_actual_rate",
    ]
    if any(column not in summary.columns for column in control_columns):
        return summary
    control = summary[
        summary["profile_id"].eq("price_control")
        & summary["status"].eq("scored")
    ][control_columns]
    control = control.rename(
        columns={
            "roc_auc": "control_roc_auc",
            "average_precision": "control_average_precision",
            "prediction_actual_corr": "control_prediction_actual_corr",
            "top_quintile_actual_rate": "control_top_quintile_actual_rate",
            "bottom_quintile_actual_rate": "control_bottom_quintile_actual_rate",
        }
    )
    summary = summary.merge(control, on=["target", "scope"], how="left")
    for metric in ("roc_auc", "average_precision", "prediction_actual_corr", "top_quintile_actual_rate", "bottom_quintile_actual_rate"):
        control_metric = f"control_{metric}"
        if metric in summary and control_metric in summary:
            summary[f"{metric}_delta_vs_price"] = pd.to_numeric(summary[metric], errors="coerce") - pd.to_numeric(summary[control_metric], errors="coerce")
    return summary


def _actual_labels(pair_canonical: str = PAIR_CANONICAL) -> pd.DataFrame:
    frame = pd.read_parquet(
        SNAPSHOT_PATH,
        columns=[
            "date",
            "pair",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "context_news_web_present_24h",
            "context_global_present",
            "conf_orderbook_ready_market_count",
            "orderbook_usable_loose",
            "orderbook_tick_present",
        ],
    )
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = frame[frame["pair"].astype(str).str.upper().eq(pair_canonical)].copy()
    frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    frame["score_live_media_ready"] = (
        pd.to_numeric(frame.get("context_news_web_present_24h"), errors="coerce").fillna(0.0).gt(0.0)
        | pd.to_numeric(frame.get("context_global_present"), errors="coerce").fillna(0.0).gt(0.0)
    ).astype(float)
    frame["score_live_media_orderbook_ready"] = (
        frame["score_live_media_ready"].eq(1.0)
        & pd.to_numeric(frame.get("conf_orderbook_ready_market_count"), errors="coerce").fillna(0.0).gt(0.0)
    ).astype(float)
    frame["score_live_media_orderbook_loose_ready"] = (
        frame["score_live_media_ready"].eq(1.0)
        & (
            pd.to_numeric(frame.get("orderbook_usable_loose"), errors="coerce").fillna(0.0).gt(0.0)
            | pd.to_numeric(frame.get("orderbook_tick_present"), errors="coerce").fillna(0.0).gt(0.0)
        )
    ).astype(float)
    for horizon in TARGET_HORIZONS:
        frame[f"&-future_return_{horizon}h"] = frame["close"].shift(-horizon) / frame["close"] - 1.0
        upside = FormattedContextHypothesisFreqAIBaseStrategy._future_extreme(frame["high"], horizon, "max") / frame["close"] - 1.0
        drawdown = FormattedContextHypothesisFreqAIBaseStrategy._future_extreme(frame["low"], horizon, "min") / frame["close"] - 1.0
        frame[f"&-future_max_upside_{horizon}h"] = upside
        frame[f"&-future_max_drawdown_{horizon}h"] = drawdown
        reward, danger, large_move = FormattedContextHypothesisFreqAIBaseStrategy._path_thresholds(horizon)
        frame[f"&-long_reward_before_danger_{horizon}h"] = FormattedContextHypothesisFreqAIBaseStrategy._reward_before_danger(
            frame["close"],
            horizon=horizon,
            reward=reward,
            danger=-danger,
            long_side=True,
        )
        frame[f"&-short_reward_before_danger_{horizon}h"] = FormattedContextHypothesisFreqAIBaseStrategy._reward_before_danger(
            frame["close"],
            horizon=horizon,
            reward=reward,
            danger=-danger,
            long_side=False,
        )
        frame[f"&-large_upside_next_{horizon}h"] = (
            frame[f"&-future_max_upside_{horizon}h"].ge(large_move).astype(float).where(frame[f"&-future_max_upside_{horizon}h"].notna())
        )
        frame[f"&-large_drawdown_next_{horizon}h"] = (
            frame[f"&-future_max_drawdown_{horizon}h"].le(-large_move).astype(float).where(frame[f"&-future_max_drawdown_{horizon}h"].notna())
        )
    frame = FormattedContextHypothesisFreqAIBaseStrategy._append_structure_path_labels(frame)
    keep = [
        "date",
        "score_live_media_ready",
        "score_live_media_orderbook_ready",
        "score_live_media_orderbook_loose_ready",
        *[target for target in TARGETS if target in frame],
    ]
    return frame[keep]


def _load_predictions(model_dir: Path) -> pd.DataFrame:
    prediction_dir = model_dir / "backtesting_predictions"
    files = sorted(prediction_dir.glob("*_prediction.feather"))
    frames = []
    for path in files:
        frame = pd.read_feather(path)
        frame["prediction_file"] = path.name
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    predictions = pd.concat(frames, ignore_index=True)
    predictions["date"] = pd.to_datetime(predictions["date"], utc=True, errors="coerce")
    return predictions.dropna(subset=["date"]).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def _score_target(item: dict[str, Any], merged: pd.DataFrame, target: str, scope: str) -> dict[str, Any]:
    row = {**_row_base(item), "target": target, "scope": scope}
    if target not in merged:
        row.update({"status": "missing_prediction_column", "rows": 0})
        return row
    actual_col = f"{target}_actual"
    if actual_col not in merged:
        row.update({"status": "missing_actual_column", "rows": 0})
        return row
    valid = merged[[target, actual_col]].apply(pd.to_numeric, errors="coerce").dropna()
    row["rows"] = int(len(valid))
    if valid.empty or valid[target].nunique() < 2:
        row.update({"status": "unscorable", "actual_event_rate": None})
        return row
    row["status"] = "scored"
    row["actual_event_rate"] = float(valid[actual_col].mean())
    ranked = valid.sort_values(target)
    bucket_size = max(1, int(len(ranked) * 0.2))
    bottom = ranked.head(bucket_size)
    top = ranked.tail(bucket_size)
    row["top_quintile_actual_rate"] = float(top[actual_col].mean())
    row["bottom_quintile_actual_rate"] = float(bottom[actual_col].mean())
    row["prediction_actual_corr"] = float(valid[target].corr(valid[actual_col]))
    actual_values = set(valid[actual_col].dropna().astype(float).unique())
    if actual_values.issubset({0.0, 1.0}) and len(actual_values) == 2:
        row["roc_auc"] = float(roc_auc_score(valid[actual_col].astype(int), valid[target]))
        row["average_precision"] = float(average_precision_score(valid[actual_col].astype(int), valid[target]))
    else:
        row["roc_auc"] = None
        row["average_precision"] = None
    return row


def _row_base(item: dict[str, Any]) -> dict[str, Any]:
    return {
        "profile_id": item.get("profile_id"),
        "identifier": item.get("identifier"),
        "strategy": item.get("strategy"),
        "hypothesis_group": item.get("hypothesis_group"),
        "theory": item.get("theory"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
