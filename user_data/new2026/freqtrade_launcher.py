#!/usr/bin/env python3
"""
Minimal Freqtrade launcher GUI.

Purpose
- Remove repetitive CLI typing during backtest / hyperopt / dry-run / live-run work.
- Preserve normal Freqtrade behaviour by launching the documented CLI commands.
- Support an in-process launch path so an IDE debugger can step into both the
  strategy and Freqtrade internals when this launcher itself is started under
  the debugger.

Important design constraints
- This launcher only exposes inputs that normally belong at the Freqtrade CLI
  or configuration layer. It intentionally does not edit strategy-level
  parameters.
- The `trade` command does not provide a `--pairs` CLI override. When a manual
  pair list is requested for live or dry-run sessions, the launcher creates a
  temporary overlay config with `exchange.pair_whitelist` / `pair_blacklist`
  and passes it as the final `-c` file. This matches Freqtrade's documented
  multi-config merge model, where later config files override earlier ones.
- In-process execution is best suited to debugging. Repeated in-process runs
  inside one Python session may retain imported module state. If a later debug
  run behaves oddly, restart the launcher and rerun.
"""

from __future__ import annotations

import ast
import codecs
import csv
import io
import importlib.util
import json
import os
import queue
import random
import runpy
import shlex
import signal
import subprocess
import sqlite3
import sys
import tempfile
import threading
import time
import traceback
import webbrowser
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request
from zipfile import BadZipFile, ZipFile

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, simpledialog, ttk

from orderbook_metrics import estimate_storage_usage, normalize_freqtrade_pair_to_binance_symbol, normalize_whitelist_pairs


for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


APP_TITLE = "Freqtrade Launcher"
APP_DIR = Path(__file__).resolve().parent
PRESET_FILE = "freqtrade_launcher_presets.json"
DEFAULT_STRATEGY_FILE = "../strategies/HybridRecoveryGridStrategy_v11.py"
NEWS_COLLECTOR_FILE = "research/collectors/news_research_collector.py"
NEWS_SOURCES_FILE = "research/config/news_research_sources.json"
NEWS_DATA_DIR = "../runtime/news"
NEWS_DB_FILE = "news_events.sqlite"
NEWS_STATUS_FILE = "collector_status.json"
NEWS_PID_FILE = "collector.pid"
NEWS_STOP_FILE = "collector.stop"
NEWS_LOG_FILE = "news_collector.log"
WEB_COLLECTOR_FILE = "research/collectors/web_research_collector.py"
WEB_SOURCES_FILE = "research/config/web_research_sources.json"
WEB_DATA_DIR = "../runtime/web"
WEB_DB_FILE = "web_events.sqlite"
WEB_STATUS_FILE = "collector_status.json"
WEB_PID_FILE = "collector.pid"
WEB_STOP_FILE = "collector.stop"
WEB_LOG_FILE = "web_collector.log"
ORDERBOOK_COLLECTOR_FILE = "orderbook_collector.py"
ORDERBOOK_SOURCES_FILE = "orderbook_sources.json"
ORDERBOOK_DATA_DIR = "../runtime/orderbook"
ORDERBOOK_DB_FILE = "orderbook_events.sqlite"
ORDERBOOK_STATUS_FILE = "collector_status.json"
ORDERBOOK_PID_FILE = "collector.pid"
ORDERBOOK_STOP_FILE = "collector.stop"
ORDERBOOK_LOG_FILE = "orderbook_collector.log"
AUTO_LAST_USED_PRESET = "__auto_last_used_state__"
LAST_USED_STATE_FILE = "../runtime/freqtrade_launcher_state.json"
ADJUST_EVENT_MARKER = "ADJUST_EVENT_JSON"
EXPLORER_STATUS_MARKER = "EXPLORER_STATUS_JSON"


def utf8_subprocess_env(extra: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    if extra:
        env.update(extra)
    return env
EXPLORER_RUNNER_FILE = "hyperopt_explorer_runner.py"
EXPLORER_MARKET_WINDOWS_FILE = "explorer_market_windows.json"
EXPLORER_METADATA_DIR = "../runtime/explorer_metadata"
EXPLORER_LATEST_SUMMARY_FILE = "latest_summary.json"
EXPLORER_CUSTOM_BATCH_FILE = "hyperopt_custom_batches.json"
EXPLORER_SELECTION_MODES = ["Random families", "Random tags", "Random namespace values", "Custom batches"]
MAX_HYPEROPT_EPOCHS = 800


def app_path(relative_path: str | Path) -> Path:
    return APP_DIR / Path(relative_path)

RUN_TYPE_TO_COMMAND = {
    "Live": "trade",
    "Dry-run": "trade",
    "Backtest": "backtesting",
    "Lookahead Analysis": "lookahead-analysis",
    "Hyperopt": "hyperopt",
    "Download Data": "download-data",
}

BACKENDS = ["Subprocess", "In-process (debug)"]
RUN_TYPES = list(RUN_TYPE_TO_COMMAND.keys())
VERBOSE_VALUES = ["0", "1", "2", "3"]
LOG_LEVEL_VALUES = ["INFO", "VERBOSE", "DEBUG", "TRACE"]
LOG_LEVEL_TO_VERBOSE = {"INFO": "0", "VERBOSE": "1", "DEBUG": "2", "TRACE": "3"}
VERBOSE_TO_LOG_LEVEL = {"0": "INFO", "1": "VERBOSE", "2": "DEBUG", "3": "TRACE"}
EXPORT_VALUES = ["default", "none", "trades", "signals"]
BREAKDOWN_VALUES = ["none", "day", "week", "month", "year", "weekday"]
DATA_FORMAT_VALUES = ["", "json", "jsongz", "feather", "parquet"]
TRADING_MODE_VALUES = ["", "spot", "margin", "futures"]
RESEARCH_LAB_CRYPTO_GROUPS = {
    "crypto",
    "crypto_media",
    "exchange_announcements",
    "protocol_foundations",
    "defi_protocols",
    "infrastructure",
    "aggregators",
}
RESEARCH_LAB_OFFICIAL_CONFIDENCE = {"official_source", "official_page"}
RESEARCH_LAB_SOURCE_HEALTH_BASE_COLUMNS = (
    "source_id",
    "source_group",
    "enabled",
    "last_success_at",
    "last_failure_at",
    "last_error",
    "items_last_fetch",
    "inserted_last_fetch",
    "duplicates_last_fetch",
)
RESEARCH_LAB_SOURCE_HEALTH_SCORE_COLUMNS = (
    "source_id",
    "source_group",
    "enabled",
    "market_relevance",
    "source_priority",
    "usefulness_score",
    "avg_priority_score",
    "high_priority_articles",
    "duplicate_ratio",
    "last_success_at",
    "last_failure_at",
    "last_error",
    "items_last_fetch",
    "inserted_last_fetch",
    "duplicates_last_fetch",
)
HYPEROPT_SPACE_VALUES = [
    "default",
    "all",
    "buy",
    "sell",
    "roi",
    "stoploss",
    "trailing",
    "protection",
    "trades",
    "buy sell",
    "buy sell roi stoploss trailing",
]
HYPEROPT_LOSS_VALUES = [
    "ShortTradeDurHyperOptLoss",
    "OnlyProfitHyperOptLoss",
    "SharpeHyperOptLoss",
    "SharpeHyperOptLossDaily",
    "SortinoHyperOptLoss",
    "SortinoHyperOptLossDaily",
    "CalmarHyperOptLoss",
    "MaxDrawDownHyperOptLoss",
    "MaxDrawDownRelativeHyperOptLoss",
    "MaxDrawDownPerPairHyperOptLoss",
    "ProfitDrawDownHyperOptLoss",
    "MultiMetricHyperOptLoss",
]
BUILTIN_HYPEROPT_LOSS_DESCRIPTIONS = {
    "ShortTradeDurHyperOptLoss": "Default-style objective that rewards profit while penalizing long average trade duration and too few trades.",
    "OnlyProfitHyperOptLoss": "Optimizes total profit only. Useful as a simple baseline, but it ignores drawdown, trade duration, and trade count quality.",
    "SharpeHyperOptLoss": "Optimizes the Sharpe ratio across trades, balancing profit against volatility of returns.",
    "SharpeHyperOptLossDaily": "Optimizes daily Sharpe ratio, balancing daily returns against day-to-day volatility.",
    "SortinoHyperOptLoss": "Optimizes the Sortino ratio across trades, focusing risk pressure on downside volatility.",
    "SortinoHyperOptLossDaily": "Optimizes daily Sortino ratio, focusing on downside volatility in daily returns.",
    "CalmarHyperOptLoss": "Optimizes Calmar ratio, emphasizing profit relative to maximum drawdown.",
    "MaxDrawDownHyperOptLoss": "Focuses on reducing absolute maximum drawdown during the backtest period.",
    "MaxDrawDownRelativeHyperOptLoss": "Focuses on reducing relative maximum drawdown as a percentage of equity.",
    "MaxDrawDownPerPairHyperOptLoss": "Focuses on drawdown control per pair, helping avoid one pair dominating risk.",
    "ProfitDrawDownHyperOptLoss": "Balances profit seeking with drawdown control.",
    "MultiMetricHyperOptLoss": "Combines several quality metrics such as profit, drawdown, profit factor, expectancy, winrate, and trade count.",
}
SPACES_HINT = "Built-ins vary by Freqtrade version. Common values: default, all, buy, sell, roi, stoploss, trailing, protection, trades. Custom spaces only work on versions that support them."
REVIEW_BREAKDOWN_VALUES = ["none", "day", "week", "month", "year", "weekday"]
ANALYSIS_GROUP_HINT = "0 1 2 3 4 5"
EXPLORER_WINDOW_MODE_VALUES = ["regime", "all"]
EXPLORER_REGIME_VALUES = ["bull", "bear", "chop", "crash", "crossover"]
EXPLORER_MARKET_TYPE_LABELS = {
    "bull": "Bull",
    "bear": "Bear",
    "chop": "Chop",
    "crash": "Crash",
    "crossover": "Crossover",
}
EXPLORER_MARKET_TYPE_COMPACT_LABELS = {
    "bull": "Bull",
    "bear": "Bear",
    "chop": "Chop",
    "crash": "Crash",
    "crossover": "Xover",
}
EXPLORER_12M_HOLDOUT_WINDOWS = [
    {
        "name": "holdout_12m_20200101_20210101",
        "label": "20200101-20210101",
        "timerange": "20200101-20210101",
        "regime": "holdout",
        "segment_type": "holdout_12m",
        "market_sequence": "chop -> COVID crash -> recovery bull -> late-2020 breakout",
        "btc_low_note": "Approx low: $3.8k during the March 2020 COVID liquidity crash.",
        "btc_high_note": "Approx high: $29k near the end of December 2020.",
        "event_note": "COVID liquidity crash, aggressive global liquidity response, DeFi summer, institutional accumulation, late-2020 breakout.",
        "tooltip": "20200101-20210101\nMarket sequence: chop -> COVID crash -> recovery bull -> late-2020 breakout\nBTC range: approx $3.8k low to $29k high\nNotes: COVID liquidity crash, liquidity response, institutional accumulation, late-2020 breakout.",
    },
    {
        "name": "holdout_12m_20200601_20210601",
        "label": "20200601-20210601",
        "timerange": "20200601-20210601",
        "regime": "holdout",
        "segment_type": "holdout_12m",
        "market_sequence": "post-COVID recovery -> late-2020 breakout -> early-2021 institutional bull -> May 2021 crash",
        "btc_low_note": "Approx low: $8.8k around early June 2020.",
        "btc_high_note": "Approx high: $64.9k around April 2021.",
        "event_note": "Post-COVID recovery, institutional/Tesla narrative, Coinbase listing cycle, then May 2021 leverage and China-mining/regulatory selloff.",
        "tooltip": "20200601-20210601\nMarket sequence: post-COVID recovery -> late-2020 breakout -> early-2021 institutional bull -> May 2021 crash\nBTC range: approx $8.8k low to $64.9k high\nNotes: institutional/Tesla narrative, Coinbase listing cycle, May 2021 crash.",
    },
    {
        "name": "holdout_12m_20210101_20220101",
        "label": "20210101-20220101",
        "timerange": "20210101-20220101",
        "regime": "holdout",
        "segment_type": "holdout_12m",
        "market_sequence": "institutional bull -> May crash -> summer recovery -> November ATH -> risk-off pullback",
        "btc_low_note": "Approx low: $28.8k during the June/July 2021 correction.",
        "btc_high_note": "Approx high: $68.8k at the November 2021 ATH.",
        "event_note": "Early-2021 institutional rally, May crash, China mining ban effects, El Salvador adoption, November ATH, Fed tightening/taper expectations.",
        "tooltip": "20210101-20220101\nMarket sequence: institutional bull -> May crash -> summer recovery -> November ATH -> risk-off pullback\nBTC range: approx $28.8k low to $68.8k high\nNotes: May crash, China mining ban effects, El Salvador adoption, November ATH, Fed tightening fears.",
    },
    {
        "name": "holdout_12m_20210601_20220601",
        "label": "20210601-20220601",
        "timerange": "20210601-20220601",
        "regime": "holdout",
        "segment_type": "holdout_12m",
        "market_sequence": "post-May crash recovery -> second-leg bull -> ATH -> bear transition -> Terra/LUNA crash",
        "btc_low_note": "Approx low: $25k-$26k during the May 2022 Terra/LUNA selloff.",
        "btc_high_note": "Approx high: $68.8k at the November 2021 ATH.",
        "event_note": "Second-leg 2021 bull run, November ATH, macro tightening, risk-off rotation, Terra/LUNA collapse in May 2022.",
        "tooltip": "20210601-20220601\nMarket sequence: post-May crash recovery -> second-leg bull -> ATH -> bear transition -> Terra/LUNA crash\nBTC range: approx $25k-$26k low to $68.8k high\nNotes: November ATH, macro tightening, Terra/LUNA collapse.",
    },
    {
        "name": "holdout_12m_20220101_20230101",
        "label": "20220101-20230101",
        "timerange": "20220101-20230101",
        "regime": "holdout",
        "segment_type": "holdout_12m",
        "market_sequence": "bear drift -> Terra/LUNA crash -> 3AC/credit contagion -> FTX breakdown -> cycle low",
        "btc_low_note": "Approx low: $15.5k after the FTX collapse in November 2022.",
        "btc_high_note": "Approx high: $48k around late March 2022.",
        "event_note": "Fed tightening cycle, Terra/LUNA collapse, 3AC/credit contagion, FTX collapse, deep crypto bear-market sentiment.",
        "tooltip": "20220101-20230101\nMarket sequence: bear drift -> Terra/LUNA crash -> 3AC contagion -> FTX breakdown -> cycle low\nBTC range: approx $15.5k low to $48k high\nNotes: Terra/LUNA, 3AC contagion, FTX collapse, 2022 bear market.",
    },
    {
        "name": "holdout_12m_20220601_20230601",
        "label": "20220601-20230601",
        "timerange": "20220601-20230601",
        "regime": "holdout",
        "segment_type": "holdout_12m",
        "market_sequence": "post-LUNA bear -> FTX crash -> bottoming range -> Q1 2023 recovery",
        "btc_low_note": "Approx low: $15.5k after the FTX collapse in November 2022.",
        "btc_high_note": "Approx high: $31k-$32k around June 2022 / April 2023 recovery highs.",
        "event_note": "Post-LUNA deleveraging, Celsius/3AC aftershocks, FTX collapse, early-2023 recovery from cycle lows.",
        "tooltip": "20220601-20230601\nMarket sequence: post-LUNA bear -> FTX crash -> bottoming range -> Q1 2023 recovery\nBTC range: approx $15.5k low to $31k-$32k high\nNotes: post-LUNA deleveraging, FTX collapse, Q1 2023 recovery.",
    },
    {
        "name": "holdout_12m_20230101_20240101",
        "label": "20230101-20240101",
        "timerange": "20230101-20240101",
        "regime": "holdout",
        "segment_type": "holdout_12m",
        "market_sequence": "bear-bottom recovery -> spring/summer range -> ETF anticipation bull",
        "btc_low_note": "Approx low: $16.5k near the start of January 2023.",
        "btc_high_note": "Approx high: $44k-$45k in December 2023.",
        "event_note": "Recovery from 2022 lows, regional-bank/liquidity narrative, long range/compression, spot Bitcoin ETF anticipation into year-end.",
        "tooltip": "20230101-20240101\nMarket sequence: bear-bottom recovery -> spring/summer range -> ETF anticipation bull\nBTC range: approx $16.5k low to $44k-$45k high\nNotes: 2022-low recovery, range/compression, spot ETF anticipation.",
    },
    {
        "name": "holdout_12m_20230601_20240601",
        "label": "20230601-20240601",
        "timerange": "20230601-20240601",
        "regime": "holdout",
        "segment_type": "holdout_12m",
        "market_sequence": "compression -> ETF approval rally -> March 2024 ATH -> halving cooldown",
        "btc_low_note": "Approx low: $24k-$25k during the September 2023 compression/drawdown.",
        "btc_high_note": "Approx high: $73k-$74k around the March 2024 pre-halving ATH.",
        "event_note": "ETF anticipation, SEC spot Bitcoin ETF approval in January 2024, strong ETF inflows, March ATH, April 2024 halving, post-halving cooldown.",
        "tooltip": "20230601-20240601\nMarket sequence: compression -> ETF approval rally -> March 2024 ATH -> halving cooldown\nBTC range: approx $24k-$25k low to $73k-$74k high\nNotes: ETF approval, ETF inflows, March ATH, April halving.",
    },
    {
        "name": "holdout_12m_20240101_20250101",
        "label": "20240101-20250101",
        "timerange": "20240101-20250101",
        "regime": "holdout",
        "segment_type": "holdout_12m",
        "market_sequence": "ETF approval bull -> post-halving range -> August drawdown -> Q4 breakout",
        "btc_low_note": "Approx low: $38k-$39k during the January 2024 ETF sell-the-news pullback.",
        "btc_high_note": "Approx high: about $108k in December 2024.",
        "event_note": "Spot ETF launch, March ATH, April halving, post-halving chop, August drawdown, late-2024 election/pro-crypto and institutional rally.",
        "tooltip": "20240101-20250101\nMarket sequence: ETF approval bull -> post-halving range -> August drawdown -> Q4 breakout\nBTC range: approx $38k-$39k low to about $108k high\nNotes: ETF launch, March ATH, halving, August drawdown, late-2024 breakout.",
    },
    {
        "name": "holdout_12m_20240601_20250601",
        "label": "20240601-20250601",
        "timerange": "20240601-20250601",
        "regime": "holdout",
        "segment_type": "holdout_12m",
        "market_sequence": "post-halving chop -> August drawdown -> late-2024 breakout -> early-2025 correction/recovery",
        "btc_low_note": "Approx low: $49k-$50k during the August 2024 drawdown.",
        "btc_high_note": "Approx high: about $109k around January 2025.",
        "event_note": "Post-halving cooldown, August 2024 risk-off drawdown, Q4 2024 breakout, January 2025 high, February/March 2025 tariff/Bybit/risk-off correction.",
        "tooltip": "20240601-20250601\nMarket sequence: post-halving chop -> August drawdown -> late-2024 breakout -> early-2025 correction/recovery\nBTC range: approx $49k-$50k low to about $109k high\nNotes: post-halving chop, August drawdown, Q4 breakout, early-2025 correction.",
    },
    {
        "name": "holdout_12m_20250101_20260101",
        "label": "20250101-20260101",
        "timerange": "20250101-20260101",
        "regime": "holdout",
        "segment_type": "holdout_12m",
        "market_sequence": "early-2025 drawdown/recovery -> mid-2025 bull -> October ATH -> liquidation crash -> year-end bear reset",
        "btc_low_note": "Approx low: $74k-$76k during the 2025 risk-off drawdown phase.",
        "btc_high_note": "Approx high: about $126k during the October 2025 ATH.",
        "event_note": "Early-2025 tariff/Bybit/risk-off drawdown, mid-2025 recovery, October ATH above $126k, October 10 liquidation shock, weak November/December reset.",
        "tooltip": "20250101-20260101\nMarket sequence: early-2025 drawdown/recovery -> mid-2025 bull -> October ATH -> liquidation crash -> year-end bear reset\nBTC range: approx $74k-$76k low to about $126k high\nNotes: early-2025 risk-off, October ATH, October liquidation shock, year-end reset.",
    },
]
EXPLORER_DECISION_TREE_LABELS = {
    "Objective-only (champion-relative)": "objective",
}
EXPLORER_DECISION_TREE_KEYS = {value: key for key, value in EXPLORER_DECISION_TREE_LABELS.items()}
CONSOLE_SEARCH_KEY_TERMS = [
    "Acceptance",
    "Decision: ACCEPTED",
    "best=True",
    "REJECTED",
    "Starting backtest w:",
    "Backtest comparison",
    "Explorer loop",
    "HyperOpt window:",
    "Backtest windows:",
    "weighted_acceptance_score",
    "params_changed",
    "result_missing",
    "backtest_failed",
    "Traceback",
    "Error",
]
EXPLORER_SUMMARY_PREFIXES = (
    "Explorer loop",
    "HyperOpt window:",
    "Backtest windows:",
    "Selected random tag pool for this Explorer launch:",
    "Selected random family pool for this Explorer launch:",
    "Selected custom batches for this Explorer launch:",
    "Run-level selected target:",
    "Run-level selected targets:",
    "Target:",
    "Mode: Custom batches",
    "Params:",
    "Spaces:",
    "Excluded:",
    "Active HyperOpt params:",
    "Min-param padding tags:",
    "Min-param padding families:",
    "Warning:",
    "Collected challenger:",
    "No challengers",
    "Champion backtest baseline:",
    "Champion baseline reused from cache:",
    "Champion baseline backtest executed:",
    "Starting champion baseline backtest",
    "Champion backtest failed",
    "=== Challenger",
    "HyperOpt loss:",
    "Starting backtest with challenger",
    "Backtest failed",
    "HyperOpt failed",
    "Explorer target",
    "No hyperopt result file found",
    "Could not read loss",
    "Search run for",
    "Acceptance:",
    "Accepted",
    "Champion updated",
    "No challenger beat the champion baseline",
)
EXPLORER_SUMMARY_TABLE_HEADERS = (
    "Backtest comparison",
    "Role          Weighted score",
    "Window                              Champion Profit",
)
EXPLORER_SUMMARY_TABLE_ROW_PREFIXES = (
    "Champion",
    "Challenger",
    "Delta",
    "holdout_",
)
PAIR_REFERENCE_GROUPS: list[dict[str, Any]] = [
    {
        "name": "Top 10 Market Cap",
        "note": "Static USDT futures reference list based on large-cap crypto names. For live ranking, use Freqtrade MarketCapPairList.",
        "pairs": ["BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "TRX", "AVAX", "LINK"],
    },
    {
        "name": "Top 100 Market Cap",
        "note": "Broad static large/mid-cap reference list. Exchange availability can vary.",
        "pairs": [
            "BTC", "ETH", "BNB", "SOL", "XRP", "ADA", "DOGE", "TRX", "AVAX", "LINK",
            "TON", "DOT", "MATIC", "BCH", "LTC", "SHIB", "UNI", "ATOM", "XLM", "ETC",
            "ICP", "FIL", "APT", "ARB", "NEAR", "OP", "HBAR", "VET", "INJ", "IMX",
            "TIA", "SUI", "SEI", "TAO", "RENDER", "RNDR", "AAVE", "MKR", "RUNE", "GRT",
            "ALGO", "EGLD", "STX", "KAS", "FLOW", "QNT", "SAND", "MANA", "AXS", "CHZ",
            "APE", "GALA", "DYDX", "SNX", "LDO", "CRV", "COMP", "SUSHI", "YFI", "1INCH",
            "ZEC", "DASH", "XMR", "ROSE", "MINA", "WLD", "FET", "AGIX", "OCEAN", "ARKM",
            "PYTH", "JUP", "PENDLE", "ENA", "WIF", "PEPE", "BONK", "FLOKI", "MEME", "ORDI",
            "BOME", "NOT", "STRK", "ZK", "ZRO", "BLUR", "ENS", "JTO", "JASMY", "IOTA",
            "KAVA", "WAVES", "CFX", "ONE", "LRC", "BAT", "ZIL", "CELO", "AR", "GMT",
        ],
    },
    {
        "name": "Stable Coins",
        "note": "Stablecoin quote/base references. Many are not useful for directional futures strategies but are handy for exclusions.",
        "pairs": ["USDC", "FDUSD", "TUSD", "DAI", "USDE", "USDP", "PYUSD", "EUR", "EURI", "AEUR"],
    },
    {
        "name": "DeFi Coins",
        "note": "DEX, lending, yield, and DeFi governance names.",
        "pairs": ["UNI", "AAVE", "MKR", "LDO", "CRV", "COMP", "SNX", "SUSHI", "YFI", "1INCH", "PENDLE", "ENA", "DYDX", "GMX", "CAKE"],
    },
    {
        "name": "Meme Coins",
        "note": "High social-beta meme basket.",
        "pairs": ["DOGE", "SHIB", "PEPE", "WIF", "BONK", "FLOKI", "MEME", "BOME", "TURBO", "MEW", "PNUT", "ACT"],
    },
    {
        "name": "Top Volume Candidates",
        "note": "Static proxy for liquid pairs. For actual exchange volume, use Freqtrade VolumePairList.",
        "pairs": ["BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "ADA", "LINK", "AVAX", "WIF", "PEPE", "SUI", "NEAR", "LTC", "BCH"],
    },
    {
        "name": "High Volatility Candidates",
        "note": "Static high-beta basket. For candle-derived volatility filtering, use Freqtrade VolatilityFilter.",
        "pairs": ["WIF", "PEPE", "BONK", "BOME", "DOGE", "SHIB", "ORDI", "TIA", "SEI", "SUI", "INJ", "JUP", "PENDLE", "ENA", "STRK"],
    },
    {
        "name": "Layer 1 Majors",
        "note": "Base-layer networks.",
        "pairs": ["BTC", "ETH", "SOL", "BNB", "ADA", "AVAX", "TRX", "DOT", "ATOM", "NEAR", "ICP", "APT", "SUI", "SEI", "ALGO"],
    },
    {
        "name": "Layer 2 / Scaling",
        "note": "Ethereum scaling and rollup-related names.",
        "pairs": ["ARB", "OP", "MATIC", "STRK", "ZK", "ZRO", "IMX", "LRC", "METIS", "MANTA", "ALT"],
    },
    {
        "name": "AI / Data",
        "note": "AI, data, compute, and indexing narratives.",
        "pairs": ["TAO", "RENDER", "RNDR", "FET", "AGIX", "OCEAN", "ARKM", "GRT", "WLD", "NMR", "PHB", "AI"],
    },
    {
        "name": "Gaming / Metaverse",
        "note": "Game, metaverse, and entertainment tokens.",
        "pairs": ["IMX", "SAND", "MANA", "AXS", "GALA", "APE", "GMT", "MAGIC", "YGG", "PIXEL", "PORTAL"],
    },
    {
        "name": "Privacy Coins",
        "note": "Privacy-oriented networks. Some exchanges restrict these by region.",
        "pairs": ["XMR", "ZEC", "DASH", "ROSE", "SCRT", "ZEN", "MINA"],
    },
    {
        "name": "Exchange Tokens",
        "note": "Centralized and decentralized exchange-related tokens.",
        "pairs": ["BNB", "OKB", "KCS", "CRO", "GT", "FTT", "DYDX", "UNI", "CAKE", "GMX", "JUP"],
    },
    {
        "name": "RWA / Tokenization",
        "note": "Real-world asset and institutional/tokenization narrative names.",
        "pairs": ["ONDO", "MKR", "PENDLE", "ENA", "CFG", "POLYX", "OM", "TRU", "GFI"],
    },
    {
        "name": "Payments / Transfers",
        "note": "Payments, settlement, and fast-transfer networks.",
        "pairs": ["XRP", "XLM", "LTC", "BCH", "TRX", "DASH", "CELO", "HBAR", "IOTA", "ALGO"],
    },
    {
        "name": "Infrastructure / Oracles",
        "note": "Oracle, storage, messaging, and infrastructure names.",
        "pairs": ["LINK", "PYTH", "BAND", "API3", "FIL", "AR", "GRT", "AKT", "ANKR", "WAVES", "IOTX"],
    },
]

@dataclass
class BuildResult:
    """Fully prepared launch information.

    command_args contains only the arguments after `python -m freqtrade`.
    preview_command is the full command shown in the GUI.
    temp_config_path is populated only when a temporary overlay config was
    created for pair overrides in live / dry-run mode.
    """

    command_args: list[str]
    preview_command: list[str]
    temp_config_path: str | None = None


@dataclass
class HyperoptLossInfo:
    name: str
    source: str
    description: str
    path: str | None = None


@dataclass(frozen=True)
class CollectorProfile:
    key: str
    label: str
    collector_file: str
    sources_file: str
    data_dir_name: str
    db_file: str
    status_file: str
    pid_file: str
    stop_file: str
    log_file: str
    default_interval_minutes: int


NEWS_COLLECTOR_PROFILE = CollectorProfile(
    key="news",
    label="News Lab",
    collector_file=NEWS_COLLECTOR_FILE,
    sources_file=NEWS_SOURCES_FILE,
    data_dir_name=NEWS_DATA_DIR,
    db_file=NEWS_DB_FILE,
    status_file=NEWS_STATUS_FILE,
    pid_file=NEWS_PID_FILE,
    stop_file=NEWS_STOP_FILE,
    log_file=NEWS_LOG_FILE,
    default_interval_minutes=15,
)

WEB_COLLECTOR_PROFILE = CollectorProfile(
    key="web",
    label="Web Lab",
    collector_file=WEB_COLLECTOR_FILE,
    sources_file=WEB_SOURCES_FILE,
    data_dir_name=WEB_DATA_DIR,
    db_file=WEB_DB_FILE,
    status_file=WEB_STATUS_FILE,
    pid_file=WEB_PID_FILE,
    stop_file=WEB_STOP_FILE,
    log_file=WEB_LOG_FILE,
    default_interval_minutes=360,
)



class ToolTip:
    """Minimal hover tooltip for Tk widgets.

    Purpose
    - Keep batch-tab option meanings close to the controls without forcing a big
      permanent wall of text into the UI.
    - Stay lightweight and dependency-free.
    """

    def __init__(self, widget: tk.Widget, text: str) -> None:
        self.widget = widget
        self.text = text
        self.tipwindow: tk.Toplevel | None = None
        self._show_after_id: str | None = None
        widget.bind("<Enter>", self._schedule_show, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")
        widget.bind("<Destroy>", self._hide, add="+")

    def _schedule_show(self, _event: tk.Event | None = None) -> None:
        self._cancel_schedule()
        self._show_after_id = self.widget.after(450, self._show)

    def _cancel_schedule(self) -> None:
        if self._show_after_id is not None:
            try:
                self.widget.after_cancel(self._show_after_id)
            except Exception:
                pass
            self._show_after_id = None

    def _show(self) -> None:
        self._cancel_schedule()
        if self.tipwindow is not None or not self.text:
            return
        try:
            x = self.widget.winfo_rootx() + 18
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
        except Exception:
            return
        self.tipwindow = tk.Toplevel(self.widget)
        self.tipwindow.wm_overrideredirect(True)
        self.tipwindow.wm_geometry(f"+{x}+{y}")
        label = tk.Label(
            self.tipwindow,
            text=self.text,
            justify="left",
            background="#ffffe0",
            relief="solid",
            borderwidth=1,
            wraplength=420,
            padx=8,
            pady=6,
        )
        label.pack()

    def _hide(self, _event: tk.Event | None = None) -> None:
        self._cancel_schedule()
        if self.tipwindow is not None:
            try:
                self.tipwindow.destroy()
            except Exception:
                pass
            self.tipwindow = None


class QueueWriter(io.TextIOBase):
    """Redirect text output into the GUI queue.

    Freqtrade prints to stdout / stderr during normal CLI execution. Redirecting
    those streams lets the launcher show the exact same output in the console
    panel while still running in-process for debugger visibility.
    """

    def __init__(self, output_queue: queue.Queue[tuple[str, str]]) -> None:
        super().__init__()
        self.output_queue = output_queue

    def write(self, s: str) -> int:
        if s:
            self.output_queue.put(("stdout", s))
        return len(s)

    def flush(self) -> None:
        return None


class OrderedPathList(ttk.LabelFrame):
    """Simple ordered list editor for repeated path arguments.

    This is used for the Freqtrade `-c/--config` list where file order matters,
    since later config files override earlier ones.
    """

    def __init__(self, master: tk.Misc, title: str, filetypes: list[tuple[str, str]]) -> None:
        super().__init__(master, text=title)
        self.filetypes = filetypes
        self.title = title

        self.listbox = tk.Listbox(self, height=6, exportselection=False)
        self.listbox.grid(row=0, column=0, rowspan=5, sticky="nsew", padx=(8, 4), pady=8)

        scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.listbox.yview)
        scrollbar.grid(row=0, column=1, rowspan=5, sticky="ns", pady=8)
        self.listbox.configure(yscrollcommand=scrollbar.set)

        add_button = ttk.Button(self, text="Add", command=self.add_items)
        add_button.grid(row=0, column=2, sticky="ew", padx=8, pady=(8, 4))
        remove_button = ttk.Button(self, text="Remove", command=self.remove_selected)
        remove_button.grid(row=1, column=2, sticky="ew", padx=8, pady=4)
        up_button = ttk.Button(self, text="Up", command=lambda: self.move_selected(-1))
        up_button.grid(row=2, column=2, sticky="ew", padx=8, pady=4)
        down_button = ttk.Button(self, text="Down", command=lambda: self.move_selected(1))
        down_button.grid(row=3, column=2, sticky="ew", padx=8, pady=4)
        clear_button = ttk.Button(self, text="Clear", command=self.clear)
        clear_button.grid(row=4, column=2, sticky="ew", padx=8, pady=(4, 8))

        ToolTip(self.listbox, f"Ordered {title.lower()}. Later files override earlier ones when Freqtrade loads them.")
        ToolTip(add_button, f"Add one or more {title.lower()} entries.")
        ToolTip(remove_button, f"Remove the selected {title.lower()} entry.")
        ToolTip(up_button, f"Move the selected {title.lower()} entry earlier in the load order.")
        ToolTip(down_button, f"Move the selected {title.lower()} entry later in the load order.")
        ToolTip(clear_button, f"Remove all {title.lower()} entries.")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.on_change: callable[[], None] | None = None

    def set_on_change(self, callback: callable[[], None]) -> None:
        self.on_change = callback

    def _notify_change(self) -> None:
        if self.on_change:
            self.on_change()

    def add_items(self) -> None:
        paths = filedialog.askopenfilenames(filetypes=self.filetypes)
        for path in paths:
            self.listbox.insert(tk.END, path)
        if paths:
            self._notify_change()

    def remove_selected(self) -> None:
        selected = list(self.listbox.curselection())
        for idx in reversed(selected):
            self.listbox.delete(idx)
        if selected:
            self._notify_change()

    def move_selected(self, delta: int) -> None:
        selected = self.listbox.curselection()
        if len(selected) != 1:
            return
        idx = selected[0]
        new_idx = idx + delta
        if new_idx < 0 or new_idx >= self.listbox.size():
            return
        value = self.listbox.get(idx)
        self.listbox.delete(idx)
        self.listbox.insert(new_idx, value)
        self.listbox.selection_set(new_idx)
        self._notify_change()

    def clear(self) -> None:
        if self.listbox.size() == 0:
            return
        self.listbox.delete(0, tk.END)
        self._notify_change()

    def get_items(self) -> list[str]:
        return list(self.listbox.get(0, tk.END))

    def set_items(self, items: Iterable[str]) -> None:
        self.listbox.delete(0, tk.END)
        for item in items:
            self.listbox.insert(tk.END, item)
        self._notify_change()


class FreqtradeLauncher(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1240x920")
        self.minsize(1080, 760)

        self.output_queue: queue.Queue[tuple[str, str]] = queue.Queue()
        self.current_process: subprocess.Popen[str] | None = None
        self.worker_thread: threading.Thread | None = None
        self.monitor_thread: threading.Thread | None = None
        self.monitor_stop_event: threading.Event | None = None
        self.current_temp_config: str | None = None
        self.presets_path = str(app_path(PRESET_FILE))
        self.last_used_state_path = str(app_path(LAST_USED_STATE_FILE))
        self.presets: dict[str, dict[str, Any]] = {}
        self.stdout_parse_buffer = ""
        self.adjust_events: list[dict[str, Any]] = []
        self.explorer_summary_block = ""
        self.explorer_summary_section = ""
        self.hyperopt_loss_info_by_name: dict[str, HyperoptLossInfo] = {}
        self.hyperopt_loss_scan_key: tuple[str, str, str] | None = None
        self.hyperopt_loss_combo: ttk.Combobox | None = None
        self.explorer_hyperopt_loss_combo: ttk.Combobox | None = None
        self.explorer_tab_run_button: ttk.Button | None = None
        self.explorer_target_count_label_widget: ttk.Label | None = None
        self.explorer_target_count_entry_widget: ttk.Entry | None = None
        self.explorer_namespace_label_widget: ttk.Label | None = None
        self.explorer_min_params_label_widget: ttk.Label | None = None
        self.explorer_min_params_entry_widget: ttk.Entry | None = None
        self.news_source_health_tree: ttk.Treeview | None = None
        self.web_source_health_tree: ttk.Treeview | None = None
        self.orderbook_pair_preview_tree: ttk.Treeview | None = None
        self.orderbook_latest_metrics_tree: ttk.Treeview | None = None
        self.frequi_launch_button: ttk.Button | None = None
        self._orderbook_last_capacity_popup_level = "ok"

        self.explorer_total_runs = 0
        self.explorer_current_run = 0
        self.explorer_last_status: dict[str, Any] = {}
        self.explorer_current_hyperopt_window = "-"
        self.explorer_accepted_count = 0
        self.explorer_rejected_count = 0
        self.explorer_loop_count = 0
        self.explorer_loop_log_entries = 0
        self.explorer_current_loop_targets: dict[int, dict[str, Any]] = {}
        self.explorer_initial_champion_stats: dict[str, Any] = {}
        self.explorer_current_champion_stats: dict[str, Any] = {}
        self.active_run_type: str = ""

        self._build_variables()
        self._build_ui()
        self._load_presets_from_disk()
        self.refresh_hyperopt_loss_dropdown()
        self._bind_variable_traces()
        self.refresh_mode_options()
        self.refresh_all_command_previews()
        self.refresh_news_status()
        self.refresh_web_status()
        self.refresh_orderbook_pair_preview()
        self.refresh_orderbook_status(show_popup=False)
        self.after(100, self._drain_output_queue)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_variables(self) -> None:
        self.run_type_var = tk.StringVar(value="Backtest")
        self.backend_var = tk.StringVar(value=BACKENDS[0])
        self.preset_var = tk.StringVar(value="")

        self.project_root_var = tk.StringVar(value="")
        self.python_exe_var = tk.StringVar(value=sys.executable)
        self.userdir_var = tk.StringVar(value="")
        self.datadir_var = tk.StringVar(value="")
        self.logfile_var = tk.StringVar(value="")
        self.verbose_var = tk.StringVar(value="0")
        self.log_level_var = tk.StringVar(value="INFO")
        self.no_color_var = tk.BooleanVar(value=False)

        self.strategy_file_var = tk.StringVar(value=str(app_path(DEFAULT_STRATEGY_FILE)))
        self.strategy_class_var = tk.StringVar(value="")
        self.recursive_strategy_search_var = tk.BooleanVar(value=False)

        self.pair_mode_var = tk.StringVar(value="config")
        self.pair_reference_format_var = tk.StringVar(value="USDT futures")
        self.pair_reference_selected_var = tk.StringVar(value="")
        self.pair_reference_tree: ttk.Treeview | None = None
        self.pair_reference_details: scrolledtext.ScrolledText | None = None

        self.db_url_var = tk.StringVar(value="")
        self.fee_var = tk.StringVar(value="")
        self.dry_run_wallet_var = tk.StringVar(value="")
        self.stake_amount_var = tk.StringVar(value="")
        self.max_open_trades_var = tk.StringVar(value="")

        self.timeframe_var = tk.StringVar(value="")
        self.timerange_var = tk.StringVar(value="")
        self.timeframe_detail_var = tk.StringVar(value="")
        self.enable_protections_var = tk.BooleanVar(value=False)
        self.enable_position_stacking_var = tk.BooleanVar(value=False)

        self.backtest_export_var = tk.StringVar(value="signals")
        self.backtest_breakdown_var = tk.StringVar(value="none")
        self.lookahead_min_trade_amount_var = tk.StringVar(value="")
        self.lookahead_targeted_trade_amount_var = tk.StringVar(value="")
        self.lookahead_export_filename_var = tk.StringVar(value="")
        self.lookahead_allow_limit_orders_var = tk.BooleanVar(value=False)

        self.hyperopt_epochs_var = tk.StringVar(value="100")
        self.hyperopt_spaces_var = tk.StringVar(value="default")
        self.hyperopt_jobs_var = tk.StringVar(value="")
        self.hyperopt_random_state_var = tk.StringVar(value="")
        self.hyperopt_min_trades_var = tk.StringVar(value="")
        self.hyperopt_loss_var = tk.StringVar(value="")
        self.hyperopt_early_stop_var = tk.StringVar(value="")
        self.hyperopt_ignore_missing_spaces_var = tk.BooleanVar(value=False)
        self.hyperopt_disable_param_export_var = tk.BooleanVar(value=False)
        self.hyperopt_analyze_per_epoch_var = tk.BooleanVar(value=False)
        self.hyperopt_print_all_var = tk.BooleanVar(value=False)
        self.hyperopt_monitor_results_var = tk.BooleanVar(value=False)
        self.hyperopt_results_file_var = tk.StringVar(value="")
        self.hyperopt_results_poll_ms_var = tk.StringVar(value="1000")

        self.explorer_sampling_seed_var = tk.StringVar(value="")
        self.explorer_max_loops_var = tk.StringVar(value="")
        self.explorer_epochs_var = tk.StringVar(value="")
        self.explorer_random_state_var = tk.StringVar(value="")
        self.explorer_backtest_workers_var = tk.StringVar(value="12")
        self.explorer_window_vars: dict[str, tk.BooleanVar] = {}
        self.explorer_hyperopt_market_type_vars: dict[str, tk.BooleanVar] = {
            value: tk.BooleanVar(value=True) for value in EXPLORER_REGIME_VALUES
        }
        self.explorer_backtest_market_type_vars: dict[str, tk.BooleanVar] = {
            value: tk.BooleanVar(value=True) for value in EXPLORER_REGIME_VALUES
        }
        self.explorer_hyperopt_market_type_buttons: dict[str, tk.Button] = {}
        self.explorer_backtest_market_type_buttons: dict[str, tk.Button] = {}
        self.explorer_selection_mode_var = tk.StringVar(value="Random families")
        self.explorer_target_count_var = tk.StringVar(value="3")
        self.explorer_target_namespace_var = tk.StringVar(value="mode")
        self.explorer_target_namespace_combo: ttk.Combobox | None = None
        self.explorer_target_count_tooltips: list[ToolTip] = []
        self.explorer_family_count_var = tk.StringVar(value="3")
        self.explorer_tag_count_var = tk.StringVar(value="5")
        self.explorer_min_param_count_var = tk.StringVar(value="0")
        self.explorer_keeper_enabled_var = tk.BooleanVar(value=True)
        self.explorer_keeper_save_dir_var = tk.StringVar(value="")
        self.explorer_keeper_win_numerator_var = tk.StringVar(value="5")
        self.explorer_keeper_win_denominator_var = tk.StringVar(value="6")
        self.explorer_keeper_min_profit_per_window_var = tk.StringVar(value="200")
        self.catalog_view_var = tk.StringVar(value="By Family")
        self.catalog_search_var = tk.StringVar(value="")
        self.catalog_filter_var = tk.StringVar(value="Show all")
        self.custom_batch_name_var = tk.StringVar(value="")
        self.custom_batch_description_var = tk.StringVar(value="")
        self.custom_batch_status_var = tk.StringVar(value="")
        self.custom_batch_editing_id: str | None = None
        self.custom_batch_draft_sources: list[dict[str, Any]] = []
        self.custom_batch_draft_excluded: list[str] = []
        self.custom_batch_selected_run_ids: set[str] = set()
        self.custom_batches_payload: dict[str, Any] = {"version": 1, "updated_at": "", "batches": []}
        self.catalog_data: dict[str, Any] = {}
        self.catalog_current_values: dict[str, Any] = {}
        self.catalog_tree_item_data: dict[str, dict[str, Any]] = {}
        self.catalog_tree: ttk.Treeview | None = None
        self.catalog_details_text: scrolledtext.ScrolledText | None = None
        self.catalog_resolved_tree: ttk.Treeview | None = None
        self.custom_batch_sources_list: tk.Listbox | None = None
        self.custom_batch_excluded_list: tk.Listbox | None = None
        self.custom_batch_committed_tree: ttk.Treeview | None = None
        self.explorer_backtest_count_vars: dict[str, tk.StringVar] = {
            value: tk.StringVar(value="0") for value in EXPLORER_REGIME_VALUES
        }
        self.explorer_12m_holdout_vars: dict[str, tk.BooleanVar] = {
            str(window["label"]): tk.BooleanVar(value=str(window["label"]) == "20240101-20250101")
            for window in EXPLORER_12M_HOLDOUT_WINDOWS
        }

        self.console_search_var = tk.StringVar(value="")
        self.console_key_term_var = tk.StringVar(value="")
        self.console_search_status_var = tk.StringVar(value="")
        self.console_search_case_var = tk.BooleanVar(value=False)
        self.console_follow_tail_var = tk.BooleanVar(value=True)
        self.explorer_summary_search_var = tk.StringVar(value="")
        self.explorer_summary_search_status_var = tk.StringVar(value="")
        self.explorer_summary_search_case_var = tk.BooleanVar(value=False)
        self.explorer_summary_follow_tail_var = tk.BooleanVar(value=True)
        self.explorer_summary_context_var = tk.StringVar(value="Loss: - | HyperOpt window: -")
        self.explorer_tally_var = tk.StringVar(value=self._default_explorer_tally_text())

        self.download_exchange_var = tk.StringVar(value="")
        self.download_pairs_file_var = tk.StringVar(value="")
        self.download_pairs_text: scrolledtext.ScrolledText | None = None
        self.download_timeframes_var = tk.StringVar(value="")
        self.download_days_var = tk.StringVar(value="")
        self.download_new_pairs_days_var = tk.StringVar(value="")
        self.download_timerange_var = tk.StringVar(value="")
        self.download_trading_mode_var = tk.StringVar(value="")
        self.download_candle_types_var = tk.StringVar(value="")
        self.download_data_format_ohlcv_var = tk.StringVar(value="")
        self.download_data_format_trades_var = tk.StringVar(value="")
        self.download_include_inactive_var = tk.BooleanVar(value=False)
        self.download_no_parallel_var = tk.BooleanVar(value=False)
        self.download_dl_trades_var = tk.BooleanVar(value=False)
        self.download_convert_var = tk.BooleanVar(value=False)
        self.download_erase_var = tk.BooleanVar(value=False)
        self.download_prepend_var = tk.BooleanVar(value=False)

        self.frequi_url_var = tk.StringVar(value="http://127.0.0.1:8080")
        self.frequi_open_after_launch_var = tk.BooleanVar(value=True)
        self.frequi_status_var = tk.StringVar(value="Stopped")

        self.news_config_path_var = tk.StringVar(value=str(app_path(NEWS_SOURCES_FILE)))
        self.news_data_dir_var = tk.StringVar(value=str(app_path(NEWS_DATA_DIR)))
        self.news_db_path_var = tk.StringVar(value=str(app_path(NEWS_DATA_DIR) / NEWS_DB_FILE))
        self.news_interval_seconds_var = tk.StringVar(value="15")
        self.news_once_var = tk.BooleanVar(value=False)
        self.news_command_preview_var = tk.StringVar(value="")
        self.news_status_var = tk.StringVar(value="unknown")
        self.news_pid_var = tk.StringVar(value="-")
        self.news_started_at_var = tk.StringVar(value="-")
        self.news_heartbeat_at_var = tk.StringVar(value="-")
        self.news_last_fetch_at_var = tk.StringVar(value="-")
        self.news_total_articles_var = tk.StringVar(value="-")
        self.news_new_articles_last_cycle_var = tk.StringVar(value="-")
        self.news_last_error_var = tk.StringVar(value="-")
        self.news_scoring_note_var = tk.StringVar(value="")
        self.news_summary_vars = {
            "total_articles": tk.StringVar(value="-"),
            "high_priority_articles": tk.StringVar(value="-"),
            "crypto_articles": tk.StringVar(value="-"),
            "macro_articles": tk.StringVar(value="-"),
            "official_source_articles": tk.StringVar(value="-"),
            "failing_sources": tk.StringVar(value="-"),
            "top_source_group": tk.StringVar(value="-"),
        }
        self.news_health_filter_vars = {
            "text": tk.StringVar(value=""),
            "enabled_only": tk.BooleanVar(value=False),
            "failing_only": tk.BooleanVar(value=False),
            "crypto_only": tk.BooleanVar(value=False),
            "official_only": tk.BooleanVar(value=False),
            "disable_candidates_only": tk.BooleanVar(value=False),
        }
        self.web_config_path_var = tk.StringVar(value=str(app_path(WEB_SOURCES_FILE)))
        self.web_data_dir_var = tk.StringVar(value=str(app_path(WEB_DATA_DIR)))
        self.web_db_path_var = tk.StringVar(value=str(app_path(WEB_DATA_DIR) / WEB_DB_FILE))
        self.web_interval_seconds_var = tk.StringVar(value="360")
        self.web_once_var = tk.BooleanVar(value=False)
        self.web_command_preview_var = tk.StringVar(value="")
        self.web_status_var = tk.StringVar(value="unknown")
        self.web_pid_var = tk.StringVar(value="-")
        self.web_started_at_var = tk.StringVar(value="-")
        self.web_heartbeat_at_var = tk.StringVar(value="-")
        self.web_last_fetch_at_var = tk.StringVar(value="-")
        self.web_total_articles_var = tk.StringVar(value="-")
        self.web_new_articles_last_cycle_var = tk.StringVar(value="-")
        self.web_last_error_var = tk.StringVar(value="-")
        self.web_scoring_note_var = tk.StringVar(value="")
        self.web_summary_vars = {
            "total_articles": tk.StringVar(value="-"),
            "high_priority_articles": tk.StringVar(value="-"),
            "crypto_articles": tk.StringVar(value="-"),
            "macro_articles": tk.StringVar(value="-"),
            "official_source_articles": tk.StringVar(value="-"),
            "failing_sources": tk.StringVar(value="-"),
            "top_source_group": tk.StringVar(value="-"),
        }
        self.web_health_filter_vars = {
            "text": tk.StringVar(value=""),
            "enabled_only": tk.BooleanVar(value=False),
            "failing_only": tk.BooleanVar(value=False),
            "crypto_only": tk.BooleanVar(value=False),
            "official_only": tk.BooleanVar(value=False),
            "disable_candidates_only": tk.BooleanVar(value=False),
        }
        self.orderbook_config_path_var = tk.StringVar(value=str(app_path(ORDERBOOK_SOURCES_FILE)))
        self.orderbook_data_dir_var = tk.StringVar(value=str(app_path(ORDERBOOK_DATA_DIR)))
        self.orderbook_exchange_var = tk.StringVar(value="Binance USD-M Futures")
        self.orderbook_market_type_var = tk.StringVar(value="futures")
        self.orderbook_depth_levels_var = tk.StringVar(value="20")
        self.orderbook_stream_update_ms_var = tk.StringVar(value="500")
        self.orderbook_metric_interval_seconds_var = tk.StringVar(value="1")
        self.orderbook_snapshot_interval_seconds_var = tk.StringVar(value="60")
        self.orderbook_store_snapshots_var = tk.BooleanVar(value=True)
        self.orderbook_capacity_warning_mb_var = tk.StringVar(value="500")
        self.orderbook_capacity_critical_mb_var = tk.StringVar(value="2000")
        self.orderbook_max_symbols_var = tk.StringVar(value="12")
        self.orderbook_status_var = tk.StringVar(value="unknown")
        self.orderbook_pid_var = tk.StringVar(value="-")
        self.orderbook_started_at_var = tk.StringVar(value="-")
        self.orderbook_heartbeat_at_var = tk.StringVar(value="-")
        self.orderbook_last_message_at_var = tk.StringVar(value="-")
        self.orderbook_last_metric_at_var = tk.StringVar(value="-")
        self.orderbook_pair_count_var = tk.StringVar(value="-")
        self.orderbook_active_streams_var = tk.StringVar(value="-")
        self.orderbook_message_count_var = tk.StringVar(value="-")
        self.orderbook_metric_count_var = tk.StringVar(value="-")
        self.orderbook_db_mb_var = tk.StringVar(value="-")
        self.orderbook_data_dir_mb_var = tk.StringVar(value="-")
        self.orderbook_estimated_mb_per_day_var = tk.StringVar(value="-")
        self.orderbook_capacity_level_var = tk.StringVar(value="-")
        self.orderbook_last_error_var = tk.StringVar(value="-")
        self.orderbook_estimated_metric_rows_var = tk.StringVar(value="-")
        self.orderbook_estimated_snapshot_rows_var = tk.StringVar(value="-")
        self.orderbook_estimated_days_to_warning_var = tk.StringVar(value="-")
        self.orderbook_preview_pair_count_var = tk.StringVar(value="0")
        self.orderbook_preview_symbol_count_var = tk.StringVar(value="0")
        self.orderbook_pair_warning_var = tk.StringVar(value="")

        self.review_hyperopt_file_var = tk.StringVar(value="")
        self.review_hyperopt_limit_var = tk.StringVar(value="20")
        self.review_hyperopt_index_var = tk.StringVar(value="-1")
        self.review_hyperopt_best_var = tk.BooleanVar(value=False)
        self.review_hyperopt_profitable_var = tk.BooleanVar(value=False)
        self.review_hyperopt_print_json_var = tk.BooleanVar(value=False)
        self.review_hyperopt_no_header_var = tk.BooleanVar(value=False)
        self.review_hyperopt_no_details_var = tk.BooleanVar(value=False)
        self.review_hyperopt_breakdown_var = tk.StringVar(value="none")

        self.review_backtest_file_var = tk.StringVar(value="")
        self.review_backtest_directory_var = tk.StringVar(value="")
        self.review_backtest_show_pair_list_var = tk.BooleanVar(value=False)
        self.review_backtest_breakdown_var = tk.StringVar(value="none")
        self.review_analysis_groups_var = tk.StringVar(value="0 1 2 5")
        self.review_enter_reasons_var = tk.StringVar(value="")
        self.review_exit_reasons_var = tk.StringVar(value="")
        self.review_indicator_list_var = tk.StringVar(value="")
        self.review_entry_only_var = tk.BooleanVar(value=False)
        self.review_exit_only_var = tk.BooleanVar(value=False)
        self.review_rejected_signals_var = tk.BooleanVar(value=False)
        self.review_analysis_to_csv_var = tk.BooleanVar(value=False)
        self.review_analysis_csv_path_var = tk.StringVar(value="")

        self.command_preview_var = tk.StringVar(value="")

    def _build_ui(self) -> None:
        notebook = ttk.Notebook(self)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        self.tab_run = ttk.Frame(notebook)
        self.tab_common = ttk.Frame(notebook)
        self.tab_pairs = ttk.Frame(notebook)
        self.tab_mode = ttk.Frame(notebook)
        self.tab_download = ttk.Frame(notebook)
        self.tab_news_lab = ttk.Frame(notebook)
        self.tab_web_lab = ttk.Frame(notebook)
        self.tab_orderbook_lab = ttk.Frame(notebook)
        self.tab_explorer = ttk.Frame(notebook)
        self.tab_review = ttk.Frame(notebook)

        notebook.add(self.tab_run, text="Run")
        notebook.add(self.tab_common, text="Common CLI")
        notebook.add(self.tab_pairs, text="Pairs")
        notebook.add(self.tab_mode, text="Mode Options")
        notebook.add(self.tab_download, text="Download Data")
        notebook.add(self.tab_news_lab, text="News Lab")
        notebook.add(self.tab_web_lab, text="Web Lab")
        notebook.add(self.tab_orderbook_lab, text="Order Book Lab")
        notebook.add(self.tab_explorer, text="Explorer")
        notebook.add(self.tab_review, text="Review Results")

        self._build_run_tab()
        self._build_common_tab()
        self._build_pairs_tab()
        self._build_mode_tab()
        self._build_download_tab()
        self._build_news_lab_tab()
        self._build_web_lab_tab()
        self._build_orderbook_lab_tab()
        self._build_explorer_tab()
        self._build_review_tab()

    def _build_run_tab(self) -> None:
        root = self.tab_run
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(3, weight=1)

        controls = ttk.LabelFrame(root, text="Session")
        controls.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        for i in range(8):
            controls.grid_columnconfigure(i, weight=1 if i in (1, 3, 5, 7) else 0)

        run_type_label = ttk.Label(controls, text="Run type")
        run_type_label.grid(row=0, column=0, sticky="w", padx=8, pady=8)
        run_type_combo = ttk.Combobox(controls, textvariable=self.run_type_var, values=RUN_TYPES, state="readonly", width=20)
        run_type_combo.grid(row=0, column=1, sticky="ew", padx=8, pady=8)

        execution_label = ttk.Label(controls, text="Execution")
        execution_label.grid(row=0, column=2, sticky="w", padx=8, pady=8)
        execution_combo = ttk.Combobox(controls, textvariable=self.backend_var, values=BACKENDS, state="readonly", width=20)
        execution_combo.grid(row=0, column=3, sticky="ew", padx=8, pady=8)

        preset_label = ttk.Label(controls, text="Preset")
        preset_label.grid(row=0, column=4, sticky="w", padx=8, pady=8)
        self.preset_combo = ttk.Combobox(controls, textvariable=self.preset_var, state="readonly")
        self.preset_combo.grid(row=0, column=5, sticky="ew", padx=8, pady=8)

        load_button = ttk.Button(controls, text="Load", command=self.load_selected_preset)
        load_button.grid(row=0, column=6, sticky="ew", padx=8, pady=8)
        save_button = ttk.Button(controls, text="Save", command=self.save_preset)
        save_button.grid(row=0, column=7, sticky="ew", padx=8, pady=8)
        timerange_label, timerange_entry = self._labeled_entry(controls, 1, 0, "Timerange", self.timerange_var)
        save_as_button = ttk.Button(controls, text="Save As", command=self.save_preset_as)
        save_as_button.grid(row=1, column=6, sticky="ew", padx=8, pady=(0, 8))
        delete_button = ttk.Button(controls, text="Delete", command=self.delete_selected_preset)
        delete_button.grid(row=1, column=7, sticky="ew", padx=8, pady=(0, 8))
        self._add_tooltips(
            "Context: Primary workflow selector for launcher command building. Outcome: Chooses which command template and tab options are applied. "
            "Example: Select `Backtest` before setting timerange and export options.",
            run_type_label,
            run_type_combo,
        )
        self._add_tooltips(
            "Context: Process execution backend selector. Outcome: Determines whether command runs as subprocess or in-process debug mode. "
            "Example: Choose `In-process (debug)` to step through strategy code.",
            execution_label,
            execution_combo,
        )
        self._add_tooltips(
            "Context: Choose a saved launcher profile. Outcome: The selected preset name becomes the source for Load/Save actions. "
            "Example: Pick 'BTC_15m_Backtest' before pressing Load.",
            preset_label,
            self.preset_combo,
        )
        self._add_tooltips(
            "Context: Import all values from the selected preset into the current UI. Outcome: Current fields are replaced by that preset. "
            "Example: Select 'Dryrun_Default' then Load to switch quickly.",
            load_button,
        )
        self._add_tooltips(
            "Context: Persist current UI values into the currently selected preset. Outcome: That preset is overwritten with your latest settings. "
            "Example: Change timeframe to 15m and press Save.",
            save_button,
        )
        self._add_tooltips(
            "Context: Create a new named preset from the current UI state. Outcome: A new preset entry is added without deleting others. "
            "Example: Save As 'ETH_Hyperopt_Test'.",
            save_as_button,
        )
        self._add_tooltips(
            "Context: Remove an existing preset entry. Outcome: The selected preset is deleted from preset storage. "
            "Example: Delete an outdated one-off test preset.",
            delete_button,
        )
        self._add_tooltips(
            "Context: Global run date/candle slice used by multiple command types. Outcome: Adds --timerange when filled; blank omits it. "
            "Example: 20240101-20240331 or -200.",
            timerange_label,
            timerange_entry,
        )

        actions = ttk.Frame(root)
        actions.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        self.run_button = ttk.Button(actions, text="Run", command=self.on_run)
        self.run_button.pack(side="left", padx=(0, 8))
        self.explorer_run_button = ttk.Button(actions, text="Run Explorer", command=self.on_run_explorer)
        self.explorer_run_button.pack(side="left", padx=(0, 8))
        self.stop_button = ttk.Button(actions, text="Stop", command=self.on_stop, state="disabled")
        self.stop_button.pack(side="left")
        ttk.Label(actions, text="Run this launcher under the PyCharm debugger and select In-process (debug) to step into strategy / Freqtrade code.").pack(side="left", padx=16)
        self._add_tooltips(
            "Context: Primary launcher action for the selected Run type. Outcome: Builds command preview, then executes the command. "
            "Example: Run Backtest with current pairs and timerange.",
            self.run_button,
        )
        self._add_tooltips(
            "Context: Starts the Explorer workflow from the Explorer tab configuration. Outcome: Runs explorer runner with current explorer options. "
            "Example: Start random-tag exploration with selected windows.",
            self.explorer_run_button,
        )
        self._add_tooltips(
            "Context: Emergency/control stop for the active launcher process slot. Outcome: Sends stop/terminate flow and returns UI to idle. "
            "Example: Stop a long hyperopt run early.",
            self.stop_button,
        )

        preview = ttk.LabelFrame(root, text="Generated command")
        preview.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        preview.grid_columnconfigure(0, weight=1)
        preview_entry = ttk.Entry(preview, textvariable=self.command_preview_var)
        preview_entry.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        ToolTip(
            preview_entry,
            "Context: Final command generated from current UI fields. Outcome: Shows exactly what Run will execute; this field is read-only. "
            "Example: `python -m freqtrade backtesting -i 5m --timerange 20240101-20240331`.",
        )

        console = ttk.LabelFrame(root, text="Console")
        console.grid(row=3, column=0, sticky="nsew", padx=8, pady=(0, 8))
        console.grid_columnconfigure(0, weight=1)
        console.grid_rowconfigure(0, weight=1)

        console_notebook = ttk.Notebook(console)
        console_notebook.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)

        raw_console_tab = ttk.Frame(console_notebook)
        raw_console_tab.grid_columnconfigure(0, weight=1)
        raw_console_tab.grid_rowconfigure(1, weight=1)
        console_notebook.add(raw_console_tab, text="Raw console")

        explorer_summary_tab = ttk.Frame(console_notebook)
        explorer_summary_tab.grid_columnconfigure(0, weight=1)
        explorer_summary_tab.grid_columnconfigure(1, weight=0)
        explorer_summary_tab.grid_rowconfigure(1, weight=1)
        console_notebook.add(explorer_summary_tab, text="Explorer summary")

        console_tools = ttk.Frame(raw_console_tab)
        console_tools.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 0))
        console_tools.grid_columnconfigure(3, weight=1)
        console_find_label = ttk.Label(console_tools, text="Find")
        console_find_label.grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.console_key_term_combo = ttk.Combobox(
            console_tools,
            textvariable=self.console_key_term_var,
            values=CONSOLE_SEARCH_KEY_TERMS,
            state="readonly",
            width=30,
        )
        self.console_key_term_combo.grid(row=0, column=1, sticky="ew", padx=(0, 6))
        self.console_key_term_combo.bind("<<ComboboxSelected>>", lambda _event: self.apply_console_key_term())
        console_or_label = ttk.Label(console_tools, text="or")
        console_or_label.grid(row=0, column=2, sticky="w", padx=(0, 6))
        self.console_search_entry = ttk.Entry(console_tools, textvariable=self.console_search_var)
        self.console_search_entry.grid(row=0, column=3, sticky="ew", padx=(0, 6))
        self.console_search_entry.bind("<Return>", lambda _event: self.find_console_next())
        self.console_search_entry.bind("<Shift-Return>", lambda _event: self.find_console_previous())
        prev_button = ttk.Button(console_tools, text="Previous", command=self.find_console_previous)
        prev_button.grid(row=0, column=4, sticky="ew", padx=(0, 6))
        next_button = ttk.Button(console_tools, text="Next", command=self.find_console_next)
        next_button.grid(row=0, column=5, sticky="ew", padx=(0, 18))
        clear_button = ttk.Button(console_tools, text="Clear", command=self.clear_console_search)
        clear_button.grid(row=0, column=6, sticky="ew", padx=(0, 6))
        bottom_button = ttk.Button(console_tools, text="Bottom", command=self.scroll_console_to_bottom)
        bottom_button.grid(row=0, column=7, sticky="ew", padx=(0, 6))
        follow_tail = ttk.Checkbutton(console_tools, text="Follow tail", variable=self.console_follow_tail_var)
        follow_tail.grid(row=0, column=8, sticky="w", padx=(0, 6))
        case_check = ttk.Checkbutton(console_tools, text="Case", variable=self.console_search_case_var)
        case_check.grid(row=0, column=9, sticky="w", padx=(0, 6))
        ttk.Label(console_tools, textvariable=self.console_search_status_var, foreground="#666666").grid(row=0, column=10, sticky="w")
        self._add_tooltips(
            "Context: Search tools for the live raw console stream. Outcome: Highlight and jump between matches while output continues. "
            "Example: Search for 'Traceback' or choose 'Decision: ACCEPTED'.",
            console_find_label,
            self.console_key_term_combo,
            console_or_label,
            self.console_search_entry,
        )
        self._add_tooltips(
            "Context: Navigate search results. Outcome: Moves selection to the previous highlighted match. "
            "Example: Step backward through repeated 'ERROR' lines.",
            prev_button,
        )
        self._add_tooltips(
            "Context: Navigate search results. Outcome: Moves selection to the next highlighted match. "
            "Example: Step forward through each acceptance decision.",
            next_button,
        )
        self._add_tooltips(
            "Context: Reset current search state. Outcome: Clears query and match highlights in the raw console. "
            "Example: Clear after finishing a 'Traceback' scan.",
            clear_button,
        )
        self._add_tooltips(
            "Context: Output navigation aid. Outcome: Scrolls directly to the newest console lines. "
            "Example: Jump to latest log after reviewing older messages.",
            bottom_button,
        )
        self._add_tooltips(
            "Context: Live monitoring mode. Outcome: Keeps viewport pinned to newest output while enabled. "
            "Example: Watch hyperopt progress in real time.",
            follow_tail,
        )
        self._add_tooltips(
            "Context: Search matching mode. Outcome: Enables case-sensitive matching when checked. "
            "Example: Distinguish 'Error' from 'error'.",
            case_check,
        )

        self.console = scrolledtext.ScrolledText(raw_console_tab, wrap="word", height=30)
        self.console.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        self.console.tag_configure("search_match", background="#fff2a8")
        self.console.tag_configure("search_active", background="#f6a23a", foreground="#000000")
        self.console.configure(state="disabled")

        explorer_summary_tools = ttk.Frame(explorer_summary_tab)
        explorer_summary_tools.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 0))
        explorer_summary_tools.grid_columnconfigure(1, weight=1)
        explorer_find_label = ttk.Label(explorer_summary_tools, text="Find")
        explorer_find_label.grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.explorer_summary_search_entry = ttk.Entry(explorer_summary_tools, textvariable=self.explorer_summary_search_var)
        self.explorer_summary_search_entry.grid(row=0, column=1, sticky="ew", padx=(0, 6))
        self.explorer_summary_search_entry.bind("<Return>", lambda _event: self.find_explorer_summary_next())
        self.explorer_summary_search_entry.bind("<Shift-Return>", lambda _event: self.find_explorer_summary_previous())
        accepted_button = ttk.Button(explorer_summary_tools, text="Find Accepted", command=self.find_explorer_summary_accepted)
        accepted_button.grid(row=0, column=2, sticky="ew", padx=(0, 6))
        summary_prev_button = ttk.Button(explorer_summary_tools, text="Previous", command=self.find_explorer_summary_previous)
        summary_prev_button.grid(row=0, column=3, sticky="ew", padx=(0, 6))
        summary_next_button = ttk.Button(explorer_summary_tools, text="Next", command=self.find_explorer_summary_next)
        summary_next_button.grid(row=0, column=4, sticky="ew", padx=(0, 6))
        summary_clear_button = ttk.Button(explorer_summary_tools, text="Clear", command=self.clear_explorer_summary)
        summary_clear_button.grid(row=0, column=5, sticky="ew", padx=(0, 6))
        summary_bottom_button = ttk.Button(explorer_summary_tools, text="Bottom", command=self.scroll_explorer_summary_to_bottom)
        summary_bottom_button.grid(row=0, column=6, sticky="ew", padx=(0, 6))
        summary_follow_tail = ttk.Checkbutton(explorer_summary_tools, text="Follow tail", variable=self.explorer_summary_follow_tail_var)
        summary_follow_tail.grid(row=0, column=7, sticky="w", padx=(0, 6))
        summary_case_check = ttk.Checkbutton(explorer_summary_tools, text="Case", variable=self.explorer_summary_search_case_var)
        summary_case_check.grid(row=0, column=8, sticky="w", padx=(0, 6))
        ttk.Label(explorer_summary_tools, textvariable=self.explorer_summary_search_status_var, foreground="#666666").grid(row=0, column=9, sticky="w", padx=(0, 12))
        ttk.Label(
            explorer_summary_tools,
            text="Filtered Explorer milestones, validation tables, and decisions.",
            foreground="#666666",
        ).grid(row=0, column=10, sticky="w")
        ttk.Label(
            explorer_summary_tools,
            textvariable=self.explorer_summary_context_var,
            foreground="#666666",
        ).grid(row=1, column=0, columnspan=11, sticky="w", pady=(4, 0))
        self._add_tooltips(
            "Context: Search tools for Explorer milestones and validation summaries. Outcome: Highlights matches and lets you navigate decision history quickly. "
            "Example: Find 'Champion updated' across loops.",
            explorer_find_label,
            self.explorer_summary_search_entry,
        )
        self._add_tooltips(
            "Context: Fast decision triage shortcut. Outcome: Sets a search for accepted decisions and jumps to matches. "
            "Example: Review only accepted challenger events.",
            accepted_button,
        )
        self._add_tooltips(
            "Context: Navigate summary search results. Outcome: Moves to previous match in Explorer summary. "
            "Example: Walk backward through 'Decision: ACCEPTED' entries.",
            summary_prev_button,
        )
        self._add_tooltips(
            "Context: Navigate summary search results. Outcome: Moves to next match in Explorer summary. "
            "Example: Step through each loop warning.",
            summary_next_button,
        )
        self._add_tooltips(
            "Context: Reset Explorer summary pane state. Outcome: Clears summary output plus search query/highlights/status. "
            "Example: Start a fresh read after a new Explorer launch.",
            summary_clear_button,
        )
        self._add_tooltips(
            "Context: Summary navigation aid. Outcome: Jumps to the latest summary lines. "
            "Example: Return to newest loop after reading prior runs.",
            summary_bottom_button,
        )
        self._add_tooltips(
            "Context: Live summary monitoring mode. Outcome: Keeps summary viewport at newest output when enabled. "
            "Example: Watch acceptance/rejection updates live.",
            summary_follow_tail,
        )
        self._add_tooltips(
            "Context: Summary search matching mode. Outcome: Enables case-sensitive matching in Explorer summary. "
            "Example: Match exact marker casing only.",
            summary_case_check,
        )
        self._refresh_explorer_summary_context()

        self.explorer_summary_console = scrolledtext.ScrolledText(explorer_summary_tab, wrap="word", height=30)
        self.explorer_summary_console.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        self.explorer_summary_console.tag_configure("summary_search_match", background="#fff2a8")
        self.explorer_summary_console.tag_configure("summary_search_active", background="#f6a23a", foreground="#000000")
        self.explorer_summary_console.configure(state="disabled")

        explorer_live_frame = ttk.Frame(explorer_summary_tab)
        explorer_live_frame.grid(row=1, column=1, sticky="nsew", padx=(0, 8), pady=8)
        explorer_live_frame.grid_columnconfigure(0, weight=1)
        explorer_live_frame.grid_rowconfigure(1, weight=1)

        tally_frame = ttk.LabelFrame(explorer_live_frame, text="Explorer tally")
        tally_frame.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        ttk.Label(
            tally_frame,
            textvariable=self.explorer_tally_var,
            justify="left",
            width=46,
        ).grid(row=0, column=0, sticky="ew", padx=8, pady=8)

        loop_frame = ttk.LabelFrame(explorer_live_frame, text="Loop decisions")
        loop_frame.grid(row=1, column=0, sticky="nsew")
        loop_frame.grid_columnconfigure(0, weight=1)
        loop_frame.grid_rowconfigure(0, weight=1)
        self.explorer_loop_log = scrolledtext.ScrolledText(loop_frame, wrap="word", width=46, height=20)
        self.explorer_loop_log.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.explorer_loop_log.configure(state="disabled")
        self.console_search_var.trace_add("write", lambda *_args: self._on_console_search_changed())
        self.explorer_summary_search_var.trace_add("write", lambda *_args: self._on_explorer_summary_search_changed())
        self.bind_all("<Control-f>", self.focus_console_search)

    def _build_common_tab(self) -> None:
        root = self.tab_common
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(5, weight=1)

        self._path_row(
            root,
            0,
            "Freqtrade project root",
            self.project_root_var,
            directory=True,
            tooltip="Root folder that the launcher uses as cwd and PYTHONPATH base when it starts Freqtrade or opens a strategy file.",
        )
        self._path_row(
            root,
            1,
            "Python executable",
            self.python_exe_var,
            directory=False,
            executable=True,
            tooltip="Python interpreter used to run Freqtrade. Point this at the environment that has Freqtrade installed.",
        )
        self._path_row(
            root,
            2,
            "User data dir",
            self.userdir_var,
            directory=True,
            tooltip="Freqtrade userdata root. This is where config discovery, strategies, data, logs, and result files usually live.",
        )
        self._path_row(
            root,
            3,
            "Data dir",
            self.datadir_var,
            directory=True,
            tooltip="Historical candle and trade data directory passed to Freqtrade as --datadir.",
        )
        self._path_row(
            root,
            4,
            "Log file",
            self.logfile_var,
            directory=False,
            tooltip="Optional logfile passed to Freqtrade. Useful when you want a persistent CLI log in addition to the launcher console.",
        )

        options = ttk.LabelFrame(root, text="Common arguments")
        options.grid(row=5, column=0, columnspan=3, sticky="nsew", padx=8, pady=8)
        for i in range(6):
            options.grid_columnconfigure(i, weight=1)

        log_level_label = ttk.Label(options, text="Log level")
        log_level_label.grid(row=0, column=0, sticky="w", padx=8, pady=8)
        log_level_combo = ttk.Combobox(options, textvariable=self.log_level_var, values=LOG_LEVEL_VALUES, state="readonly", width=12)
        log_level_combo.grid(row=0, column=1, sticky="w", padx=8, pady=8)
        no_color_check = ttk.Checkbutton(options, text="--no-color", variable=self.no_color_var)
        no_color_check.grid(row=0, column=2, sticky="w", padx=8, pady=8)
        self._add_tooltips(
            "Controls how much the launcher asks Freqtrade to log. INFO is the calm default, while DEBUG and TRACE expose progressively more internals.",
            log_level_label,
            log_level_combo,
        )
        self._add_tooltips(
            "Disables ANSI color codes in Freqtrade output. Useful when logs are redirected to a file or another parser.",
            no_color_check,
        )

        self.config_editor = OrderedPathList(options, "Config files (order matters)", [("JSON files", "*.json"), ("All files", "*.*")])
        self.config_editor.grid(row=1, column=0, columnspan=6, sticky="nsew", padx=8, pady=8)
        self.config_editor.set_on_change(self.refresh_all_command_previews)
        ToolTip(
            self.config_editor,
            "Ordered config file stack. Freqtrade loads repeated -c files in order, so later files override earlier ones.",
        )
        options.grid_rowconfigure(1, weight=1)

        strategy = ttk.LabelFrame(root, text="Strategy")
        strategy.grid(row=6, column=0, columnspan=3, sticky="ew", padx=8, pady=(0, 8))
        strategy.grid_columnconfigure(1, weight=1)

        self._path_row(
            strategy,
            0,
            "Strategy file",
            self.strategy_file_var,
            directory=False,
            callback=self.on_browse_strategy,
            tooltip="Python strategy file that the launcher uses to derive --strategy-path and to discover class names.",
        )

        strategy_class_label = ttk.Label(strategy, text="Strategy class")
        strategy_class_label.grid(row=1, column=0, sticky="w", padx=8, pady=8)
        self.strategy_class_combo = ttk.Combobox(strategy, textvariable=self.strategy_class_var)
        self.strategy_class_combo.grid(row=1, column=1, sticky="ew", padx=8, pady=8)
        recursive_check = ttk.Checkbutton(strategy, text="--recursive-strategy-search", variable=self.recursive_strategy_search_var)
        recursive_check.grid(row=2, column=1, sticky="w", padx=8, pady=(0, 8))
        self._add_tooltips(
            "Class name of the strategy to run. This is passed to Freqtrade as --strategy.",
            strategy_class_label,
            self.strategy_class_combo,
        )
        self._add_tooltips(
            "Searches strategy subfolders recursively when looking for the selected strategy class.",
            recursive_check,
        )

        info = ttk.LabelFrame(strategy, text="Notes")
        info.grid(row=3, column=0, columnspan=3, sticky="ew", padx=8, pady=8)
        ttk.Label(
            info,
            text=(
                "The launcher uses the selected file's directory as --strategy-path and the selected class name as --strategy. "
                "The file itself is not edited or copied."
            ),
            wraplength=950,
            justify="left",
        ).grid(row=0, column=0, sticky="w", padx=8, pady=8)

    def _build_pairs_tab(self) -> None:
        root = self.tab_pairs
        root.grid_columnconfigure(0, weight=1)
        root.grid_columnconfigure(1, weight=1)
        root.grid_rowconfigure(2, weight=1)

        mode_frame = ttk.LabelFrame(root, text="Pair selection mode")
        mode_frame.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=8)
        config_mode = ttk.Radiobutton(mode_frame, text="Use config pairlist", variable=self.pair_mode_var, value="config")
        config_mode.grid(row=0, column=0, sticky="w", padx=8, pady=8)
        manual_mode = ttk.Radiobutton(mode_frame, text="Manual pairs override", variable=self.pair_mode_var, value="manual")
        manual_mode.grid(row=0, column=1, sticky="w", padx=8, pady=8)
        ttk.Label(
            mode_frame,
            text=(
                "Backtest / Hyperopt use -p/--pairs. Live / Dry-run generate a temporary overlay config with "
                "exchange.pair_whitelist and optionally pair_blacklist."
            ),
            wraplength=980,
            justify="left",
        ).grid(row=1, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 8))
        self._add_tooltips(
            "Uses the pairlist defined in your config files. This is the closest match to the bot's normal live behavior.",
            config_mode,
        )
        self._add_tooltips(
            "Lets the launcher override the config pairlist with the symbols you type or pick in this tab.",
            manual_mode,
        )

        whitelist_frame = ttk.LabelFrame(root, text="Whitelist pairs")
        whitelist_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        whitelist_frame.grid_rowconfigure(1, weight=1)
        whitelist_frame.grid_columnconfigure(0, weight=1)
        whitelist_buttons = ttk.Frame(whitelist_frame)
        whitelist_buttons.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 0))
        add_typed = ttk.Button(whitelist_buttons, text="Add typed", command=lambda: self.add_typed_pair(self.pairs_text))
        add_typed.pack(side="left", padx=(0, 8))
        remove_selected = ttk.Button(whitelist_buttons, text="Remove selected", command=lambda: self.remove_selected_pairs(self.pairs_text))
        remove_selected.pack(side="left", padx=(0, 8))
        normalize = ttk.Button(whitelist_buttons, text="Normalize", command=lambda: self.normalize_pair_text(self.pairs_text))
        normalize.pack(side="left")
        self.pairs_text = scrolledtext.ScrolledText(whitelist_frame, wrap="word", height=12)
        self.pairs_text.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        self.pairs_text.bind("<KeyRelease>", lambda _e: (self.refresh_all_command_previews(), self.refresh_orderbook_pair_preview()))
        whitelist_tip = (
            "Pairs used for manual mode, backtesting, hyperopt, and download-data when you choose to use the manual list. "
            "Freqtrade accepts space-separated pairs on the CLI, and the launcher normalizes commas and newlines too."
        )
        ToolTip(self.pairs_text, whitelist_tip)
        ToolTip(
            add_typed,
            "Context: Quick pair entry helper for whitelist maintenance. Outcome: Appends typed pairs into the whitelist box. "
            "Example: Add `BTC/USDT ETH/USDT` in one prompt.",
        )
        ToolTip(
            remove_selected,
            "Context: Whitelist cleanup action. Outcome: Removes selected/current whitelist line(s) from the pair list. "
            "Example: Remove a delisted symbol before backtesting.",
        )
        ToolTip(
            normalize,
            "Context: Pair list hygiene for stable command generation. Outcome: Trims spacing and deduplicates whitelist entries. "
            "Example: `btc/usdt, BTC/USDT` becomes one normalized pair.",
        )
        ttk.Label(whitelist_frame, text="Comma, space, or newline separated.").grid(row=2, column=0, sticky="w", padx=8, pady=(0, 8))

        blacklist_frame = ttk.LabelFrame(root, text="Blacklist pairs")
        blacklist_frame.grid(row=1, column=1, sticky="nsew", padx=8, pady=8)
        blacklist_frame.grid_rowconfigure(1, weight=1)
        blacklist_frame.grid_columnconfigure(0, weight=1)
        blacklist_buttons = ttk.Frame(blacklist_frame)
        blacklist_buttons.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 0))
        add_typed_black = ttk.Button(blacklist_buttons, text="Add typed", command=lambda: self.add_typed_pair(self.blacklist_text))
        add_typed_black.pack(side="left", padx=(0, 8))
        remove_selected_black = ttk.Button(blacklist_buttons, text="Remove selected", command=lambda: self.remove_selected_pairs(self.blacklist_text))
        remove_selected_black.pack(side="left", padx=(0, 8))
        normalize_black = ttk.Button(blacklist_buttons, text="Normalize", command=lambda: self.normalize_pair_text(self.blacklist_text))
        normalize_black.pack(side="left")
        self.blacklist_text = scrolledtext.ScrolledText(blacklist_frame, wrap="word", height=12)
        self.blacklist_text.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        self.blacklist_text.bind("<KeyRelease>", lambda _e: self.refresh_all_command_previews())
        blacklist_tip = (
            "Pairs to exclude when manual pair mode is enabled. The launcher only turns this into a temporary overlay for live and dry-run commands."
        )
        ToolTip(self.blacklist_text, blacklist_tip)
        ToolTip(
            add_typed_black,
            "Context: Quick pair entry helper for blacklist maintenance. Outcome: Appends typed pairs into the blacklist box. "
            "Example: Add `LUNA/USDT` to exclude it in manual mode.",
        )
        ToolTip(
            remove_selected_black,
            "Context: Blacklist cleanup action. Outcome: Removes selected/current blacklist line(s). "
            "Example: Remove a pair after risk conditions improve.",
        )
        ToolTip(
            normalize_black,
            "Context: Pair list hygiene for blacklist consistency. Outcome: Trims spacing and deduplicates blacklist entries. "
            "Example: Merge repeated entries to one symbol.",
        )
        ttk.Label(blacklist_frame, text="Optional. Applied only when manual pair mode is active.").grid(row=2, column=0, sticky="w", padx=8, pady=(0, 8))

        reference_frame = ttk.LabelFrame(root, text="Reference pair lists")
        reference_frame.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=8, pady=8)
        reference_frame.grid_columnconfigure(0, weight=1)
        reference_frame.grid_columnconfigure(1, weight=2)
        reference_frame.grid_rowconfigure(1, weight=1)

        reference_controls = ttk.Frame(reference_frame)
        reference_controls.grid(row=0, column=0, columnspan=2, sticky="ew", padx=8, pady=(8, 0))
        pair_format_label = ttk.Label(reference_controls, text="Pair format")
        pair_format_label.pack(side="left")
        format_combo = ttk.Combobox(
            reference_controls,
            textvariable=self.pair_reference_format_var,
            values=["USDT futures", "USDT spot"],
            state="readonly",
            width=14,
        )
        format_combo.pack(side="left", padx=(8, 16))
        format_combo.bind("<<ComboboxSelected>>", lambda _event: self.populate_pair_reference_tree())
        add_to_white = ttk.Button(reference_controls, text="Add to whitelist", command=lambda: self.add_reference_group_to_widget(self.pairs_text))
        add_to_white.pack(side="left", padx=(0, 8))
        add_to_black = ttk.Button(reference_controls, text="Add to blacklist", command=lambda: self.add_reference_group_to_widget(self.blacklist_text))
        add_to_black.pack(side="left", padx=(0, 8))
        add_to_download = ttk.Button(reference_controls, text="Add to download", command=self.add_reference_group_to_download)
        add_to_download.pack(side="left", padx=(0, 8))
        replace_white = ttk.Button(reference_controls, text="Replace whitelist", command=lambda: self.replace_widget_with_reference_group(self.pairs_text))
        replace_white.pack(side="left")
        self._add_tooltips(
            "Selects whether the reference pairs are formatted as spot or futures symbols before being copied into a pair list.",
            pair_format_label,
            format_combo,
        )
        self._add_tooltips(
            "Context: Apply a curated reference list to trading whitelist quickly. Outcome: Appends selected reference group to whitelist. "
            "Example: Add 'Top 10 Market Cap' into whitelist.",
            add_to_white,
        )
        self._add_tooltips(
            "Context: Apply curated exclusions. Outcome: Appends selected reference group to blacklist. "
            "Example: Add a high-volatility group to blacklist temporarily.",
            add_to_black,
        )
        self._add_tooltips(
            "Context: Seed Download Data pairs from curated groups. Outcome: Appends selected reference group into Download pairs. "
            "Example: Load a spot-ready reference set before data fetch.",
            add_to_download,
        )
        self._add_tooltips(
            "Context: Reset whitelist to one curated baseline. Outcome: Replaces current whitelist with selected reference group only. "
            "Example: Replace ad-hoc list with 'Top 100 Market Cap'.",
            replace_white,
        )

        columns = ("count", "note")
        self.pair_reference_tree = ttk.Treeview(reference_frame, columns=columns, show="tree headings", height=10, selectmode="browse")
        self.pair_reference_tree.heading("#0", text="List")
        self.pair_reference_tree.heading("count", text="Pairs")
        self.pair_reference_tree.heading("note", text="Use")
        self.pair_reference_tree.column("#0", width=210, anchor="w")
        self.pair_reference_tree.column("count", width=60, anchor="center", stretch=False)
        self.pair_reference_tree.column("note", width=420, anchor="w")
        self.pair_reference_tree.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=8)
        reference_scroll = ttk.Scrollbar(reference_frame, orient="vertical", command=self.pair_reference_tree.yview)
        reference_scroll.grid(row=1, column=0, sticky="nse", padx=(0, 4), pady=8)
        self.pair_reference_tree.configure(yscrollcommand=reference_scroll.set)
        self.pair_reference_tree.bind("<<TreeviewSelect>>", lambda _event: self.update_pair_reference_details())
        self.pair_reference_tree.bind("<Double-Button-1>", lambda _event: self.add_reference_group_to_widget(self.pairs_text))
        ToolTip(
            self.pair_reference_tree,
            "Context: Curated pair-list library for quick pair workflows. Outcome: Selecting a row enables copy actions; double-click adds to whitelist. "
            "Example: Double-click a group to stage it for manual backtesting.",
        )

        self.pair_reference_details = scrolledtext.ScrolledText(reference_frame, wrap="word", height=10)
        self.pair_reference_details.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=8)
        self.pair_reference_details.configure(state="disabled")
        ToolTip(
            self.pair_reference_details,
            "Context: Documentation panel for the selected reference list. Outcome: Shows usage notes and intent without editing. "
            "Example: Verify whether a list targets futures or spot before copying.",
        )
        ttk.Label(
            reference_frame,
            text="These are curated launcher references. Freqtrade can also build dynamic lists from MarketCapPairList, VolumePairList, and filters like VolatilityFilter in config.",
            wraplength=980,
            justify="left",
        ).grid(row=2, column=0, columnspan=2, sticky="w", padx=8, pady=(0, 8))
        self.populate_pair_reference_tree()

    def selected_pair_reference_group(self) -> dict[str, Any] | None:
        tree = self.pair_reference_tree
        if tree is None:
            return None
        selected = tree.selection()
        if not selected:
            return None
        name = tree.item(selected[0], "text")
        for group in PAIR_REFERENCE_GROUPS:
            if group["name"] == name:
                return group
        return None

    def format_reference_pairs(self, symbols: Iterable[str]) -> list[str]:
        suffix = "" if self.pair_reference_format_var.get() == "USDT spot" else ":USDT"
        pairs: list[str] = []
        seen: set[str] = set()
        for symbol in symbols:
            symbol = str(symbol).strip().upper()
            if not symbol:
                continue
            pair = symbol if "/" in symbol else f"{symbol}/USDT{suffix}"
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
        return pairs

    def populate_pair_reference_tree(self) -> None:
        tree = self.pair_reference_tree
        if tree is None:
            return
        tree.delete(*tree.get_children())
        for group in PAIR_REFERENCE_GROUPS:
            pairs = self.format_reference_pairs(group.get("pairs", []))
            tree.insert("", tk.END, text=group["name"], values=(len(pairs), group.get("note", "")))
        first = tree.get_children()
        if first:
            tree.selection_set(first[0])
            tree.focus(first[0])
        self.update_pair_reference_details()

    def update_pair_reference_details(self) -> None:
        details = self.pair_reference_details
        if details is None:
            return
        group = self.selected_pair_reference_group()
        details.configure(state="normal")
        details.delete("1.0", tk.END)
        if group is None:
            details.insert(tk.END, "Select a reference list to preview its pairs.")
        else:
            pairs = self.format_reference_pairs(group.get("pairs", []))
            self.pair_reference_selected_var.set(group["name"])
            details.insert(tk.END, f"{group['name']}\n\n")
            details.insert(tk.END, f"{group.get('note', '')}\n\n")
            details.insert(tk.END, "\n".join(pairs))
        details.configure(state="disabled")

    def add_reference_group_to_widget(self, widget: scrolledtext.ScrolledText | None) -> None:
        if widget is None:
            return
        group = self.selected_pair_reference_group()
        if group is None:
            return
        self.append_pairs_to_widget(widget, self.format_reference_pairs(group.get("pairs", [])))

    def replace_widget_with_reference_group(self, widget: scrolledtext.ScrolledText | None) -> None:
        if widget is None:
            return
        group = self.selected_pair_reference_group()
        if group is None:
            return
        pairs = self.format_reference_pairs(group.get("pairs", []))
        widget.delete("1.0", tk.END)
        widget.insert("1.0", "\n".join(pairs))
        if pairs:
            widget.insert(tk.END, "\n")
        self.pair_mode_var.set("manual")
        self.refresh_all_command_previews()

    def add_reference_group_to_download(self) -> None:
        if self.download_pairs_text is None:
            return
        group = self.selected_pair_reference_group()
        if group is None:
            return
        self.append_pairs_to_widget(self.download_pairs_text, self.format_reference_pairs(group.get("pairs", [])), set_manual_mode=False)

    def add_typed_pair(self, widget: scrolledtext.ScrolledText | None, *, set_manual_mode: bool = True) -> None:
        if widget is None:
            return
        value = simpledialog.askstring(APP_TITLE, "Pair(s) to add, comma/space/newline separated:")
        if value:
            self.append_pairs_to_widget(widget, parse_token_list(value), set_manual_mode=set_manual_mode)

    def append_pairs_to_widget(self, widget: scrolledtext.ScrolledText | None, pairs: Iterable[str], *, set_manual_mode: bool = True) -> None:
        if widget is None:
            return
        existing = parse_token_list(widget.get("1.0", tk.END))
        seen = set(existing)
        updated = list(existing)
        for pair in pairs:
            pair = str(pair).strip()
            if pair and pair not in seen:
                updated.append(pair)
                seen.add(pair)
        widget.delete("1.0", tk.END)
        widget.insert("1.0", "\n".join(updated))
        if updated:
            widget.insert(tk.END, "\n")
        if set_manual_mode:
            self.pair_mode_var.set("manual")
        self.refresh_all_command_previews()

    def normalize_pair_text(self, widget: scrolledtext.ScrolledText | None, *, set_manual_mode: bool = True) -> None:
        if widget is None:
            return
        self.append_pairs_to_widget(widget, [], set_manual_mode=set_manual_mode)

    def remove_selected_pairs(self, widget: scrolledtext.ScrolledText | None) -> None:
        if widget is None:
            return
        try:
            selected = widget.get(tk.SEL_FIRST, tk.SEL_LAST)
        except tk.TclError:
            line_start = widget.index("insert linestart")
            line_end = widget.index("insert lineend")
            selected = widget.get(line_start, line_end)
        selected_pairs = set(parse_token_list(selected))
        if not selected_pairs:
            return
        remaining = [pair for pair in parse_token_list(widget.get("1.0", tk.END)) if pair not in selected_pairs]
        widget.delete("1.0", tk.END)
        widget.insert("1.0", "\n".join(remaining))
        if remaining:
            widget.insert(tk.END, "\n")
        self.refresh_all_command_previews()

    def copy_whitelist_to_download_pairs(self) -> None:
        if self.download_pairs_text is None:
            return
        pairs = parse_token_list(self.pairs_text.get("1.0", tk.END))
        self.download_pairs_text.delete("1.0", tk.END)
        self.download_pairs_text.insert("1.0", "\n".join(pairs))
        if pairs:
            self.download_pairs_text.insert(tk.END, "\n")
        self.refresh_all_command_previews()

    def _build_mode_tab(self) -> None:
        root = self.tab_mode
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(0, weight=1)
        root.grid_rowconfigure(1, weight=0)

        self.mode_stack = ttk.Frame(root)
        self.mode_stack.grid(row=0, column=0, sticky="nsew")
        self.mode_stack.grid_columnconfigure(0, weight=1)

        frequi_tools = ttk.LabelFrame(root, text="FreqUI utility")
        frequi_tools.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        frequi_tools.grid_columnconfigure(1, weight=1)
        ttk.Label(frequi_tools, text="FreqUI URL").grid(row=0, column=0, sticky="w", padx=8, pady=8)
        ttk.Entry(frequi_tools, textvariable=self.frequi_url_var).grid(row=0, column=1, sticky="ew", padx=8, pady=8)
        self.frequi_launch_button = ttk.Button(frequi_tools, text="Launch FreqUI", command=self.on_launch_frequi)
        self.frequi_launch_button.grid(row=0, column=2, sticky="ew", padx=8, pady=8)

        self.trade_frame = ttk.LabelFrame(self.mode_stack, text="Trade / Dry-run options")
        self.backtest_frame = ttk.LabelFrame(self.mode_stack, text="Backtest options")
        self.lookahead_frame = ttk.LabelFrame(self.mode_stack, text="Lookahead Analysis options")
        self.hyperopt_frame = ttk.LabelFrame(self.mode_stack, text="Hyperopt options")
        self.other_mode_frame = ttk.LabelFrame(self.mode_stack, text="Mode note")

        for frame in (self.trade_frame, self.backtest_frame, self.lookahead_frame, self.hyperopt_frame, self.other_mode_frame):
            frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
            frame.grid_columnconfigure(1, weight=1)
            frame.grid_columnconfigure(3, weight=1)

        trade_db_tip = (
            "Trade database URL used for live or dry-run bot storage. Freqtrade defaults to "
            "sqlite:///tradesv3.sqlite for live and sqlite:///tradesv3.dryrun.sqlite for dry-run; "
            "changing this points the launcher at another SQLite or database-backed trade ledger. "
            "Leave blank to omit --db-url and keep Freqtrade's normal default/config behavior."
        )
        fee_tip = (
            "Fee ratio used by simulations. Freqtrade applies it on both entry and exit, so a 0.1% "
            "fee is counted twice in backtesting and dry-run estimates. Leave blank to omit --fee "
            "and keep strategy/config defaults."
        )
        dry_run_wallet_tip = (
            "Simulated starting balance for backtesting, lookahead-analysis, hyperopt, and dry-run. It must be larger "
            "than stake amount or trades can be blocked before they open. Leave blank to omit "
            "--dry-run-wallet and keep strategy/config defaults."
        )
        timeframe_tip = (
            "Primary candle size for backtesting, lookahead-analysis, or hyperopt. The timeframe must exist in your "
            "downloaded data, and the strategy or config can also supply it. Leave blank to omit -i "
            "so strategy/config defaults apply."
        )
        timerange_tip = (
            "Slices the historical data used by the run. You can use candles, dates, or timestamps, "
            "for example a range like 20240101-20240331 or a candle count like -200. Leave blank to "
            "omit --timerange and use the full available data range."
        )
        timeframe_detail_tip = (
            "Optional lower timeframe used to simulate intra-candle movement during backtesting. "
            "It improves accuracy, but it loads more data, uses more RAM, and must be smaller than "
            "the main timeframe. Leave blank to omit --timeframe-detail."
        )
        stake_amount_tip = (
            "Amount allocated per simulated trade. In backtesting, lookahead-analysis, and hyperopt it works together with "
            "dry-run wallet and max open trades. Leave blank to omit --stake-amount and keep "
            "strategy/config defaults."
        )
        max_open_trades_tip = (
            "Cap on concurrent open trades. Raising it can reveal trades that would otherwise be "
            "masked by an occupied slot; an extremely high value effectively removes the cap. Leave "
            "blank to omit --max-open-trades and keep strategy/config defaults."
        )
        enable_protections_tip = (
            "Runs configured protections during backtesting, lookahead-analysis, or hyperopt, including tools such as "
            "StoplossGuard, MaxDrawdown, and CooldownPeriod. It is useful for realism, but it slows "
            "simulation down."
        )
        position_stacking_tip = (
            "Allows the same pair to be entered more than once during backtesting, lookahead-analysis, or hyperopt. "
            "Dry-run and live trading do not use this behavior, so it is best treated as a "
            "simulation-only check."
        )
        export_tip = (
            "Controls what backtesting writes to disk. Default means the launcher sends no explicit "
            "export flag, so Freqtrade uses its own default backtesting export behavior, which is "
            "currently trades. none skips export; trades writes the standard backtest result file "
            "under user_data/backtest_results; signals also exports the signal-candle data needed "
            "for deeper backtest analysis and often produces much larger files."
        )
        breakdown_tip = (
            "Adds breakdown tables to backtesting output. None leaves the summary uncluttered; the "
            "other choices group performance by day, week, month, year, or weekday."
        )
        spaces_tip = (
            "Selects which hyperopt search spaces are optimized. Default runs the usual set, all "
            "includes every available space, and the other entries let you focus on buy, sell, roi, "
            "stoploss, trailing, protection, or trades. The bundled combined entries are shortcuts "
            "for common mixes."
        )
        loss_tip = (
            "Chooses the loss function that scores each epoch. Different loss classes reward different "
            "trade-offs, such as pure profit, Sharpe/Sortino ratio, drawdown control, or short trade "
            "duration."
        )
        jobs_tip = (
            "Number of CPU cores used for hyperopt workers. Enter direct core counts only: 1 uses one core, "
            "20 uses twenty cores. Leave blank to omit -j and let Freqtrade choose."
        )
        random_state_tip = (
            "Sets the random seed for reproducible hyperopt runs. Using the same seed with the same "
            "data and settings makes results easier to compare. Leave blank to omit --random-state."
        )
        min_trades_tip = (
            "Minimum desired trade count for hyperopt evaluations. This helps avoid rewarding "
            "parameter sets that look good only because they generated too few trades. Leave blank "
            "to omit --min-trades."
        )
        print_all_tip = (
            "Prints every epoch instead of only the best results. This is useful when you want to "
            "inspect the full search, not just the top-scoring rows."
        )
        disable_param_export_tip = (
            "Stops hyperopt from automatically exporting optimized parameter files at the end of the "
            "run. That keeps temporary searches from overwriting strategy parameter files."
        )
        ignore_missing_spaces_tip = (
            "Suppresses errors when a requested space has no parameters in the strategy. That is "
            "useful when you want one preset to work across strategies with different parameter "
            "coverage."
        )
        analyze_per_epoch_tip = (
            "Recalculates populate_indicators once per epoch instead of once per pair up front. It "
            "uses more CPU, but can cut RAM pressure and help when .range-heavy parameters would "
            "otherwise blow up memory."
        )
        early_stop_tip = (
            "Stops hyperopt after the given number of epochs without improvement. Freqtrade suggests "
            "using a modest value so long runs can end once they stop learning. Leave blank to "
            "disable early-stop and run all epochs."
        )
        monitor_tip = (
            "Tails the latest .fthypt file while hyperopt runs so you can watch progress live. If "
            "Result file is blank, the launcher auto-detects the newest file in userdir/hyperopt_results."
        )
        result_file_tip = (
            "Optional explicit .fthypt or .pickle file to tail or inspect. Leave it blank and the "
            "launcher watches the newest file in userdir/hyperopt_results instead."
        )
        poll_ms_tip = (
            "How often, in milliseconds, the launcher checks the results file while tailing it. Lower "
            "values update faster; higher values reduce file polling."
        )
        minimum_trade_amount_tip = (
            "Minimum number of trades required before look-ahead analysis proceeds. This helps avoid "
            "false confidence from tiny sample sizes. Leave blank to omit --minimum-trade-amount."
        )
        targeted_trade_amount_tip = (
            "Target number of trades for look-ahead analysis. Larger targets usually increase runtime "
            "but improve confidence in bias checks. Leave blank to omit --targeted-trade-amount."
        )
        lookahead_export_filename_tip = (
            "Optional output filename for look-ahead analysis results. Leave blank to let Freqtrade use "
            "its default naming."
        )
        allow_limit_orders_tip = (
            "Allows limit orders during look-ahead checks. This can increase false positives in some "
            "strategies, so keep it off unless you need to mirror limit-order behavior."
        )

        # Trade / Dry-run
        db_label, db_entry = self._labeled_entry(self.trade_frame, 0, 0, "DB URL", self.db_url_var)
        fee_label, fee_entry = self._labeled_entry(self.trade_frame, 0, 2, "Fee", self.fee_var)
        wallet_label, wallet_entry = self._labeled_entry(self.trade_frame, 1, 0, "Dry-run wallet", self.dry_run_wallet_var)
        self._add_tooltips(trade_db_tip, db_label, db_entry)
        self._add_tooltips(fee_tip, fee_label, fee_entry)
        self._add_tooltips(dry_run_wallet_tip, wallet_label, wallet_entry)
        ttk.Label(
            self.trade_frame,
            text="Live uses the selected config(s) as-is. Dry-run adds --dry-run. Manual pairs for either mode are applied through a temporary overlay config.",
            wraplength=920,
            justify="left",
        ).grid(row=2, column=0, columnspan=4, sticky="w", padx=8, pady=8)

        # Shared backtest / hyperopt section helper rows
        timeframe_label, timeframe_entry = self._labeled_entry(self.backtest_frame, 0, 0, "Timeframe", self.timeframe_var)
        timerange_label, timerange_entry = self._labeled_entry(self.backtest_frame, 0, 2, "Timerange", self.timerange_var)
        timeframe_detail_label, timeframe_detail_entry = self._labeled_entry(self.backtest_frame, 1, 0, "Timeframe detail", self.timeframe_detail_var)
        fee2_label, fee2_entry = self._labeled_entry(self.backtest_frame, 1, 2, "Fee", self.fee_var)
        wallet2_label, wallet2_entry = self._labeled_entry(self.backtest_frame, 2, 0, "Dry-run wallet", self.dry_run_wallet_var)
        stake_label, stake_entry = self._labeled_entry(self.backtest_frame, 2, 2, "Stake amount", self.stake_amount_var)
        max_open_label, max_open_entry = self._labeled_entry(self.backtest_frame, 3, 0, "Max open trades", self.max_open_trades_var)
        self._add_tooltips(timeframe_tip, timeframe_label, timeframe_entry)
        self._add_tooltips(timerange_tip, timerange_label, timerange_entry)
        self._add_tooltips(timeframe_detail_tip, timeframe_detail_label, timeframe_detail_entry)
        self._add_tooltips(fee_tip, fee2_label, fee2_entry)
        self._add_tooltips(dry_run_wallet_tip, wallet2_label, wallet2_entry)
        self._add_tooltips(stake_amount_tip, stake_label, stake_entry)
        self._add_tooltips(max_open_trades_tip, max_open_label, max_open_entry)
        backtest_protections = ttk.Checkbutton(self.backtest_frame, text="--enable-protections", variable=self.enable_protections_var)
        backtest_protections.grid(row=4, column=0, sticky="w", padx=8, pady=8)
        backtest_position_stacking = ttk.Checkbutton(self.backtest_frame, text="--eps / --enable-position-stacking", variable=self.enable_position_stacking_var)
        backtest_position_stacking.grid(row=4, column=1, sticky="w", padx=8, pady=8)
        export_label = ttk.Label(self.backtest_frame, text="Export")
        export_label.grid(row=5, column=0, sticky="w", padx=8, pady=8)
        export_combo = ttk.Combobox(self.backtest_frame, textvariable=self.backtest_export_var, values=EXPORT_VALUES, state="readonly", width=18)
        export_combo.grid(row=5, column=1, sticky="w", padx=8, pady=8)
        breakdown_label = ttk.Label(self.backtest_frame, text="Breakdown")
        breakdown_label.grid(row=5, column=2, sticky="w", padx=8, pady=8)
        breakdown_combo = ttk.Combobox(self.backtest_frame, textvariable=self.backtest_breakdown_var, values=BREAKDOWN_VALUES, state="readonly", width=18)
        breakdown_combo.grid(row=5, column=3, sticky="w", padx=8, pady=8)
        self._add_tooltips(enable_protections_tip, backtest_protections)
        self._add_tooltips(position_stacking_tip, backtest_position_stacking)
        self._add_tooltips(export_tip, export_label, export_combo)
        self._add_tooltips(breakdown_tip, breakdown_label, breakdown_combo)

        look_timeframe_label, look_timeframe_entry = self._labeled_entry(self.lookahead_frame, 0, 0, "Timeframe", self.timeframe_var)
        look_timerange_label, look_timerange_entry = self._labeled_entry(self.lookahead_frame, 0, 2, "Timerange", self.timerange_var)
        look_timeframe_detail_label, look_timeframe_detail_entry = self._labeled_entry(self.lookahead_frame, 1, 0, "Timeframe detail", self.timeframe_detail_var)
        look_fee_label, look_fee_entry = self._labeled_entry(self.lookahead_frame, 1, 2, "Fee", self.fee_var)
        look_wallet_label, look_wallet_entry = self._labeled_entry(self.lookahead_frame, 2, 0, "Dry-run wallet", self.dry_run_wallet_var)
        look_stake_label, look_stake_entry = self._labeled_entry(self.lookahead_frame, 2, 2, "Stake amount", self.stake_amount_var)
        look_max_open_label, look_max_open_entry = self._labeled_entry(self.lookahead_frame, 3, 0, "Max open trades", self.max_open_trades_var)
        self._add_tooltips(timeframe_tip, look_timeframe_label, look_timeframe_entry)
        self._add_tooltips(timerange_tip, look_timerange_label, look_timerange_entry)
        self._add_tooltips(timeframe_detail_tip, look_timeframe_detail_label, look_timeframe_detail_entry)
        self._add_tooltips(fee_tip, look_fee_label, look_fee_entry)
        self._add_tooltips(dry_run_wallet_tip, look_wallet_label, look_wallet_entry)
        self._add_tooltips(stake_amount_tip, look_stake_label, look_stake_entry)
        self._add_tooltips(max_open_trades_tip, look_max_open_label, look_max_open_entry)
        lookahead_protections = ttk.Checkbutton(self.lookahead_frame, text="--enable-protections", variable=self.enable_protections_var)
        lookahead_protections.grid(row=4, column=0, sticky="w", padx=8, pady=8)
        lookahead_position_stacking = ttk.Checkbutton(self.lookahead_frame, text="--eps / --enable-position-stacking", variable=self.enable_position_stacking_var)
        lookahead_position_stacking.grid(row=4, column=1, sticky="w", padx=8, pady=8)
        lookahead_recursive_search = ttk.Checkbutton(self.lookahead_frame, text="--recursive-strategy-search", variable=self.recursive_strategy_search_var)
        lookahead_recursive_search.grid(row=4, column=2, sticky="w", padx=8, pady=8)
        self._add_tooltips(enable_protections_tip, lookahead_protections)
        self._add_tooltips(position_stacking_tip, lookahead_position_stacking)
        self._add_tooltips(
            "Searches strategy subfolders recursively when looking for the selected strategy class.",
            lookahead_recursive_search,
        )
        look_min_trade_label, look_min_trade_entry = self._labeled_entry(
            self.lookahead_frame,
            5,
            0,
            "Min trade amount",
            self.lookahead_min_trade_amount_var,
        )
        look_targeted_label, look_targeted_entry = self._labeled_entry(
            self.lookahead_frame,
            5,
            2,
            "Targeted trade amount",
            self.lookahead_targeted_trade_amount_var,
        )
        look_export_label, look_export_entry = self._labeled_entry(
            self.lookahead_frame,
            6,
            0,
            "Export filename",
            self.lookahead_export_filename_var,
        )
        look_allow_limit_orders = ttk.Checkbutton(
            self.lookahead_frame,
            text="--allow-limit-orders",
            variable=self.lookahead_allow_limit_orders_var,
        )
        look_allow_limit_orders.grid(row=6, column=2, sticky="w", padx=8, pady=8)
        self._add_tooltips(minimum_trade_amount_tip, look_min_trade_label, look_min_trade_entry)
        self._add_tooltips(targeted_trade_amount_tip, look_targeted_label, look_targeted_entry)
        self._add_tooltips(lookahead_export_filename_tip, look_export_label, look_export_entry)
        self._add_tooltips(allow_limit_orders_tip, look_allow_limit_orders)

        hyper_timeframe_label, hyper_timeframe_entry = self._labeled_entry(self.hyperopt_frame, 0, 0, "Timeframe", self.timeframe_var)
        hyper_timerange_label, hyper_timerange_entry = self._labeled_entry(self.hyperopt_frame, 0, 2, "Timerange", self.timerange_var)
        hyper_timeframe_detail_label, hyper_timeframe_detail_entry = self._labeled_entry(self.hyperopt_frame, 1, 0, "Timeframe detail", self.timeframe_detail_var)
        hyper_fee_label, hyper_fee_entry = self._labeled_entry(self.hyperopt_frame, 1, 2, "Fee", self.fee_var)
        hyper_wallet_label, hyper_wallet_entry = self._labeled_entry(self.hyperopt_frame, 2, 0, "Dry-run wallet", self.dry_run_wallet_var)
        hyper_stake_label, hyper_stake_entry = self._labeled_entry(self.hyperopt_frame, 2, 2, "Stake amount", self.stake_amount_var)
        hyper_max_open_label, hyper_max_open_entry = self._labeled_entry(self.hyperopt_frame, 3, 0, "Max open trades", self.max_open_trades_var)
        hyper_epochs_label, hyper_epochs_entry = self._labeled_entry(self.hyperopt_frame, 3, 2, "Epoch multiplier", self.hyperopt_epochs_var)
        self._add_tooltips(timeframe_tip, hyper_timeframe_label, hyper_timeframe_entry)
        self._add_tooltips(timerange_tip, hyper_timerange_label, hyper_timerange_entry)
        self._add_tooltips(timeframe_detail_tip, hyper_timeframe_detail_label, hyper_timeframe_detail_entry)
        self._add_tooltips(fee_tip, hyper_fee_label, hyper_fee_entry)
        self._add_tooltips(dry_run_wallet_tip, hyper_wallet_label, hyper_wallet_entry)
        self._add_tooltips(stake_amount_tip, hyper_stake_label, hyper_stake_entry)
        self._add_tooltips(max_open_trades_tip, hyper_max_open_label, hyper_max_open_entry)
        self._add_tooltips(
            "Multiplier used to compute total hyperopt epochs as: job workers * epoch multiplier. "
            f"Example: 10 workers and multiplier 10 runs 100 epochs. Total epochs are capped at {MAX_HYPEROPT_EPOCHS}. "
            "Leave blank to omit -e and let Freqtrade use its default.",
            hyper_epochs_label,
            hyper_epochs_entry,
        )

        spaces_label = ttk.Label(self.hyperopt_frame, text="Spaces")
        spaces_label.grid(row=4, column=0, sticky="w", padx=8, pady=8)
        spaces_combo = ttk.Combobox(
            self.hyperopt_frame,
            textvariable=self.hyperopt_spaces_var,
            values=HYPEROPT_SPACE_VALUES,
            width=28,
        )
        spaces_combo.grid(row=4, column=1, sticky="ew", padx=8, pady=8)
        ttk.Label(
            self.hyperopt_frame,
            text=SPACES_HINT,
            foreground="#666666",
            wraplength=920,
            justify="left",
        ).grid(row=5, column=0, columnspan=4, sticky="w", padx=8, pady=(0, 8))
        self._add_tooltips(spaces_tip, spaces_label, spaces_combo)
        jobs_label, jobs_entry = self._labeled_entry(self.hyperopt_frame, 6, 0, "Job workers", self.hyperopt_jobs_var)
        random_state_label, random_state_entry = self._labeled_entry(self.hyperopt_frame, 6, 2, "Random state", self.hyperopt_random_state_var)
        min_trades_label, min_trades_entry = self._labeled_entry(self.hyperopt_frame, 7, 0, "Min trades", self.hyperopt_min_trades_var)
        self._add_tooltips(jobs_tip, jobs_label, jobs_entry)
        self._add_tooltips(random_state_tip, random_state_label, random_state_entry)
        self._add_tooltips(min_trades_tip, min_trades_label, min_trades_entry)
        hyperopt_loss_label = ttk.Label(self.hyperopt_frame, text="Hyperopt loss")
        hyperopt_loss_label.grid(row=7, column=2, sticky="w", padx=8, pady=8)
        loss_picker = ttk.Frame(self.hyperopt_frame)
        loss_picker.grid(row=7, column=3, sticky="ew", padx=8, pady=8)
        loss_picker.grid_columnconfigure(0, weight=1)
        self.hyperopt_loss_combo = ttk.Combobox(
            loss_picker,
            textvariable=self.hyperopt_loss_var,
            values=HYPEROPT_LOSS_VALUES,
            width=28,
        )
        self.hyperopt_loss_combo.grid(row=0, column=0, sticky="ew")
        self.hyperopt_loss_combo.bind("<<ComboboxSelected>>", lambda _event: self.show_hyperopt_loss_popup())
        self.hyperopt_loss_combo.bind("<Double-Button-1>", lambda _event: self.show_hyperopt_loss_popup())
        ttk.Button(loss_picker, text="Details", command=self.show_hyperopt_loss_popup).grid(row=0, column=1, sticky="ew", padx=(6, 0))
        ttk.Button(loss_picker, text="Refresh", command=lambda: self.refresh_hyperopt_loss_dropdown(force=True)).grid(row=0, column=2, sticky="ew", padx=(6, 0))
        self._add_tooltips(loss_tip, hyperopt_loss_label, loss_picker, self.hyperopt_loss_combo)
        ttk.Label(
            self.hyperopt_frame,
            text="Custom losses are discovered from freqtrade/optimize/hyperopt_loss and user_data/hyperopts.",
            foreground="#666666",
            wraplength=460,
            justify="left",
        ).grid(row=8, column=2, columnspan=2, sticky="w", padx=8, pady=(0, 8))
        early_stop_label, early_stop_entry = self._labeled_entry(self.hyperopt_frame, 8, 0, "Early stop", self.hyperopt_early_stop_var)
        hyperopt_protections = ttk.Checkbutton(self.hyperopt_frame, text="--enable-protections", variable=self.enable_protections_var)
        hyperopt_protections.grid(row=9, column=0, sticky="w", padx=8, pady=8)
        hyperopt_position_stacking = ttk.Checkbutton(self.hyperopt_frame, text="--eps / --enable-position-stacking", variable=self.enable_position_stacking_var)
        hyperopt_position_stacking.grid(row=9, column=1, sticky="w", padx=8, pady=8)
        ignore_missing_spaces = ttk.Checkbutton(self.hyperopt_frame, text="--ignore-missing-spaces", variable=self.hyperopt_ignore_missing_spaces_var)
        ignore_missing_spaces.grid(row=10, column=0, sticky="w", padx=8, pady=8)
        disable_param_export = ttk.Checkbutton(self.hyperopt_frame, text="--disable-param-export", variable=self.hyperopt_disable_param_export_var)
        disable_param_export.grid(row=10, column=1, sticky="w", padx=8, pady=8)
        analyze_per_epoch = ttk.Checkbutton(self.hyperopt_frame, text="--analyze-per-epoch", variable=self.hyperopt_analyze_per_epoch_var)
        analyze_per_epoch.grid(row=10, column=2, sticky="w", padx=8, pady=8)
        print_all = ttk.Checkbutton(self.hyperopt_frame, text="--print-all", variable=self.hyperopt_print_all_var)
        print_all.grid(row=11, column=0, sticky="w", padx=8, pady=8)
        monitor_results = ttk.Checkbutton(self.hyperopt_frame, text="Tail latest .fthypt during run", variable=self.hyperopt_monitor_results_var)
        monitor_results.grid(row=11, column=1, sticky="w", padx=8, pady=8)
        result_file_label, result_file_entry = self._labeled_entry(self.hyperopt_frame, 12, 0, "Result file (optional)", self.hyperopt_results_file_var)
        result_file_button = ttk.Button(self.hyperopt_frame, text="Browse", command=self.browse_hyperopt_result_file)
        result_file_button.grid(row=12, column=2, sticky="ew", padx=8, pady=8)
        poll_label, poll_entry = self._labeled_entry(self.hyperopt_frame, 13, 0, "Poll ms", self.hyperopt_results_poll_ms_var)
        self._add_tooltips(early_stop_tip, early_stop_label, early_stop_entry)
        self._add_tooltips(enable_protections_tip, hyperopt_protections)
        self._add_tooltips(position_stacking_tip, hyperopt_position_stacking)
        self._add_tooltips(ignore_missing_spaces_tip, ignore_missing_spaces)
        self._add_tooltips(disable_param_export_tip, disable_param_export)
        self._add_tooltips(analyze_per_epoch_tip, analyze_per_epoch)
        self._add_tooltips(print_all_tip, print_all)
        self._add_tooltips(monitor_tip, monitor_results)
        self._add_tooltips(result_file_tip, result_file_label, result_file_entry, result_file_button)
        self._add_tooltips(poll_ms_tip, poll_label, poll_entry)
        ttk.Label(self.hyperopt_frame, text="Leave file blank to auto-detect the newest .fthypt in userdir/hyperopt_results.", foreground="#666666", wraplength=920, justify="left").grid(row=13, column=2, columnspan=2, sticky="w", padx=8, pady=8)

        ttk.Label(
            self.other_mode_frame,
            text="Use the dedicated Download Data tab when that run type is selected.",
            wraplength=920,
            justify="left",
        ).grid(row=0, column=0, sticky="w", padx=8, pady=8)


    def _build_download_tab(self) -> None:
        root = self.tab_download
        root.grid_columnconfigure(1, weight=1)
        root.grid_columnconfigure(3, weight=1)
        root.grid_rowconfigure(6, weight=1)

        exchange_label, exchange_entry = self._labeled_entry(root, 0, 0, "Exchange", self.download_exchange_var)
        pairs_file_label, pairs_file_entry, pairs_file_button = self._path_row(
            root,
            0,
            "Pairs file",
            self.download_pairs_file_var,
            directory=False,
            tooltip="Optional pairs.json file used as the source pair list for download-data. If you already have config whitelists, the launcher can also use those instead.",
        )
        timeframes_label, timeframes_entry = self._labeled_entry(root, 1, 0, "Timeframes", self.download_timeframes_var)
        days_label, days_entry = self._labeled_entry(root, 1, 2, "Days", self.download_days_var)
        new_pairs_label, new_pairs_entry = self._labeled_entry(root, 2, 0, "New pairs days", self.download_new_pairs_days_var)
        download_timerange_label, download_timerange_entry = self._labeled_entry(root, 2, 2, "Timerange", self.download_timerange_var)

        trading_mode_label = ttk.Label(root, text="Trading mode")
        trading_mode_label.grid(row=3, column=0, sticky="w", padx=8, pady=8)
        trading_mode_combo = ttk.Combobox(root, textvariable=self.download_trading_mode_var, values=TRADING_MODE_VALUES, width=18)
        trading_mode_combo.grid(row=3, column=1, sticky="w", padx=8, pady=8)
        candle_types_label, candle_types_entry = self._labeled_entry(root, 3, 2, "Candle types", self.download_candle_types_var)

        ohlcv_label = ttk.Label(root, text="OHLCV format")
        ohlcv_label.grid(row=4, column=0, sticky="w", padx=8, pady=8)
        ohlcv_combo = ttk.Combobox(root, textvariable=self.download_data_format_ohlcv_var, values=DATA_FORMAT_VALUES, width=18)
        ohlcv_combo.grid(row=4, column=1, sticky="w", padx=8, pady=8)
        trades_label = ttk.Label(root, text="Trades format")
        trades_label.grid(row=4, column=2, sticky="w", padx=8, pady=8)
        trades_combo = ttk.Combobox(root, textvariable=self.download_data_format_trades_var, values=DATA_FORMAT_VALUES, width=18)
        trades_combo.grid(row=4, column=3, sticky="w", padx=8, pady=8)

        checks = ttk.LabelFrame(root, text="Flags")
        checks.grid(row=5, column=0, columnspan=4, sticky="ew", padx=8, pady=8)
        for i in range(3):
            checks.grid_columnconfigure(i, weight=1)
        include_inactive = ttk.Checkbutton(checks, text="--include-inactive-pairs", variable=self.download_include_inactive_var)
        include_inactive.grid(row=0, column=0, sticky="w", padx=8, pady=8)
        no_parallel = ttk.Checkbutton(checks, text="--no-parallel-download", variable=self.download_no_parallel_var)
        no_parallel.grid(row=0, column=1, sticky="w", padx=8, pady=8)
        dl_trades = ttk.Checkbutton(checks, text="--dl-trades", variable=self.download_dl_trades_var)
        dl_trades.grid(row=0, column=2, sticky="w", padx=8, pady=8)
        convert_check = ttk.Checkbutton(checks, text="--convert", variable=self.download_convert_var)
        convert_check.grid(row=1, column=0, sticky="w", padx=8, pady=8)
        erase_check = ttk.Checkbutton(checks, text="--erase", variable=self.download_erase_var)
        erase_check.grid(row=1, column=1, sticky="w", padx=8, pady=8)
        prepend_check = ttk.Checkbutton(checks, text="--prepend", variable=self.download_prepend_var)
        prepend_check.grid(row=1, column=2, sticky="w", padx=8, pady=8)

        self._add_tooltips(
            "Exchange whose market data you want to download. If a config file already defines the exchange, you can often leave this blank.",
            exchange_label,
            exchange_entry,
        )
        self._add_tooltips(
            "Source pairs file. For many workflows this is the config whitelist or a pairs.json file that enumerates which markets to download.",
            pairs_file_label,
            pairs_file_entry,
            pairs_file_button,
        )
        self._add_tooltips(
            "Space-separated list of candle timeframes to download. The default docs behavior is 1m and 5m when you do not override it.",
            timeframes_label,
            timeframes_entry,
        )
        self._add_tooltips(
            "Number of days of history to fetch when you want a relative window. Helpful for incremental refreshes.",
            days_label,
            days_entry,
        )
        self._add_tooltips(
            "Additional recent days to fetch for new pairs, letting you top up data for newly added markets.",
            new_pairs_label,
            new_pairs_entry,
        )
        self._add_tooltips(
            "Absolute or relative timerange used instead of --days. Useful when you want a fixed backtest slice or an incremental refresh from a known start date.",
            download_timerange_label,
            download_timerange_entry,
        )
        self._add_tooltips(
            "Trading mode determines whether Freqtrade fetches spot, margin, or futures data and which candle types are considered normal.",
            trading_mode_label,
            trading_mode_combo,
        )
        self._add_tooltips(
            "Selects the candle type to fetch. For futures data, Freqtrade can use spot, futures, mark, index, premiumIndex, or funding_rate depending on the exchange.",
            candle_types_label,
            candle_types_entry,
        )
        self._add_tooltips(
            "File format used for OHLCV candle data. Feather is the default, while json/jsongz/parquet are other supported storage formats.",
            ohlcv_label,
            ohlcv_combo,
        )
        self._add_tooltips(
            "File format used for historic trades data. The same supported formats apply as OHLCV, but trades are separate tick data.",
            trades_label,
            trades_combo,
        )
        self._add_tooltips(
            "Context: Include non-active markets during data acquisition. Outcome: Adds --include-inactive-pairs so delisted/inactive pairs are still fetched when available. "
            "Example: Keep historical data continuity for retired pairs.",
            include_inactive,
        )
        self._add_tooltips(
            "Context: Rate-limit safe download mode. Outcome: Adds --no-parallel-download to fetch sequentially. "
            "Example: Use on strict exchanges that throttle concurrent requests.",
            no_parallel,
        )
        self._add_tooltips(
            "Context: Tick-level history workflow. Outcome: Adds --dl-trades to fetch trades instead of OHLCV candles. "
            "Example: Download trades first, then resample locally.",
            dl_trades,
        )
        self._add_tooltips(
            "Context: Data transformation mode. Outcome: Adds --convert to rewrite existing files into selected formats. "
            "Example: Convert prior JSON OHLCV data to Feather.",
            convert_check,
        )
        self._add_tooltips(
            "Context: Clean refresh mode. Outcome: Adds --erase to remove matching local data before download. "
            "Example: Rebuild a corrupted timeframe dataset from scratch.",
            erase_check,
        )
        self._add_tooltips(
            "Context: Historical extension mode. Outcome: Adds --prepend to fetch older candles before existing start date. "
            "Example: Extend 2023 history back into 2022.",
            prepend_check,
        )

        download_pairs = ttk.LabelFrame(root, text="Download pairs")
        download_pairs.grid(row=6, column=0, columnspan=4, sticky="nsew", padx=8, pady=8)
        download_pairs.grid_columnconfigure(0, weight=1)
        download_pairs.grid_rowconfigure(1, weight=1)
        download_pair_buttons = ttk.Frame(download_pairs)
        download_pair_buttons.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 0))
        add_download = ttk.Button(download_pair_buttons, text="Add typed", command=lambda: self.add_typed_pair(self.download_pairs_text, set_manual_mode=False))
        add_download.pack(side="left", padx=(0, 8))
        remove_download = ttk.Button(download_pair_buttons, text="Remove selected", command=lambda: self.remove_selected_pairs(self.download_pairs_text))
        remove_download.pack(side="left", padx=(0, 8))
        normalize_download = ttk.Button(download_pair_buttons, text="Normalize", command=lambda: self.normalize_pair_text(self.download_pairs_text, set_manual_mode=False))
        normalize_download.pack(side="left", padx=(0, 8))
        use_whitelist = ttk.Button(download_pair_buttons, text="Use whitelist", command=self.copy_whitelist_to_download_pairs)
        use_whitelist.pack(side="left")
        self.download_pairs_text = scrolledtext.ScrolledText(download_pairs, wrap="word", height=8)
        self.download_pairs_text.grid(row=1, column=0, sticky="nsew", padx=8, pady=8)
        self.download_pairs_text.bind("<KeyRelease>", lambda _e: self.refresh_all_command_previews())
        ToolTip(
            self.download_pairs_text,
            "Context: Explicit pair override for Download Data command. Outcome: Filled list becomes `-p` pairs; empty list falls back to Pairs tab whitelist in manual mode. "
            "Example: Keep download scope to `BTC/USDT ETH/USDT` only.",
        )
        ToolTip(
            add_download,
            "Context: Quick pair entry for Download Data. Outcome: Appends typed pairs to the download pair box. "
            "Example: Add `BTC/USDT XRP/USDT` before running download-data.",
        )
        ToolTip(
            remove_download,
            "Context: Download pair list cleanup. Outcome: Removes selected/current line(s) from Download pairs. "
            "Example: Remove a pair with repeated API errors.",
        )
        ToolTip(
            normalize_download,
            "Context: Pair formatting cleanup before CLI build. Outcome: Trims whitespace and deduplicates download pairs. "
            "Example: Collapse repeated symbols into one entry.",
        )
        ToolTip(
            use_whitelist,
            "Context: Reuse manual whitelist for data collection. Outcome: Copies Pairs tab whitelist into Download pairs box. "
            "Example: Keep download set aligned with backtest pair set.",
        )
        ttk.Label(
            download_pairs,
            text="If this box has pairs, Download Data uses it for -p. If it is empty, manual mode falls back to the Pairs tab whitelist.",
            wraplength=920,
            justify="left",
        ).grid(row=2, column=0, sticky="w", padx=8, pady=(0, 8))

        ttk.Label(
            root,
            text="Timeframes can be comma or space separated, for example: 5m 15m 1h. Timerange example: 20240101-20240331.",
            wraplength=920,
            justify="left",
        ).grid(row=7, column=0, columnspan=4, sticky="w", padx=8, pady=(0, 8))

    def _build_news_lab_tab(self) -> None:
        root = self.tab_news_lab
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(7, weight=1)

        intro_text = (
            "News Lab starts a separate background collector that stores timestamped news in SQLite and keeps running "
            "after this launcher window closes. Use Start Detached Collector to launch it, Refresh Status to check "
            "health, and Request Stop when you want a clean shutdown."
        )
        intro = ttk.Label(root, text=intro_text, wraplength=1180, justify="left", foreground="#555555")
        intro.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 6))
        ToolTip(
            intro,
            "High-level overview of how News Lab works. This tab is only a controller for the standalone collector: "
            "it starts the collector, checks status files and the SQLite database, opens folders/logs, and exports data. "
            "It does not keep the collector alive itself, so closing the launcher should not stop a collector that was "
            "started with Start Detached Collector.",
        )

        controls = ttk.LabelFrame(root, text="Collector settings")
        controls.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        controls.grid_columnconfigure(1, weight=1)
        controls.grid_columnconfigure(3, weight=1)

        config_label = ttk.Label(controls, text="Config path")
        config_label.grid(row=0, column=0, sticky="w", padx=8, pady=8)
        config_entry = ttk.Entry(controls, textvariable=self.news_config_path_var)
        config_entry.grid(row=0, column=1, sticky="ew", padx=8, pady=8)
        config_button = ttk.Button(controls, text="Browse", command=lambda: self._browse_path(self.news_config_path_var, False))
        config_button.grid(row=0, column=2, sticky="ew", padx=8, pady=8)
        config_tip = (
            "JSON file that defines which feeds and APIs the collector should use. The collector reloads this file at "
            "the start of every polling cycle, so you can enable, disable, or add sources without restarting the process. "
            "If the selected file does not exist, the collector creates a starter config automatically."
        )
        ToolTip(config_label, config_tip)
        ToolTip(config_entry, config_tip)
        ToolTip(
            config_button,
            "Context: Select collector source configuration. Outcome: Sets Config path field used to build --config. "
            "Example: Choose `research/config/news_research_sources.json`.",
        )

        data_dir_label = ttk.Label(controls, text="Data dir")
        data_dir_label.grid(row=1, column=0, sticky="w", padx=8, pady=8)
        data_dir_entry = ttk.Entry(controls, textvariable=self.news_data_dir_var)
        data_dir_entry.grid(row=1, column=1, sticky="ew", padx=8, pady=8)
        data_dir_button = ttk.Button(controls, text="Browse", command=lambda: self._browse_path(self.news_data_dir_var, True))
        data_dir_button.grid(row=1, column=2, sticky="ew", padx=8, pady=8)
        data_dir_tip = (
            "Runtime working folder for News Lab. This directory holds the SQLite database, collector status JSON, PID "
            "file, stop flag, raw payload snapshots, logs, exports, and any later analysis outputs. Keeping everything "
            "under one folder makes backup, inspection, and cleanup much simpler."
        )
        ToolTip(data_dir_label, data_dir_tip)
        ToolTip(data_dir_entry, data_dir_tip)
        ToolTip(
            data_dir_button,
            "Context: Select News Lab runtime workspace. Outcome: Sets folder for DB, status, logs, snapshots, and exports. "
            "Example: Choose `user_data/runtime/news`.",
        )

        db_label = ttk.Label(controls, text="Database")
        db_label.grid(row=2, column=0, sticky="w", padx=8, pady=8)
        db_entry = ttk.Entry(controls, textvariable=self.news_db_path_var)
        db_entry.grid(row=2, column=1, sticky="ew", padx=8, pady=8)
        db_button = ttk.Button(controls, text="Browse", command=lambda: self._browse_path(self.news_db_path_var, False))
        db_button.grid(row=2, column=2, sticky="ew", padx=8, pady=8)
        db_tip = (
            "Normalized SQLite database used for deduped article storage, source health tracking, fetch logs, and CSV "
            "exports. By default it lives inside the Data dir as news_events.sqlite. Change this only if you intentionally "
            "want the collector to write to a different database file."
        )
        ToolTip(db_label, db_tip)
        ToolTip(db_entry, db_tip)
        ToolTip(
            db_button,
            "Context: Select or create the News Lab SQLite file path. Outcome: Sets --db target for normalized storage. "
            "Example: `user_data/runtime/news/news_events.sqlite`.",
        )

        interval_label = ttk.Label(controls, text="Poll interval minutes")
        interval_label.grid(row=3, column=0, sticky="w", padx=8, pady=8)
        interval_entry = ttk.Entry(controls, textvariable=self.news_interval_seconds_var)
        interval_entry.grid(row=3, column=1, sticky="ew", padx=8, pady=8)
        interval_tip = (
            "How long the collector waits between full polling cycles, in minutes. The launcher rounds to the nearest "
            "whole minute, then converts that to seconds for the collector process. Shorter intervals give fresher "
            "data but create more requests, more log volume, and more raw snapshots. Leave blank to use the default "
            "15-minute interval."
        )
        ToolTip(interval_label, interval_tip)
        ToolTip(interval_entry, interval_tip)

        run_once_check = ttk.Checkbutton(controls, text="Run once", variable=self.news_once_var)
        run_once_check.grid(row=3, column=2, sticky="w", padx=8, pady=8)
        ToolTip(
            run_once_check,
            "Run exactly one collection cycle and then exit cleanly. This is useful for smoke tests, validating a new "
            "source list, or checking that storage/export paths work before you leave a long-running detached collector "
            "running in the background.",
        )

        preview = ttk.LabelFrame(root, text="Generated command")
        preview.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        preview.grid_columnconfigure(0, weight=1)
        preview_entry = ttk.Entry(preview, textvariable=self.news_command_preview_var)
        preview_entry.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        ToolTip(
            preview,
            "Read-only preview of the exact Python command the launcher will use for News Lab. This helps with auditability "
            "and makes it easy to reproduce a run manually from a terminal if you ever want to troubleshoot outside the UI.",
        )
        ToolTip(
            preview_entry,
            "Exact detached collector command. Start Detached Collector uses this command, including paths, interval, and "
            "the optional --once flag. Editing the fields above updates this preview so you can see precisely what will run.",
        )

        buttons = ttk.Frame(root)
        buttons.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 8))
        start_button = ttk.Button(buttons, text="Start Detached Collector", command=self.on_start_news_collector)
        start_button.pack(side="left", padx=(0, 8))
        stop_button = ttk.Button(buttons, text="Request Stop", command=self.on_stop_news_collector)
        stop_button.pack(side="left", padx=(0, 8))
        refresh_button = ttk.Button(buttons, text="Refresh Status", command=self.refresh_news_status)
        refresh_button.pack(side="left", padx=(0, 8))
        open_data_button = ttk.Button(buttons, text="Open Data Folder", command=self.open_news_data_folder)
        open_data_button.pack(side="left", padx=(0, 8))
        open_log_button = ttk.Button(buttons, text="Open Log", command=self.open_news_log)
        open_log_button.pack(side="left", padx=(0, 8))
        export_articles_button = ttk.Button(buttons, text="Export CSV", command=self.on_export_news_csv)
        export_articles_button.pack(side="left", padx=(0, 8))
        export_health_button = ttk.Button(buttons, text="Export Source Health CSV", command=self.on_export_source_health_csv)
        export_health_button.pack(side="left", padx=(0, 8))
        ToolTip(
            start_button,
            "Starts the standalone News Lab collector as a detached background process. It does not use the Run, Explorer, "
            "or FreqUI shared process slot, and it should keep running even if you close this launcher window afterward. "
            "Output goes to the News Lab log file instead of the launcher console.",
        )
        ToolTip(
            stop_button,
            "Requests a clean shutdown by creating collector.stop in the data folder. The collector notices that flag after "
            "the current source finishes or during the next sleep check, writes final status, removes its PID file, and exits. "
            "This is a polite stop request, not a force kill.",
        )
        ToolTip(
            refresh_button,
            "Reloads the current News Lab state from collector_status.json and repopulates the Source health table from the "
            "SQLite database. Use this after launching, after requesting stop, or whenever you want an updated heartbeat and "
            "error snapshot.",
        )
        ToolTip(
            open_data_button,
            "Opens the News Lab runtime folder so you can inspect the database, raw snapshots, exports, analysis outputs, "
            "PID/status files, and the stop flag directly.",
        )
        ToolTip(
            open_log_button,
            "Opens the collector log file. This is the first place to check if a source starts failing, a feed returns bad "
            "data, the machine loses network access, or you want to confirm that detached startup worked.",
        )
        ToolTip(
            export_articles_button,
            "Exports the normalized articles table from SQLite to a timestamped CSV in user_data/runtime/news/exports. This is useful "
            "for spreadsheet review, downstream analytics, labeling work, or quick sanity checks outside the launcher.",
        )
        ToolTip(
            export_health_button,
            "Exports the per-source health table to CSV so you can review which feeds are healthy, failing, or mostly "
            "producing duplicates over time.",
        )

        status = ttk.LabelFrame(root, text="Status")
        status.grid(row=4, column=0, sticky="ew", padx=8, pady=(0, 8))
        for i in range(4):
            status.grid_columnconfigure(i, weight=1 if i in (1, 3) else 0)
        ToolTip(
            status,
            "Live status summary loaded from collector_status.json. If the status file says running but the heartbeat has "
            "not advanced for more than 10 minutes, the launcher marks it as stale/unknown rather than assuming the process "
            "is still healthy.",
        )
        status_items = [
            ("Status", self.news_status_var),
            ("PID", self.news_pid_var),
            ("Started at", self.news_started_at_var),
            ("Heartbeat at", self.news_heartbeat_at_var),
            ("Last fetch at", self.news_last_fetch_at_var),
            ("Total articles", self.news_total_articles_var),
            ("New articles last cycle", self.news_new_articles_last_cycle_var),
            ("Last error", self.news_last_error_var),
        ]
        status_tips = {
            "Status": "Overall collector state reported by the standalone process. Typical values are running, stopped, stop requested, error, or stale/unknown.",
            "PID": "Operating system process ID written by the collector while it is running. The collector owns this file and removes it on a clean shutdown.",
            "Started at": "UTC timestamp for when the current collector run started.",
            "Heartbeat at": "Most recent UTC heartbeat written by the collector. This should keep moving forward while the collector is healthy.",
            "Last fetch at": "UTC timestamp of the most recent source fetch attempt completed by the collector.",
            "Total articles": "Current total row count in the normalized articles table. This number should grow over time without duplicating the same stories.",
            "New articles last cycle": "How many brand-new articles were inserted during the most recent full polling cycle.",
            "Last error": "Most recent collector or source-level error summary written into status. Open the log if you need the full traceback or repeated error history.",
        }
        for index, (label_text, variable) in enumerate(status_items):
            row = index // 2
            col = (index % 2) * 2
            label_widget = ttk.Label(status, text=f"{label_text}:")
            label_widget.grid(row=row, column=col, sticky="w", padx=8, pady=4)
            value_widget = ttk.Label(status, textvariable=variable)
            value_widget.grid(row=row, column=col + 1, sticky="w", padx=8, pady=4)
            ToolTip(label_widget, status_tips.get(label_text, ""))
            ToolTip(value_widget, status_tips.get(label_text, ""))

        self._research_lab_build_source_health_section(
            root,
            "news",
            5,
            "Source-by-source detail. last_success_at and last_failure_at tell you whether a feed is currently healthy. "
            "items_last_fetch shows how many items were seen, inserted_last_fetch shows how many were truly new, and "
            "duplicates_last_fetch shows how much of the latest poll was already known.",
        )

    def _build_web_lab_tab(self) -> None:
        root = self.tab_web_lab
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(7, weight=1)

        intro = ttk.Label(
            root,
            text=(
                "Web Lab starts a separate background scraper for crypto-relevant exchange, protocol, and media pages. "
                "It is intended for official pages and blogs that may not offer a clean API or may only update a few times per day."
            ),
            wraplength=1180,
            justify="left",
            foreground="#555555",
        )
        intro.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 6))
        ToolTip(
            intro,
            "Context: Web Lab architecture overview. Outcome: Explains this tab controls a detached scraper that persists data to SQLite outside launcher lifecycle. "
            "Example: Start collector, close launcher, and data collection continues.",
        )

        controls = ttk.LabelFrame(root, text="Collector settings")
        controls.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        controls.grid_columnconfigure(1, weight=1)
        controls.grid_columnconfigure(3, weight=1)

        config_label = ttk.Label(controls, text="Config path")
        config_label.grid(row=0, column=0, sticky="w", padx=8, pady=8)
        config_entry = ttk.Entry(controls, textvariable=self.web_config_path_var)
        config_entry.grid(row=0, column=1, sticky="ew", padx=8, pady=8)
        config_button = ttk.Button(controls, text="Browse", command=lambda: self._browse_path(self.web_config_path_var, False))
        config_button.grid(row=0, column=2, sticky="ew", padx=8, pady=8)
        ToolTip(
            config_label,
            "Context: Web collector source-definition file. Outcome: This path becomes --config for detached runs. "
            "Example: `research/config/web_research_sources.json`.",
        )
        ToolTip(
            config_entry,
            "Context: Web collector source-definition file. Outcome: This path becomes --config for detached runs. "
            "Example: `research/config/web_research_sources.json`.",
        )
        ToolTip(
            config_button,
            "Context: Select the Web Lab source-definition JSON. Outcome: Updates Config path field. "
            "Example: Pick an alternate config for a niche source set.",
        )

        data_dir_label = ttk.Label(controls, text="Data dir")
        data_dir_label.grid(row=1, column=0, sticky="w", padx=8, pady=8)
        data_dir_entry = ttk.Entry(controls, textvariable=self.web_data_dir_var)
        data_dir_entry.grid(row=1, column=1, sticky="ew", padx=8, pady=8)
        data_dir_button = ttk.Button(controls, text="Browse", command=lambda: self._browse_path(self.web_data_dir_var, True))
        data_dir_button.grid(row=1, column=2, sticky="ew", padx=8, pady=8)
        ToolTip(
            data_dir_label,
            "Context: Web Lab runtime workspace root. Outcome: Holds DB, status, logs, snapshots, and exports. "
            "Example: `user_data/runtime/web`.",
        )
        ToolTip(
            data_dir_entry,
            "Context: Web Lab runtime workspace root. Outcome: Holds DB, status, logs, snapshots, and exports. "
            "Example: `user_data/runtime/web`.",
        )
        ToolTip(
            data_dir_button,
            "Context: Select runtime workspace location. Outcome: Updates Data dir used by detached collector. "
            "Example: Point to a larger disk for long runs.",
        )

        db_label = ttk.Label(controls, text="Database")
        db_label.grid(row=2, column=0, sticky="w", padx=8, pady=8)
        db_entry = ttk.Entry(controls, textvariable=self.web_db_path_var)
        db_entry.grid(row=2, column=1, sticky="ew", padx=8, pady=8)
        db_button = ttk.Button(controls, text="Browse", command=lambda: self._browse_path(self.web_db_path_var, False))
        db_button.grid(row=2, column=2, sticky="ew", padx=8, pady=8)
        ToolTip(
            db_label,
            "Context: Web Lab normalized article database. Outcome: Collector writes deduped content and health rows to this SQLite file. "
            "Example: `user_data/runtime/web/web_events.sqlite`.",
        )
        ToolTip(
            db_entry,
            "Context: Web Lab normalized article database. Outcome: Collector writes deduped content and health rows to this SQLite file. "
            "Example: `user_data/runtime/web/web_events.sqlite`.",
        )
        ToolTip(
            db_button,
            "Context: Select existing or new SQLite file for Web Lab. Outcome: Updates DB path field used in command preview. "
            "Example: Route output to an external analysis DB file.",
        )

        interval_label = ttk.Label(controls, text="Poll interval minutes")
        interval_label.grid(row=3, column=0, sticky="w", padx=8, pady=8)
        interval_entry = ttk.Entry(controls, textvariable=self.web_interval_seconds_var)
        interval_entry.grid(row=3, column=1, sticky="ew", padx=8, pady=8)
        ToolTip(interval_label, "How often Web Lab runs a full scrape cycle, in minutes. The launcher rounds to the nearest whole minute and converts that to seconds for the collector. Leave blank to use the default 360-minute interval.")
        ToolTip(interval_entry, "How often Web Lab runs a full scrape cycle, in minutes. The launcher rounds to the nearest whole minute and converts that to seconds for the collector. Leave blank to use the default 360-minute interval.")

        run_once_check = ttk.Checkbutton(controls, text="Run once", variable=self.web_once_var)
        run_once_check.grid(row=3, column=2, sticky="w", padx=8, pady=8)
        ToolTip(
            run_once_check,
            "Context: One-shot validation mode for scraper configuration. Outcome: Collector performs one cycle then exits cleanly. "
            "Example: Test new source filters before enabling long-running mode.",
        )

        preview = ttk.LabelFrame(root, text="Generated command")
        preview.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        preview.grid_columnconfigure(0, weight=1)
        preview_entry = ttk.Entry(preview, textvariable=self.web_command_preview_var)
        preview_entry.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        ToolTip(
            preview_entry,
            "Context: Final detached command generated from current Web Lab fields. Outcome: Shows exact command Start Detached Collector executes. "
            "Example: Verify interval/path arguments before launch.",
        )

        buttons = ttk.Frame(root)
        buttons.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 8))
        start_button = ttk.Button(buttons, text="Start Detached Collector", command=self.on_start_web_collector)
        start_button.pack(side="left", padx=(0, 8))
        stop_button = ttk.Button(buttons, text="Request Stop", command=self.on_stop_web_collector)
        stop_button.pack(side="left", padx=(0, 8))
        refresh_button = ttk.Button(buttons, text="Refresh Status", command=self.refresh_web_status)
        refresh_button.pack(side="left", padx=(0, 8))
        open_data_button = ttk.Button(buttons, text="Open Data Folder", command=self.open_web_data_folder)
        open_data_button.pack(side="left", padx=(0, 8))
        open_log_button = ttk.Button(buttons, text="Open Log", command=self.open_web_log)
        open_log_button.pack(side="left", padx=(0, 8))
        export_articles_button = ttk.Button(buttons, text="Export CSV", command=self.on_export_web_csv)
        export_articles_button.pack(side="left", padx=(0, 8))
        export_health_button = ttk.Button(buttons, text="Export Source Health CSV", command=self.on_export_web_source_health_csv)
        export_health_button.pack(side="left", padx=(0, 8))
        ToolTip(
            start_button,
            "Context: Launch background web collector. Outcome: Starts detached process that continues even if launcher closes. "
            "Example: Start overnight scraping and check status later.",
        )
        ToolTip(
            stop_button,
            "Context: Graceful stop request for detached collector. Outcome: Writes stop flag and collector exits after safe checkpoint. "
            "Example: Stop before rotating config or moving data dir.",
        )
        ToolTip(
            refresh_button,
            "Context: Runtime status sync action. Outcome: Reloads status JSON and source-health table from SQLite. "
            "Example: Confirm heartbeat and last error after startup.",
        )
        ToolTip(
            open_data_button,
            "Context: Quick access to runtime artifacts. Outcome: Opens data folder containing DB, status files, and exports. "
            "Example: Inspect snapshots for parsing diagnostics.",
        )
        ToolTip(
            open_log_button,
            "Context: Log-first troubleshooting path. Outcome: Opens collector log for network/parser/runtime errors. "
            "Example: Investigate repeated scrape failures.",
        )
        ToolTip(
            export_articles_button,
            "Context: Extract normalized article table for external analysis. Outcome: Writes article rows to CSV. "
            "Example: Load into pandas for scoring experiments.",
        )
        ToolTip(
            export_health_button,
            "Context: Extract source reliability metrics. Outcome: Writes source-health rows to CSV. "
            "Example: Rank unreliable sources before pruning config.",
        )

        status = ttk.LabelFrame(root, text="Status")
        status.grid(row=4, column=0, sticky="ew", padx=8, pady=(0, 8))
        for i in range(4):
            status.grid_columnconfigure(i, weight=1 if i in (1, 3) else 0)
        status_items = [
            ("Status", self.web_status_var),
            ("PID", self.web_pid_var),
            ("Started at", self.web_started_at_var),
            ("Heartbeat at", self.web_heartbeat_at_var),
            ("Last fetch at", self.web_last_fetch_at_var),
            ("Total articles", self.web_total_articles_var),
            ("New articles last cycle", self.web_new_articles_last_cycle_var),
            ("Last error", self.web_last_error_var),
        ]
        for index, (label_text, variable) in enumerate(status_items):
            row = index // 2
            col = (index % 2) * 2
            ttk.Label(status, text=f"{label_text}:").grid(row=row, column=col, sticky="w", padx=8, pady=4)
            ttk.Label(status, textvariable=variable).grid(row=row, column=col + 1, sticky="w", padx=8, pady=4)

        self._research_lab_build_source_health_section(
            root,
            "web",
            5,
            "Shows per-source Web Lab health. This helps catch fragile pages, empty scrapes, duplicates, and repeated failures.",
        )

    def _build_orderbook_lab_tab(self) -> None:
        root = self.tab_orderbook_lab
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(5, weight=1)

        settings = ttk.LabelFrame(root, text="Collector settings")
        settings.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        settings.grid_columnconfigure(1, weight=1)
        settings.grid_columnconfigure(3, weight=1)

        self._path_row(settings, 0, "Config path", self.orderbook_config_path_var, directory=False)
        self._path_row(settings, 1, "Data dir", self.orderbook_data_dir_var, directory=True)

        ttk.Label(settings, text="Exchange").grid(row=2, column=0, sticky="w", padx=8, pady=8)
        ttk.Combobox(
            settings,
            textvariable=self.orderbook_exchange_var,
            values=["Binance USD-M Futures"],
            state="readonly",
            width=24,
        ).grid(row=2, column=1, sticky="ew", padx=8, pady=8)
        ttk.Label(settings, text="Market type").grid(row=2, column=2, sticky="w", padx=8, pady=8)
        ttk.Combobox(
            settings,
            textvariable=self.orderbook_market_type_var,
            values=["futures"],
            state="readonly",
            width=12,
        ).grid(row=2, column=3, sticky="ew", padx=8, pady=8)

        numeric_rows = [
            ("Depth levels", self.orderbook_depth_levels_var, "Stream update ms", self.orderbook_stream_update_ms_var),
            ("Metric interval seconds", self.orderbook_metric_interval_seconds_var, "Snapshot interval seconds", self.orderbook_snapshot_interval_seconds_var),
            ("Capacity warning MB", self.orderbook_capacity_warning_mb_var, "Capacity critical MB", self.orderbook_capacity_critical_mb_var),
            ("Max symbols", self.orderbook_max_symbols_var, "", None),
        ]
        for offset, (left_label, left_var, right_label, right_var) in enumerate(numeric_rows, start=3):
            ttk.Label(settings, text=left_label).grid(row=offset, column=0, sticky="w", padx=8, pady=8)
            ttk.Entry(settings, textvariable=left_var, width=12).grid(row=offset, column=1, sticky="ew", padx=8, pady=8)
            if right_var is not None:
                ttk.Label(settings, text=right_label).grid(row=offset, column=2, sticky="w", padx=8, pady=8)
                ttk.Entry(settings, textvariable=right_var, width=12).grid(row=offset, column=3, sticky="ew", padx=8, pady=8)

        ttk.Checkbutton(settings, text="Store snapshots", variable=self.orderbook_store_snapshots_var).grid(row=7, column=0, columnspan=2, sticky="w", padx=8, pady=8)

        actions = ttk.Frame(root)
        actions.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        ttk.Button(actions, text="Start Detached Collector", command=self.start_orderbook_collector).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Request Stop", command=self.request_orderbook_stop).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Refresh Status", command=lambda: self.refresh_orderbook_status(show_popup=True)).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Open Data Folder", command=self.open_orderbook_data_dir).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Open Log", command=self.open_orderbook_log).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Export Latest Metrics CSV", command=self.export_orderbook_latest_metrics_csv).pack(side="left", padx=(0, 8))
        ttk.Button(actions, text="Normalize Whitelist", command=self.on_orderbook_normalize_whitelist).pack(side="left", padx=(0, 8))

        status = ttk.LabelFrame(root, text="Status")
        status.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        for col in range(4):
            status.grid_columnconfigure(col, weight=1 if col in (1, 3) else 0)
        status_items = [
            ("Status", self.orderbook_status_var),
            ("PID", self.orderbook_pid_var),
            ("Started at", self.orderbook_started_at_var),
            ("Heartbeat at", self.orderbook_heartbeat_at_var),
            ("Last message at", self.orderbook_last_message_at_var),
            ("Last metric at", self.orderbook_last_metric_at_var),
            ("Pair count", self.orderbook_pair_count_var),
            ("Active streams", self.orderbook_active_streams_var),
            ("Messages", self.orderbook_message_count_var),
            ("Metrics", self.orderbook_metric_count_var),
            ("DB MB", self.orderbook_db_mb_var),
            ("Data dir MB", self.orderbook_data_dir_mb_var),
            ("Capacity", self.orderbook_capacity_level_var),
            ("Last error", self.orderbook_last_error_var),
        ]
        for index, (label_text, variable) in enumerate(status_items):
            row = index // 2
            col = (index % 2) * 2
            ttk.Label(status, text=f"{label_text}:").grid(row=row, column=col, sticky="w", padx=8, pady=4)
            ttk.Label(status, textvariable=variable).grid(row=row, column=col + 1, sticky="w", padx=8, pady=4)

        estimate = ttk.LabelFrame(root, text="Storage estimate")
        estimate.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 8))
        estimate_items = [
            ("Metric rows/day", self.orderbook_estimated_metric_rows_var),
            ("Snapshot rows/day", self.orderbook_estimated_snapshot_rows_var),
            ("Estimated MB/day", self.orderbook_estimated_mb_per_day_var),
            ("Days to warning", self.orderbook_estimated_days_to_warning_var),
            ("Whitelist pairs", self.orderbook_preview_pair_count_var),
            ("Active symbols", self.orderbook_preview_symbol_count_var),
        ]
        for index, (label_text, variable) in enumerate(estimate_items):
            ttk.Label(estimate, text=f"{label_text}:").grid(row=0, column=index * 2, sticky="w", padx=8, pady=6)
            ttk.Label(estimate, textvariable=variable).grid(row=0, column=index * 2 + 1, sticky="w", padx=8, pady=6)

        pair_frame = ttk.LabelFrame(root, text="Whitelist preview")
        pair_frame.grid(row=4, column=0, sticky="ew", padx=8, pady=(0, 8))
        pair_frame.grid_columnconfigure(0, weight=1)
        self.orderbook_pair_preview_tree = ttk.Treeview(pair_frame, columns=("pair", "symbol", "status"), show="headings", height=5)
        for column, width in (("pair", 180), ("symbol", 140), ("status", 140)):
            self.orderbook_pair_preview_tree.heading(column, text=column)
            self.orderbook_pair_preview_tree.column(column, width=width, anchor="w")
        self.orderbook_pair_preview_tree.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        ttk.Label(pair_frame, textvariable=self.orderbook_pair_warning_var).grid(row=1, column=0, sticky="w", padx=8, pady=(0, 8))

        metrics = ttk.LabelFrame(root, text="Latest metrics")
        metrics.grid(row=5, column=0, sticky="nsew", padx=8, pady=(0, 8))
        metrics.grid_columnconfigure(0, weight=1)
        metrics.grid_rowconfigure(0, weight=1)
        columns = (
            "pair",
            "symbol",
            "status",
            "best_bid",
            "best_ask",
            "spread_bps",
            "imbalance_top20",
            "bid_pressure_ratio_60s",
            "ask_pressure_ratio_60s",
            "nearest_bid_wall_distance_bps",
            "nearest_ask_wall_distance_bps",
            "last_metric_at",
        )
        self.orderbook_latest_metrics_tree = ttk.Treeview(metrics, columns=columns, show="headings", height=8)
        for column in columns:
            self.orderbook_latest_metrics_tree.heading(column, text=column)
            self.orderbook_latest_metrics_tree.column(column, width=150 if column != "last_metric_at" else 220, anchor="w")
        self.orderbook_latest_metrics_tree.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        scrollbar = ttk.Scrollbar(metrics, orient="vertical", command=self.orderbook_latest_metrics_tree.yview)
        scrollbar.grid(row=0, column=1, sticky="ns", pady=8)
        self.orderbook_latest_metrics_tree.configure(yscrollcommand=scrollbar.set)


    def _build_explorer_tab(self) -> None:
        explorer_notebook = ttk.Notebook(self.tab_explorer)
        explorer_notebook.grid(row=0, column=0, sticky="nsew")
        self.tab_explorer.grid_columnconfigure(0, weight=1)
        self.tab_explorer.grid_rowconfigure(0, weight=1)
        root = ttk.Frame(explorer_notebook)
        catalog_tab = ttk.Frame(explorer_notebook)
        explorer_notebook.add(root, text="Run Explorer")
        explorer_notebook.add(catalog_tab, text="Catalog / Custom Batches")
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(4, weight=1)

        explorer_actions = ttk.Frame(root)
        explorer_actions.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 0))
        self.explorer_tab_run_button = ttk.Button(explorer_actions, text="Run Explorer", command=self.on_run_explorer)
        self.explorer_tab_run_button.pack(side="left")
        ToolTip(
            self.explorer_tab_run_button,
            "Context: Starts the Explorer workflow from the Explorer tab configuration. Outcome: Runs explorer runner with current explorer options. "
            "Example: Start random-tag exploration with selected windows.",
        )

        controls = ttk.LabelFrame(root, text="1. Explorer settings")
        controls.grid(row=1, column=0, sticky="ew", padx=8, pady=(8, 8))
        for i in range(8):
            controls.grid_columnconfigure(i, weight=1 if i in (1, 3, 5, 7) else 0)

        selection_mode_label = ttk.Label(controls, text="Selection mode")
        selection_mode_label.grid(row=0, column=0, sticky="w", padx=8, pady=8)
        selection_mode_combo = ttk.Combobox(controls, textvariable=self.explorer_selection_mode_var, values=EXPLORER_SELECTION_MODES, state="readonly", width=22)
        selection_mode_combo.grid(row=0, column=1, sticky="ew", padx=8, pady=8)
        selection_mode_tip = "Strategy tags are the source of truth. Random families samples family:* values, Random tags samples all tags, Random namespace values samples values such as mode:* or switch:*, and Custom batches uses committed exact target groups."
        ToolTip(selection_mode_label, selection_mode_tip)
        ToolTip(selection_mode_combo, selection_mode_tip)
        target_count_label = ttk.Label(controls, text="Target count")
        target_count_label.grid(row=0, column=2, sticky="w", padx=8, pady=8)
        target_count_entry = ttk.Entry(controls, textvariable=self.explorer_target_count_var, width=10)
        target_count_entry.grid(row=0, column=3, sticky="ew", padx=8, pady=8)
        self.explorer_target_count_label_widget = target_count_label
        self.explorer_target_count_entry_widget = target_count_entry
        target_count_tip = (
            "Context: Size control for the current selection mode. Outcome: Sets how many families/tags/namespace-values are sampled; "
            "blank uses mode defaults when supported. Example: Enter 12 to sample twelve targets."
        )
        self.explorer_target_count_tooltips = [ToolTip(target_count_label, target_count_tip), ToolTip(target_count_entry, target_count_tip)]
        self._refresh_explorer_target_count_label()
        sampling_seed_label = ttk.Label(controls, text="Sampling seed")
        sampling_seed_label.grid(row=1, column=0, sticky="w", padx=8, pady=(0, 8))
        sampling_seed_entry = ttk.Entry(controls, textvariable=self.explorer_sampling_seed_var, width=10)
        sampling_seed_entry.grid(row=1, column=1, sticky="ew", padx=8, pady=(0, 8))
        sampling_seed_tip = "Optional seed for Explorer target sampling. Leave blank to omit --sampling-seed."
        ToolTip(sampling_seed_label, sampling_seed_tip)
        ToolTip(sampling_seed_entry, sampling_seed_tip)
        max_loops_label = ttk.Label(controls, text="Max loops")
        max_loops_label.grid(row=1, column=2, sticky="w", padx=8, pady=(0, 8))
        max_loops_entry = ttk.Entry(controls, textvariable=self.explorer_max_loops_var, width=10)
        max_loops_entry.grid(row=1, column=3, sticky="ew", padx=8, pady=(0, 8))
        max_loops_tip = "Blank uses the runner default. 0 means uncapped and runs until stopped. A positive number runs exactly that many Explorer loops."
        ToolTip(max_loops_label, max_loops_tip)
        ToolTip(max_loops_entry, max_loops_tip)
        namespace_label = ttk.Label(controls, text="Namespace")
        namespace_label.grid(row=1, column=4, sticky="w", padx=8, pady=(0, 8))
        self.explorer_namespace_label_widget = namespace_label
        self.explorer_target_namespace_combo = ttk.Combobox(controls, textvariable=self.explorer_target_namespace_var, values=[], state="readonly", width=18)
        self.explorer_target_namespace_combo.grid(row=1, column=5, sticky="ew", padx=8, pady=(0, 8))
        namespace_tip = "Used by Random namespace values. The list is discovered dynamically from strategy tag prefixes."
        ToolTip(namespace_label, namespace_tip)
        ToolTip(self.explorer_target_namespace_combo, namespace_tip)

        epochs_label = ttk.Label(controls, text="Epoch multiplier")
        epochs_label.grid(row=2, column=0, sticky="w", padx=8, pady=(0, 8))
        epochs_entry = ttk.Entry(controls, textvariable=self.explorer_epochs_var, width=10)
        epochs_entry.grid(row=2, column=1, sticky="ew", padx=8, pady=(0, 8))
        epochs_tip = (
            "Multiplier used to compute total epochs for each Explorer hyperopt run as: "
            "job workers * epoch multiplier."
        )
        epochs_tip += f" Total epochs are capped at {MAX_HYPEROPT_EPOCHS}. Leave blank to omit --epochs and use runner defaults."
        ToolTip(epochs_label, epochs_tip)
        ToolTip(epochs_entry, epochs_tip)
        explorer_random_state_label = ttk.Label(controls, text="Random state")
        explorer_random_state_label.grid(row=2, column=2, sticky="w", padx=8, pady=(0, 8))
        explorer_random_state_entry = ttk.Entry(controls, textvariable=self.explorer_random_state_var, width=10)
        explorer_random_state_entry.grid(row=2, column=3, sticky="ew", padx=8, pady=(0, 8))
        explorer_random_state_tip = "Optional Hyperopt random state override. Leave blank to omit --random-state."
        ToolTip(explorer_random_state_label, explorer_random_state_tip)
        ToolTip(explorer_random_state_entry, explorer_random_state_tip)
        backtest_workers_label = ttk.Label(controls, text="Backtest workers")
        backtest_workers_label.grid(row=3, column=0, sticky="w", padx=8, pady=(0, 8))
        backtest_workers_entry = ttk.Entry(controls, textvariable=self.explorer_backtest_workers_var, width=10)
        backtest_workers_entry.grid(row=3, column=1, sticky="ew", padx=8, pady=(0, 8))
        backtest_workers_tip = "Parallel validation backtests. 1 runs serially. 12 is a good starting point for a 20-core machine, but reduce it if RAM or disk usage spikes."
        ToolTip(backtest_workers_label, backtest_workers_tip)
        ToolTip(backtest_workers_entry, backtest_workers_tip)
        explorer_loss_label = ttk.Label(controls, text="Hyperopt loss")
        explorer_loss_label.grid(row=3, column=2, sticky="w", padx=8, pady=(0, 8))
        explorer_loss_picker = ttk.Frame(controls)
        explorer_loss_picker.grid(row=3, column=3, columnspan=5, sticky="ew", padx=8, pady=(0, 8))
        explorer_loss_picker.grid_columnconfigure(0, weight=1)
        loss_tip = (
            "Chooses the loss function that scores each Explorer hyperopt epoch. Custom losses are "
            "discovered from freqtrade/optimize/hyperopt_loss and user_data/hyperopts."
        )
        self.explorer_hyperopt_loss_combo = ttk.Combobox(
            explorer_loss_picker,
            textvariable=self.hyperopt_loss_var,
            values=HYPEROPT_LOSS_VALUES,
            width=32,
        )
        self.explorer_hyperopt_loss_combo.grid(row=0, column=0, sticky="ew")
        self.explorer_hyperopt_loss_combo.bind("<<ComboboxSelected>>", lambda _event: self.show_hyperopt_loss_popup())
        self.explorer_hyperopt_loss_combo.bind("<Double-Button-1>", lambda _event: self.show_hyperopt_loss_popup())
        ttk.Button(explorer_loss_picker, text="Details", command=self.show_hyperopt_loss_popup).grid(row=0, column=1, sticky="ew", padx=(6, 0))
        ttk.Button(explorer_loss_picker, text="Refresh", command=lambda: self.refresh_hyperopt_loss_dropdown(force=True)).grid(row=0, column=2, sticky="ew", padx=(6, 0))
        ToolTip(explorer_loss_label, loss_tip)
        ToolTip(explorer_loss_picker, loss_tip)
        ToolTip(self.explorer_hyperopt_loss_combo, loss_tip)
        min_params_label = ttk.Label(controls, text="Min params/run")
        min_params_label.grid(row=2, column=4, sticky="w", padx=8, pady=(0, 8))
        self.explorer_min_params_label_widget = min_params_label
        min_params_entry = ttk.Entry(controls, textvariable=self.explorer_min_param_count_var, width=10)
        min_params_entry.grid(row=2, column=5, sticky="ew", padx=8, pady=(0, 8))
        self.explorer_min_params_entry_widget = min_params_entry
        ToolTip(
            min_params_entry,
            "Pads small targets with related parameters until at least this many tunable parameters are included. "
            "0 disables padding. Leave blank to omit --min-param-count-per-hyper-run.",
        )
        explorer_jobs_label = ttk.Label(controls, text="Job workers")
        explorer_jobs_label.grid(row=2, column=6, sticky="w", padx=8, pady=(0, 8))
        explorer_jobs_entry = ttk.Entry(controls, textvariable=self.hyperopt_jobs_var, width=10)
        explorer_jobs_entry.grid(row=2, column=7, sticky="ew", padx=8, pady=(0, 8))
        ToolTip(
            explorer_jobs_label,
            "Context: Hyperopt parallel worker setting shared with Mode tab. Outcome: Sets -j for each Explorer hyperopt run; blank omits -j. "
            "Example: Use 1 for one core, or 20 for twenty cores.",
        )
        ToolTip(
            explorer_jobs_entry,
            "Context: Hyperopt parallel worker setting shared with Mode tab. Outcome: Sets -j for each Explorer hyperopt run; blank omits -j. "
            "Example: Set 12 on a 20-core machine if RAM allows.",
        )

        keeper_frame = ttk.LabelFrame(root, text="2. Keeper snapshots")
        keeper_frame.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))
        keeper_frame.grid_columnconfigure(1, weight=1)
        keeper_enabled = ttk.Checkbutton(keeper_frame, text="Keeper snapshots enabled", variable=self.explorer_keeper_enabled_var)
        keeper_enabled.grid(row=0, column=0, columnspan=3, sticky="w", padx=8, pady=8)
        ToolTip(keeper_enabled, "When enabled, Explorer saves accepted merged strategy-parameter snapshots only if keeper thresholds pass. This never changes acceptance scoring; it only controls archive copies.")
        self._path_row(keeper_frame, 1, "Explorer keeper save directory", self.explorer_keeper_save_dir_var, directory=True)
        ttk.Label(keeper_frame, text="Keeper win ratio").grid(row=2, column=0, sticky="w", padx=8, pady=(0, 8))
        ratio_row = ttk.Frame(keeper_frame)
        ratio_row.grid(row=2, column=1, sticky="w", padx=8, pady=(0, 8))
        keeper_num_entry = ttk.Entry(ratio_row, textvariable=self.explorer_keeper_win_numerator_var, width=6)
        keeper_num_entry.pack(side="left")
        ttk.Label(ratio_row, text="/").pack(side="left", padx=6)
        keeper_den_entry = ttk.Entry(ratio_row, textvariable=self.explorer_keeper_win_denominator_var, width=6)
        keeper_den_entry.pack(side="left")
        ttk.Label(keeper_frame, text="Keeper min profit per backtest window").grid(row=2, column=2, sticky="w", padx=8, pady=(0, 8))
        keeper_profit_entry = ttk.Entry(keeper_frame, textvariable=self.explorer_keeper_min_profit_per_window_var, width=12)
        keeper_profit_entry.grid(row=2, column=3, sticky="w", padx=8, pady=(0, 8))
        ToolTip(
            keeper_num_entry,
            "Numerator for keeper winning-window requirement. Required wins are ceil(total_windows * numerator / denominator). "
            "Leave blank to use 5.",
        )
        ToolTip(
            keeper_den_entry,
            "Denominator for keeper winning-window requirement. Must be greater than zero. Leave blank to use 6.",
        )
        ToolTip(
            keeper_profit_entry,
            "Minimum required total challenger profit is total_windows * this value. Leave blank to use 200.",
        )

        windows_frame = ttk.LabelFrame(root, text="3. Selected market windows")
        windows_frame.grid(row=3, column=0, sticky="ew", padx=8, pady=(0, 8))
        windows_frame.grid_columnconfigure(0, weight=1)
        ToolTip(
            windows_frame,
            "Context: Candidate market-window pool for each Explorer loop. Outcome: Runner samples one eligible window for Hyperopt and uses remaining eligible windows for backtest validation. "
            "Example: Enable bull+chop windows to test robustness across both regimes.",
        )

        window_actions = ttk.Frame(windows_frame)
        window_actions.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        ttk.Button(window_actions, text="Enable all", command=lambda: self.set_explorer_windows(True)).pack(side="left", padx=(0, 8))
        ttk.Button(window_actions, text="Disable all", command=lambda: self.set_explorer_windows(False)).pack(side="left", padx=(0, 8))
        ttk.Button(window_actions, text="Only bull", command=lambda: self.set_explorer_windows_by_regime("bull")).pack(side="left", padx=(0, 8))
        ttk.Button(window_actions, text="Only bear", command=lambda: self.set_explorer_windows_by_regime("bear")).pack(side="left", padx=(0, 8))
        ttk.Button(window_actions, text="Only chop", command=lambda: self.set_explorer_windows_by_regime("chop")).pack(side="left", padx=(0, 8))
        ttk.Button(window_actions, text="Only crash", command=lambda: self.set_explorer_windows_by_regime("crash")).pack(side="left", padx=(0, 8))
        ttk.Button(window_actions, text="Only crossover", command=lambda: self.set_explorer_windows_by_regime("crossover")).pack(side="left", padx=(0, 8))
        window_grid = ttk.Frame(windows_frame)
        window_grid.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        for col in range(3):
            window_grid.grid_columnconfigure(col, weight=1)
        for index, window in enumerate(self.load_explorer_market_windows()):
            name = str(window.get("name", ""))
            regime = str(window.get("regime", ""))
            timerange = str(window.get("timerange", ""))
            if not name:
                continue
            var = self.explorer_window_vars.setdefault(name, tk.BooleanVar(value=True))
            label = f"{name} | {regime} | {timerange}"
            check = ttk.Checkbutton(window_grid, text=label, variable=var)
            check.grid(row=index // 3, column=index % 3, sticky="w", padx=8, pady=2)
            ToolTip(
                check,
                "Context: Per-window eligibility toggle for Explorer sampling. Outcome: Checked windows can be sampled for Hyperopt/backtest split; unchecked windows are excluded. "
                "Example: Disable crash windows for a bull-only experiment.",
            )

        selection_row = ttk.Frame(root)
        selection_row.grid(row=4, column=0, sticky="nsew", padx=8, pady=(0, 8))
        selection_row.grid_columnconfigure(0, weight=3)
        selection_row.grid_columnconfigure(1, weight=2)

        filters_frame = ttk.LabelFrame(selection_row, text="4. Backtest selection")
        filters_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=0)
        filters_frame.grid_columnconfigure(0, weight=1)
        filters_frame.grid_columnconfigure(1, weight=1)
        filters_frame.grid_columnconfigure(2, weight=1)
        self._build_market_type_filter_box(filters_frame, 0, "HyperOpt usable market types", "hyperopt")
        self._build_market_type_filter_box(filters_frame, 1, "Backtest usable market types", "backtest")
        self._build_backtest_count_box(filters_frame, 2)

        holdouts = ttk.LabelFrame(selection_row, text="5. 12-month holdout backtests")
        holdouts.grid(row=0, column=1, sticky="nsew", padx=(6, 0), pady=0)
        for col in range(2):
            holdouts.grid_columnconfigure(col, weight=1)
        holdout_tip = "Select at least one fixed 12-month holdout. Holdout windows are backtest-only and are never used for HyperOpt."
        ToolTip(holdouts, holdout_tip)
        for index, window in enumerate(EXPLORER_12M_HOLDOUT_WINDOWS):
            label = str(window["label"])
            var = self.explorer_12m_holdout_vars[label]
            check = ttk.Checkbutton(holdouts, text=label, variable=var)
            check.grid(row=index // 2, column=index % 2, sticky="w", padx=8, pady=3)
            ToolTip(check, str(window.get("tooltip") or holdout_tip))

        self._build_catalog_custom_batches_tab(catalog_tab)
        self._refresh_explorer_namespace_dropdown()

    def _build_catalog_custom_batches_tab(self, root: ttk.Frame) -> None:
        # TODO(custom batches):
        # This tab is the seed of a future workflow for deliberate hyperopt runs:
        # group params by tag/namespace, queue them as reusable custom batches,
        # then pair each batch with custom market windows and a custom loss
        # objective for the run. The intent is to support curated flows such as
        # "seed-entry enables across crash/bear windows", "same batch across bull
        # windows", or "entry/exit sizing batches with different objectives",
        # and to store or compare the results without relying on Explorer's
        # broader semi-random family/tag sampling. Keep the UI and resolver logic
        # aligned with that direction when extending this area.
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(1, weight=1)

        toolbar = ttk.Frame(root)
        toolbar.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        toolbar.grid_columnconfigure(3, weight=1)
        view_label = ttk.Label(toolbar, text="View")
        view_label.grid(row=0, column=0, sticky="w", padx=(0, 6))
        view_combo = ttk.Combobox(toolbar, textvariable=self.catalog_view_var, values=["By Family", "By Namespace", "Modes", "Switches", "Parameters A-Z"], state="readonly", width=18)
        view_combo.grid(row=0, column=1, sticky="w", padx=(0, 10))
        search_label = ttk.Label(toolbar, text="Search")
        search_label.grid(row=0, column=2, sticky="w", padx=(0, 6))
        search_entry = ttk.Entry(toolbar, textvariable=self.catalog_search_var)
        search_entry.grid(row=0, column=3, sticky="ew", padx=(0, 10))
        filter_combo = ttk.Combobox(toolbar, textvariable=self.catalog_filter_var, values=["Show all", "Buy space only", "Sell space only", "Boolean/switch params only", "Feature enables only", "Changed/current params"], state="readonly", width=24)
        filter_combo.grid(row=0, column=4, sticky="w", padx=(0, 10))
        ttk.Button(toolbar, text="Refresh catalog", command=self.refresh_custom_batch_catalog).grid(row=0, column=5, sticky="e")
        ToolTip(
            view_label,
            "Context: Catalog organization control. Outcome: Regroups left tree by family/namespace/mode view. "
            "Example: Switch to 'By Namespace' to inspect switch:* parameters together.",
        )
        ToolTip(
            view_combo,
            "Context: Catalog organization control. Outcome: Regroups left tree by family/namespace/mode view. "
            "Example: Use 'Parameters A-Z' for direct name lookup.",
        )
        ToolTip(
            search_label,
            "Context: Text filter over catalog content. Outcome: Narrows nodes by label/param/tag terms; blank shows all entries. "
            "Example: Search `stoploss` or `switch:enable`.",
        )
        ToolTip(
            search_entry,
            "Context: Text filter over catalog content. Outcome: Narrows nodes by label/param/tag terms; blank shows all entries. "
            "Example: Search `ema` to find EMA-related params.",
        )
        ToolTip(
            filter_combo,
            "Context: Structured filter layer after text search. Outcome: Restricts visible params by semantic category; 'Show all' applies no extra filter. "
            "Example: Choose 'Buy space only' when building a buy-focused batch.",
        )

        pane_row = tk.PanedWindow(
            root,
            orient=tk.HORIZONTAL,
            sashwidth=7,
            sashpad=2,
            opaqueresize=True,
            bd=0,
            relief="flat",
        )
        pane_row.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))

        left_host = ttk.Frame(pane_row, width=500)
        middle_host = ttk.Frame(pane_row, width=390)
        right_host = ttk.Frame(pane_row, width=450)
        pane_row.add(left_host, minsize=340, stretch="always")
        pane_row.add(middle_host, minsize=300, stretch="always")
        pane_row.add(right_host, minsize=340, stretch="always")

        left = ttk.LabelFrame(left_host, text="Parameter catalog tree")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 4), pady=0)
        left_host.grid_columnconfigure(0, weight=1)
        left_host.grid_rowconfigure(0, weight=1)
        left.grid_columnconfigure(0, weight=1)
        left.grid_rowconfigure(0, weight=1)
        self.catalog_tree = ttk.Treeview(left, show="tree")
        self.catalog_tree.grid(row=0, column=0, sticky="nsew")
        catalog_y = ttk.Scrollbar(left, orient="vertical", command=self.catalog_tree.yview)
        catalog_y.grid(row=0, column=1, sticky="ns")
        self.catalog_tree.configure(yscrollcommand=catalog_y.set)
        self.catalog_tree.bind("<<TreeviewSelect>>", lambda _event: self.on_catalog_tree_select())
        self.catalog_tree.bind("<Double-1>", lambda _event: self.add_selected_catalog_node_to_draft())

        middle = ttk.LabelFrame(middle_host, text="Selected item details")
        middle.grid(row=0, column=0, sticky="nsew", padx=4, pady=0)
        middle_host.grid_columnconfigure(0, weight=1)
        middle_host.grid_rowconfigure(0, weight=1)
        middle.grid_columnconfigure(0, weight=1)
        middle.grid_rowconfigure(0, weight=1)
        self.catalog_details_text = scrolledtext.ScrolledText(middle, wrap="word", height=18)
        self.catalog_details_text.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
        self.catalog_details_text.configure(state="disabled")

        right = ttk.LabelFrame(right_host, text="Custom batch builder")
        right.grid(row=0, column=0, sticky="nsew", padx=(4, 0), pady=0)
        right_host.grid_columnconfigure(0, weight=1)
        right_host.grid_rowconfigure(0, weight=1)
        right.grid_columnconfigure(0, weight=1)
        right.grid_rowconfigure(7, weight=1)
        right.grid_rowconfigure(11, weight=1)

        form = ttk.Frame(right)
        form.grid(row=0, column=0, sticky="ew", padx=6, pady=(6, 4))
        form.grid_columnconfigure(1, weight=1)
        batch_name_label = ttk.Label(form, text="Batch name")
        batch_name_label.grid(row=0, column=0, sticky="w", padx=(0, 6), pady=2)
        batch_name_entry = ttk.Entry(form, textvariable=self.custom_batch_name_var)
        batch_name_entry.grid(row=0, column=1, sticky="ew", pady=2)
        batch_desc_label = ttk.Label(form, text="Description")
        batch_desc_label.grid(row=1, column=0, sticky="w", padx=(0, 6), pady=2)
        batch_desc_entry = ttk.Entry(form, textvariable=self.custom_batch_description_var)
        batch_desc_entry.grid(row=1, column=1, sticky="ew", pady=2)
        ToolTip(
            batch_name_label,
            "Context: Human-readable identifier for the draft batch. Outcome: Required for Commit/Save; blank names are rejected. "
            "Example: `entry_switches_bear_test`.",
        )
        ToolTip(
            batch_name_entry,
            "Context: Human-readable identifier for the draft batch. Outcome: Required for Commit/Save; blank names are rejected. "
            "Example: `roi_stoploss_mix_v2`.",
        )
        ToolTip(
            batch_desc_label,
            "Context: Optional notes describing batch intent. Outcome: Stored with the batch metadata; blank stores an empty description. "
            "Example: `Focus on drawdown-safe buy enables`.",
        )
        ToolTip(
            batch_desc_entry,
            "Context: Optional notes describing batch intent. Outcome: Stored with the batch metadata; blank stores an empty description. "
            "Example: Add purpose and market assumptions for teammates.",
        )

        action_row = ttk.Frame(right)
        action_row.grid(row=1, column=0, sticky="ew", padx=6, pady=2)
        for label, command in (
            ("Add selected", self.add_selected_catalog_node_to_draft),
            ("Remove selected source", self.remove_selected_custom_batch_source),
            ("Exclude selected param", self.exclude_selected_custom_batch_param),
            ("Re-include selected excluded param", self.reinclude_selected_custom_batch_param),
        ):
            ttk.Button(action_row, text=label, command=command).pack(side="left", padx=(0, 4))

        ttk.Label(right, text="Sources").grid(row=2, column=0, sticky="w", padx=6, pady=(6, 0))
        self.custom_batch_sources_list = tk.Listbox(right, height=5, exportselection=False)
        self.custom_batch_sources_list.grid(row=3, column=0, sticky="ew", padx=6, pady=(2, 4))
        ttk.Label(right, text="Excluded params").grid(row=4, column=0, sticky="w", padx=6, pady=(4, 0))
        self.custom_batch_excluded_list = tk.Listbox(right, height=4, exportselection=False)
        self.custom_batch_excluded_list.grid(row=5, column=0, sticky="ew", padx=6, pady=(2, 4))
        ttk.Label(right, text="Resolved params").grid(row=6, column=0, sticky="w", padx=6, pady=(4, 0))
        self.catalog_resolved_tree = ttk.Treeview(right, columns=("name", "type", "space", "current", "default"), show="headings", height=8)
        for column, title, width in (("name", "Name", 150), ("type", "Type", 110), ("space", "Space", 55), ("current", "Current", 70), ("default", "Default", 70)):
            self.catalog_resolved_tree.heading(column, text=title)
            self.catalog_resolved_tree.column(column, width=width, stretch=column == "name")
        self.catalog_resolved_tree["displaycolumns"] = ("name", "type", "space", "current", "default")
        self.catalog_resolved_tree.grid(row=7, column=0, sticky="nsew", padx=6, pady=(2, 4))

        draft_buttons = ttk.Frame(right)
        draft_buttons.grid(row=8, column=0, sticky="ew", padx=6, pady=2)
        for label, command in (
            ("Clear draft", self.clear_custom_batch_draft),
            ("Commit batch", self.commit_custom_batch),
            ("Start new batch", self.start_new_custom_batch),
            ("Save changes", self.save_custom_batch_changes),
        ):
            ttk.Button(draft_buttons, text=label, command=command).pack(side="left", padx=(0, 4))

        # Committed batches are intentionally the durable handoff point for the
        # future workflow above: batches should be easy to revisit, compare,
        # store, and re-run against distinct market windows / loss objectives.
        ttk.Label(right, text="Committed batches ([x] runs in Custom batches mode)").grid(row=9, column=0, sticky="w", padx=6, pady=(8, 0))
        self.custom_batch_committed_tree = ttk.Treeview(right, columns=("run", "name", "sources", "params", "spaces", "status"), show="headings", height=6)
        for column, title, width in (
            ("run", "Run", 48),
            ("name", "Name", 170),
            ("sources", "Sources", 62),
            ("params", "Params", 58),
            ("spaces", "Spaces", 90),
            ("status", "Status", 95),
        ):
            self.custom_batch_committed_tree.heading(column, text=title)
            self.custom_batch_committed_tree.column(column, width=width, stretch=column == "name")
        self.custom_batch_committed_tree.grid(row=10, column=0, sticky="ew", padx=6, pady=(2, 4))
        self.custom_batch_committed_tree.bind("<ButtonRelease-1>", lambda event: self.on_committed_batch_click(event))

        committed_buttons = ttk.Frame(right)
        committed_buttons.grid(row=11, column=0, sticky="sew", padx=6, pady=(2, 4))
        for label, command in (
            ("Load committed batch for editing", self.load_selected_committed_batch_for_editing),
            ("Duplicate selected committed batch", self.duplicate_selected_committed_batch),
            ("Delete selected committed batch", self.delete_selected_committed_batch),
        ):
            ttk.Button(committed_buttons, text=label, command=command).pack(side="left", padx=(0, 4))
        ttk.Label(right, textvariable=self.custom_batch_status_var, foreground="#7A5A00", wraplength=360, justify="left").grid(row=12, column=0, sticky="ew", padx=6, pady=(0, 6))

        self.catalog_view_var.trace_add("write", lambda *_args: self.refresh_custom_batch_catalog())
        self.catalog_search_var.trace_add("write", lambda *_args: self.refresh_custom_batch_catalog())
        self.catalog_filter_var.trace_add("write", lambda *_args: self.refresh_custom_batch_catalog())
        self.custom_batch_name_var.trace_add("write", lambda *_args: self.refresh_custom_batch_draft())
        self.custom_batch_description_var.trace_add("write", lambda *_args: self.refresh_custom_batch_draft())
        self.refresh_custom_batch_catalog()

    def _custom_batches_path(self) -> Path:
        return app_path(EXPLORER_CUSTOM_BATCH_FILE)

    def _load_catalog_data(self) -> dict[str, Any]:
        strategy_file = self.strategy_file_var.get().strip()
        strategy_class = self.strategy_class_var.get().strip() or "HybridRecoveryGridStrategy"
        if not strategy_file:
            return {}
        from hyperopt_tag_catalog import build_param_catalog

        return build_param_catalog(strategy_file, strategy_class)

    def _load_current_strategy_values(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        strategy_path = Path(self.strategy_file_var.get().strip() or str(app_path(DEFAULT_STRATEGY_FILE)))
        if not strategy_path.is_absolute():
            strategy_path = app_path(strategy_path)
        path = strategy_path.with_suffix(".json")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return values
        for space, params in (data.get("params") or {}).items():
            if isinstance(params, dict):
                for name, value in params.items():
                    values[str(name)] = value
        return values

    def _catalog_strategy_spaces(self) -> dict[str, str]:
        return {
            str(name): str((info or {}).get("space") or "")
            for name, info in (self.catalog_data.get("params") or {}).items()
            if str((info or {}).get("space") or "")
        }

    def refresh_custom_batch_catalog(self) -> None:
        try:
            from hyperopt_explorer_support import load_custom_batches

            self.catalog_data = self._load_catalog_data()
            self.catalog_current_values = self._load_current_strategy_values()
            self.custom_batches_payload = load_custom_batches(self._custom_batches_path())
            self._populate_catalog_tree()
            self.refresh_custom_batch_draft()
            self.refresh_committed_custom_batches()
        except Exception as exc:
            self.custom_batch_status_var.set(f"Catalog unavailable: {exc}")

    def _param_matches_catalog_filters(self, name: str) -> bool:
        info = (self.catalog_data.get("params") or {}).get(name) or {}
        tags = {str(tag) for tag in info.get("tags") or []}
        filter_value = self.catalog_filter_var.get()
        if filter_value == "Buy space only" and str(info.get("space") or "") != "buy":
            return False
        if filter_value == "Sell space only" and str(info.get("space") or "") != "sell":
            return False
        if filter_value == "Boolean/switch params only" and info.get("parameter_type") != "BooleanParameter" and "switch:enable" not in tags:
            return False
        if filter_value == "Feature enables only" and "family:feature_enables" not in tags and "switch:enable" not in tags:
            return False
        if filter_value == "Changed/current params":
            current = self.catalog_current_values.get(name)
            if current is None or current == info.get("default"):
                return False
        return True

    def _catalog_search_hit(self, label: str, params: Iterable[str] = (), tags: Iterable[str] = ()) -> bool:
        needle = self.catalog_search_var.get().strip().lower()
        if not needle:
            return True
        haystack = [label, *[str(param) for param in params], *[str(tag) for tag in tags]]
        return any(needle in item.lower() for item in haystack)

    def _filtered_params(self, params: Iterable[str]) -> list[str]:
        return sorted(name for name in {str(param) for param in params if str(param)} if self._param_matches_catalog_filters(name))

    def _tree_insert(self, parent: str, text: str, data: dict[str, Any], open_node: bool = False) -> str:
        assert self.catalog_tree is not None
        item = self.catalog_tree.insert(parent, "end", text=text, open=open_node)
        self.catalog_tree_item_data[item] = data
        return item

    def _populate_catalog_tree(self) -> None:
        if self.catalog_tree is None:
            return
        self.catalog_tree.delete(*self.catalog_tree.get_children())
        self.catalog_tree_item_data.clear()
        catalog = self.catalog_data
        params = catalog.get("params") or {}
        tag_index = catalog.get("tag_index") or {}
        family_index = catalog.get("family_index") or {}
        namespaces = catalog.get("namespaces") or {}
        view = self.catalog_view_var.get()

        if view in {"By Namespace", "Modes", "Switches"}:
            if view == "Modes":
                namespace_iterable = ["mode"] if "mode" in namespaces else []
            elif view == "Switches":
                namespace_iterable = ["switch"] if "switch" in namespaces else []
            else:
                namespace_iterable = self._ordered_catalog_namespaces(catalog)
            for namespace in namespace_iterable:
                ns_params: set[str] = set()
                for value in namespaces.get(namespace, []):
                    ns_params.update(tag_index.get(f"{namespace}:{value}", []))
                ns_params_list = self._filtered_params(ns_params)
                if not ns_params_list or not self._catalog_search_hit(namespace, ns_params_list, [namespace]):
                    continue
                ns_item = self._tree_insert("", f"namespace:{namespace} [{len(ns_params_list)} params]", {"type": "namespace", "id": namespace, "params": ns_params_list}, True)
                for value in sorted(namespaces.get(namespace, [])):
                    tag = f"{namespace}:{value}"
                    tag_params = self._filtered_params(tag_index.get(tag, []))
                    if not tag_params or not self._catalog_search_hit(tag, tag_params, [tag]):
                        continue
                    tag_item = self._tree_insert(ns_item, f"{tag} [{len(tag_params)} params]", {"type": "namespace_value", "id": value, "namespace": namespace, "value": value, "tag": tag, "params": tag_params})
                    for param in tag_params:
                        self._tree_insert(tag_item, f"param:{param}", {"type": "param", "id": param, "params": [param]})
            return

        if view == "Parameters A-Z":
            for param in self._filtered_params(params.keys()):
                info = params.get(param) or {}
                tags = [str(tag) for tag in info.get("tags") or []]
                if not self._catalog_search_hit(param, [param], tags):
                    continue
                label = f"{param} [{info.get('parameter_type') or '-'} | {info.get('space') or '-'}]"
                param_item = self._tree_insert("", label, {"type": "param", "id": param, "params": [param]}, False)
                for tag in sorted(tags, key=lambda item: (item.split(":", 1)[0], item)):
                    self._tree_insert(param_item, tag, {"type": "tag", "id": tag, "params": [param]})
            return

        for family in sorted(family_index):
            family_params = self._filtered_params(family_index.get(family, []))
            if not family_params or not self._catalog_search_hit(family, family_params, [f"family:{family}"]):
                continue
            family_item = self._tree_insert("", f"family:{family} [{len(family_params)} params]", {"type": "family", "id": family, "params": family_params}, True)
            for namespace in self._ordered_catalog_namespaces(catalog):
                ns_tags = [f"{namespace}:{value}" for value in namespaces.get(namespace, [])]
                ns_params = sorted({param for tag in ns_tags for param in tag_index.get(tag, []) if param in family_params})
                if not ns_params:
                    continue
                ns_item = self._tree_insert(family_item, f"{namespace} [{len(ns_params)} params]", {"type": "namespace", "id": namespace, "context": {"family": family, "intersection": True}, "params": ns_params})
                for tag in ns_tags:
                    intersect = sorted(set(tag_index.get(tag, [])) & set(family_params))
                    intersect = self._filtered_params(intersect)
                    if not intersect:
                        continue
                    tag_item = self._tree_insert(ns_item, f"{tag} [{len(intersect)} params]", {"type": "family_tag_intersection", "id": f"{family}|{tag}", "family": family, "tag": tag, "context": {"family": family}, "params": intersect})
                    for param in intersect:
                        self._tree_insert(tag_item, param, {"type": "param", "id": param, "params": [param]})
            params_item = self._tree_insert(family_item, f"params [{len(family_params)} params]", {"type": "family", "id": family, "params": family_params})
            for param in family_params:
                self._tree_insert(params_item, param, {"type": "param", "id": param, "params": [param]})

    def _selected_catalog_data(self) -> dict[str, Any] | None:
        if self.catalog_tree is None:
            return None
        selected = self.catalog_tree.selection()
        return self.catalog_tree_item_data.get(selected[0]) if selected else None

    def on_catalog_tree_select(self) -> None:
        data = self._selected_catalog_data()
        if not data or self.catalog_details_text is None:
            return
        params = data.get("params") or []
        catalog_params = self.catalog_data.get("params") or {}
        lines = [f"Type: {data.get('type')}", f"Name: {data.get('id')}", f"Parameter count: {len(params)}"]
        if data.get("context"):
            lines.append(f"Context: {data.get('context')}")
            if data.get("type") == "namespace":
                lines.append("Adding this family namespace node expands to family/tag intersections for the visible namespace values.")
            else:
                lines.append("This tag is resolved as a family/tag intersection.")
        if data.get("type") == "namespace_value":
            lines.append(f"Source saved as: namespace_value {data.get('namespace')}:{data.get('value')}")
        if data.get("type") == "family_tag_intersection":
            lines.append(f"Source saved as: family_tag_intersection family:{data.get('family')} + tag:{data.get('tag')}")
        spaces = sorted({str((catalog_params.get(param) or {}).get("space") or "") for param in params if str((catalog_params.get(param) or {}).get("space") or "")})
        if spaces:
            lines.append(f"Spaces used: {', '.join(spaces)}")
        if data.get("type") == "family":
            lines.append("Source: Strategy tags (family namespace)")
        if data.get("type") == "param" and params:
            info = catalog_params.get(params[0]) or {}
            lines.extend(
                [
                    f"Parameter type: {info.get('parameter_type') or '-'}",
                    f"Space: {info.get('space') or '-'}",
                    f"Default value: {info.get('default')!r}",
                    f"Current value: {self.catalog_current_values.get(params[0], '-')!r}",
                    f"Optimize: {info.get('optimize')}",
                    f"Load: {info.get('load')}",
                    "All tags:",
                    *[f"  - {tag}" for tag in info.get("tags") or []],
                ]
            )
        else:
            lines.append("Resolved params:")
            for param in params[:200]:
                info = catalog_params.get(param) or {}
                lines.append(f"  - {param} [{info.get('parameter_type') or '-'} | {info.get('space') or '-'}]")
            if len(params) > 200:
                lines.append(f"  ... {len(params) - 200} more")
        self.catalog_details_text.configure(state="normal")
        self.catalog_details_text.delete("1.0", tk.END)
        self.catalog_details_text.insert("1.0", "\n".join(lines))
        self.catalog_details_text.configure(state="disabled")

    def _draft_batch(self) -> dict[str, Any]:
        return {
            "id": self.custom_batch_editing_id or "",
            "name": self.custom_batch_name_var.get().strip(),
            "description": self.custom_batch_description_var.get().strip(),
            "sources": deepcopy(self.custom_batch_draft_sources),
            "excluded_params": list(self.custom_batch_draft_excluded),
        }

    def _resolve_draft_batch(self) -> dict[str, Any]:
        from hyperopt_explorer_support import resolve_custom_batch

        return resolve_custom_batch(self._draft_batch(), self.catalog_data, self._catalog_strategy_spaces())

    def refresh_custom_batch_draft(self) -> None:
        if self.custom_batch_sources_list is not None:
            self.custom_batch_sources_list.delete(0, tk.END)
            from hyperopt_explorer_support import custom_batch_source_label

            for source in self.custom_batch_draft_sources:
                self.custom_batch_sources_list.insert(tk.END, custom_batch_source_label(source))
        if self.custom_batch_excluded_list is not None:
            self.custom_batch_excluded_list.delete(0, tk.END)
            for param in self.custom_batch_draft_excluded:
                self.custom_batch_excluded_list.insert(tk.END, param)
        if self.catalog_resolved_tree is not None:
            self.catalog_resolved_tree.delete(*self.catalog_resolved_tree.get_children())
            resolution = self._resolve_draft_batch() if self.catalog_data else {"params": [], "spaces": [], "stale_sources": [], "stale_excluded_params": []}
            catalog_params = self.catalog_data.get("params") or {}
            for param in resolution.get("params") or []:
                info = catalog_params.get(param) or {}
                self.catalog_resolved_tree.insert("", "end", values=(param, info.get("parameter_type") or "-", info.get("space") or "-", self.catalog_current_values.get(param, "-"), info.get("default")))
            warnings = []
            if resolution.get("stale_sources"):
                warnings.append(f"Stale sources: {len(resolution['stale_sources'])}")
            if resolution.get("stale_excluded_params"):
                warnings.append("Stale excluded params: " + ", ".join(resolution["stale_excluded_params"]))
            summary = f"Resolved {len(resolution.get('params') or [])} params | Spaces: {', '.join(resolution.get('spaces') or []) or '-'}"
            self.custom_batch_status_var.set(summary + ((" | " + " | ".join(warnings)) if warnings else ""))

    def _sources_from_catalog_data(self, data: dict[str, Any]) -> list[dict[str, Any]]:
        source_type = str(data.get("type") or "")
        source_id = str(data.get("id") or "")
        if source_type not in {"family", "namespace", "namespace_value", "family_tag_intersection", "tag", "param"} or not source_id:
            return []
        if source_type == "namespace_value":
            namespace = str(data.get("namespace") or "")
            value = str(data.get("value") or source_id)
            if not namespace or not value:
                return []
            return [{"type": "namespace_value", "id": value, "namespace": namespace, "value": value}]
        if source_type == "family_tag_intersection":
            family = str(data.get("family") or "")
            tag = str(data.get("tag") or "")
            if not family or not tag:
                return []
            return [{"type": "family_tag_intersection", "id": source_id, "family": family, "tag": tag}]
        if source_type == "namespace" and isinstance(data.get("context"), dict) and data["context"].get("family"):
            family = str(data["context"].get("family") or "")
            namespace = source_id
            visible_params = set(data.get("params") or self.catalog_data.get("family_index", {}).get(family, []))
            tags = [
                f"{namespace}:{value}"
                for value in (self.catalog_data.get("namespaces") or {}).get(namespace, [])
                if visible_params & set((self.catalog_data.get("tag_index") or {}).get(f"{namespace}:{value}", []))
            ]
            return [
                {"type": "family_tag_intersection", "id": f"{family}|{tag}", "family": family, "tag": tag}
                for tag in tags
            ]
        source: dict[str, Any] = {"type": source_type, "id": source_id, "context": data.get("context") if isinstance(data.get("context"), dict) else None}
        if source_type == "namespace" and source.get("context"):
            source["context"] = None
        return [source]

    def _source_from_catalog_data(self, data: dict[str, Any]) -> dict[str, Any] | None:
        sources = self._sources_from_catalog_data(data)
        return sources[0] if sources else None

    def add_selected_catalog_node_to_draft(self) -> None:
        sources = self._sources_from_catalog_data(self._selected_catalog_data() or {})
        if not sources:
            return
        existing = {json.dumps(item, sort_keys=True) for item in self.custom_batch_draft_sources}
        added = 0
        for source in sources:
            key = json.dumps(source, sort_keys=True)
            if key not in existing:
                self.custom_batch_draft_sources.append(source)
                existing.add(key)
                added += 1
        if added > 1:
            self.custom_batch_status_var.set(f"Added {added} family/tag intersections from the selected family namespace node.")
        self.refresh_custom_batch_draft()

    def remove_selected_custom_batch_source(self) -> None:
        if self.custom_batch_sources_list is None:
            return
        selection = self.custom_batch_sources_list.curselection()
        if selection:
            del self.custom_batch_draft_sources[selection[0]]
        self.refresh_custom_batch_draft()

    def exclude_selected_custom_batch_param(self) -> None:
        if self.catalog_resolved_tree is None:
            return
        selection = self.catalog_resolved_tree.selection()
        if not selection:
            return
        param = str(self.catalog_resolved_tree.item(selection[0], "values")[0])
        if param and param not in self.custom_batch_draft_excluded:
            self.custom_batch_draft_excluded.append(param)
        self.refresh_custom_batch_draft()

    def reinclude_selected_custom_batch_param(self) -> None:
        if self.custom_batch_excluded_list is None:
            return
        selection = self.custom_batch_excluded_list.curselection()
        if selection:
            del self.custom_batch_draft_excluded[selection[0]]
        self.refresh_custom_batch_draft()

    def clear_custom_batch_draft(self) -> None:
        self.custom_batch_draft_sources = []
        self.custom_batch_draft_excluded = []
        self.refresh_custom_batch_draft()

    def start_new_custom_batch(self) -> None:
        self.custom_batch_editing_id = None
        self.custom_batch_name_var.set("")
        self.custom_batch_description_var.set("")
        self.clear_custom_batch_draft()

    def _persist_custom_batch_payload(self) -> None:
        from hyperopt_explorer_support import save_custom_batches

        save_custom_batches(self._custom_batches_path(), self.custom_batches_payload)
        self.refresh_committed_custom_batches()

    def _batch_payload_record(self, batch_id: str | None = None) -> dict[str, Any]:
        resolution = self._resolve_draft_batch()
        now = datetime.now().astimezone().isoformat()
        return {
            "id": batch_id or "",
            "name": self.custom_batch_name_var.get().strip(),
            "description": self.custom_batch_description_var.get().strip(),
            "schema_version": 1,
            "resolution_source": "strategy_catalog",
            "sources": deepcopy(self.custom_batch_draft_sources),
            "excluded_params": list(self.custom_batch_draft_excluded),
            "resolved_params_snapshot": list(resolution.get("params") or []),
            "spaces_snapshot": list(resolution.get("spaces") or []),
            "stale_sources_snapshot": list(resolution.get("stale_sources") or []),
            "stale_params_snapshot": list(resolution.get("stale_params") or []),
            "stale_excluded_params_snapshot": list(resolution.get("stale_excluded_params") or []),
            "resolved_at": now,
            "created_at": now,
            "updated_at": now,
        }

    def commit_custom_batch(self) -> None:
        from hyperopt_explorer_support import unique_batch_id

        name = self.custom_batch_name_var.get().strip()
        resolution = self._resolve_draft_batch()
        if not name:
            messagebox.showerror(APP_TITLE, "Batch name is required.")
            return
        if not resolution.get("params"):
            messagebox.showerror(APP_TITLE, "Batch must resolve at least one current strategy parameter.")
            return
        existing_ids = {str(batch.get("id")) for batch in self.custom_batches_payload.get("batches") or []}
        batch = self._batch_payload_record(unique_batch_id(name, existing_ids))
        self.custom_batches_payload.setdefault("batches", []).append(batch)
        self.custom_batch_selected_run_ids.add(batch["id"])
        self._persist_custom_batch_payload()
        self.custom_batch_editing_id = batch["id"]

    def save_custom_batch_changes(self) -> None:
        if not self.custom_batch_editing_id:
            messagebox.showerror(APP_TITLE, "Load a committed batch before saving changes.")
            return
        resolution = self._resolve_draft_batch()
        if not self.custom_batch_name_var.get().strip() or not resolution.get("params"):
            messagebox.showerror(APP_TITLE, "Batch needs a name and at least one resolved parameter.")
            return
        for batch in self.custom_batches_payload.get("batches") or []:
            if batch.get("id") == self.custom_batch_editing_id:
                created_at = batch.get("created_at")
                batch.update(self._batch_payload_record(self.custom_batch_editing_id))
                batch["created_at"] = created_at or batch["updated_at"]
                break
        self._persist_custom_batch_payload()

    def _selected_committed_batch_id(self) -> str | None:
        if self.custom_batch_committed_tree is None:
            return None
        selected = self.custom_batch_committed_tree.selection()
        return selected[0] if selected else None

    def _batch_by_id(self, batch_id: str | None) -> dict[str, Any] | None:
        for batch in self.custom_batches_payload.get("batches") or []:
            if isinstance(batch, dict) and batch.get("id") == batch_id:
                return batch
        return None

    def load_selected_committed_batch_for_editing(self) -> None:
        batch = self._batch_by_id(self._selected_committed_batch_id())
        if not batch:
            return
        self.custom_batch_editing_id = str(batch.get("id"))
        self.custom_batch_name_var.set(str(batch.get("name") or ""))
        self.custom_batch_description_var.set(str(batch.get("description") or ""))
        self.custom_batch_draft_sources = [deepcopy(source) for source in batch.get("sources") or [] if isinstance(source, dict)]
        self.custom_batch_draft_excluded = [str(param) for param in batch.get("excluded_params") or [] if str(param)]
        self.refresh_custom_batch_draft()

    def duplicate_selected_committed_batch(self) -> None:
        from hyperopt_explorer_support import unique_batch_id

        batch = self._batch_by_id(self._selected_committed_batch_id())
        if not batch:
            return
        existing_ids = {str(item.get("id")) for item in self.custom_batches_payload.get("batches") or []}
        clone = deepcopy(batch)
        clone["name"] = f"{batch.get('name') or 'Custom batch'} copy"
        clone["id"] = unique_batch_id(str(clone["name"]), existing_ids)
        clone["created_at"] = datetime.now().astimezone().isoformat()
        clone["updated_at"] = clone["created_at"]
        self.custom_batches_payload.setdefault("batches", []).append(clone)
        self._persist_custom_batch_payload()

    def delete_selected_committed_batch(self) -> None:
        batch_id = self._selected_committed_batch_id()
        if not batch_id:
            return
        self.custom_batches_payload["batches"] = [batch for batch in self.custom_batches_payload.get("batches") or [] if batch.get("id") != batch_id]
        self.custom_batch_selected_run_ids.discard(batch_id)
        if self.custom_batch_editing_id == batch_id:
            self.start_new_custom_batch()
        self._persist_custom_batch_payload()

    def on_committed_batch_click(self, event: tk.Event) -> None:
        if self.custom_batch_committed_tree is None:
            return
        region = self.custom_batch_committed_tree.identify("region", event.x, event.y)
        column = self.custom_batch_committed_tree.identify_column(event.x)
        item = self.custom_batch_committed_tree.identify_row(event.y)
        if region == "cell" and column == "#1" and item:
            if item in self.custom_batch_selected_run_ids:
                self.custom_batch_selected_run_ids.remove(item)
            else:
                self.custom_batch_selected_run_ids.add(item)
            self.refresh_committed_custom_batches()

    def refresh_committed_custom_batches(self) -> None:
        if self.custom_batch_committed_tree is None:
            return
        from hyperopt_explorer_support import resolve_custom_batch

        existing_ids = {str(batch.get("id")) for batch in self.custom_batches_payload.get("batches") or [] if isinstance(batch, dict)}
        self.custom_batch_selected_run_ids &= existing_ids
        self.custom_batch_committed_tree.delete(*self.custom_batch_committed_tree.get_children())
        spaces = self._catalog_strategy_spaces()
        for batch in self.custom_batches_payload.get("batches") or []:
            if not isinstance(batch, dict):
                continue
            batch_id = str(batch.get("id") or "")
            resolution = resolve_custom_batch(batch, self.catalog_data, spaces) if self.catalog_data else {"params": batch.get("resolved_params_snapshot") or [], "spaces": batch.get("spaces_snapshot") or [], "stale_sources": [], "stale_params": [], "stale_excluded_params": []}
            run_mark = "[x]" if batch_id in self.custom_batch_selected_run_ids else "[ ]"
            stale_count = len(resolution.get("stale_sources") or []) + len(resolution.get("stale_params") or []) + len(resolution.get("stale_excluded_params") or [])
            status = "stale" if stale_count else ("ready" if resolution.get("params") else "empty")
            source_count = len([source for source in batch.get("sources") or [] if isinstance(source, dict)])
            self.custom_batch_committed_tree.insert(
                "",
                "end",
                iid=batch_id,
                values=(
                    run_mark,
                    batch.get("name") or batch_id,
                    source_count,
                    len(resolution.get("params") or []),
                    ", ".join(resolution.get("spaces") or []),
                    status,
                ),
            )

    def _build_review_tab(self) -> None:

        root = self.tab_review
        root.grid_columnconfigure(0, weight=1)
        root.grid_rowconfigure(2, weight=1)

        hyper = ttk.LabelFrame(root, text="Hyperopt review")
        hyper.grid(row=0, column=0, sticky="ew", padx=8, pady=8)
        for i in range(4):
            hyper.grid_columnconfigure(i, weight=1 if i in (1, 3) else 0)

        hyper_file_label, hyper_file_entry = self._labeled_entry(hyper, 0, 0, "Hyperopt file", self.review_hyperopt_file_var)
        hyper_file_browse = ttk.Button(hyper, text="Browse", command=self.browse_review_hyperopt_file)
        hyper_file_browse.grid(row=0, column=2, sticky="ew", padx=8, pady=8)
        hyper_file_latest = ttk.Button(hyper, text="Latest", command=self.set_latest_hyperopt_review_file)
        hyper_file_latest.grid(row=0, column=3, sticky="ew", padx=8, pady=8)
        no_details_check = ttk.Checkbutton(hyper, text="hyperopt-list --no-details", variable=self.review_hyperopt_no_details_var)
        no_details_check.grid(row=1, column=0, sticky="w", padx=8, pady=8)
        epoch_label, epoch_entry = self._labeled_entry(hyper, 1, 2, "Epoch index", self.review_hyperopt_index_var)
        best_check = ttk.Checkbutton(hyper, text="--best", variable=self.review_hyperopt_best_var)
        best_check.grid(row=2, column=0, sticky="w", padx=8, pady=8)
        profitable_check = ttk.Checkbutton(hyper, text="--profitable", variable=self.review_hyperopt_profitable_var)
        profitable_check.grid(row=2, column=1, sticky="w", padx=8, pady=8)
        print_json_check = ttk.Checkbutton(hyper, text="--print-json", variable=self.review_hyperopt_print_json_var)
        print_json_check.grid(row=2, column=2, sticky="w", padx=8, pady=8)
        no_header_check = ttk.Checkbutton(hyper, text="--no-header", variable=self.review_hyperopt_no_header_var)
        no_header_check.grid(row=2, column=3, sticky="w", padx=8, pady=8)
        breakdown_label = ttk.Label(hyper, text="Breakdown")
        breakdown_label.grid(row=3, column=0, sticky="w", padx=8, pady=8)
        breakdown_combo = ttk.Combobox(hyper, textvariable=self.review_hyperopt_breakdown_var, values=REVIEW_BREAKDOWN_VALUES, width=16, state="readonly")
        breakdown_combo.grid(row=3, column=1, sticky="w", padx=8, pady=8)
        buttons = ttk.Frame(hyper)
        buttons.grid(row=4, column=0, columnspan=4, sticky="ew", padx=8, pady=8)
        hyper_list = ttk.Button(buttons, text="Hyperopt List", command=self.on_review_hyperopt_list)
        hyper_list.pack(side="left", padx=(0, 8))
        show_epoch = ttk.Button(buttons, text="Show Epoch", command=self.on_review_hyperopt_show)
        show_epoch.pack(side="left", padx=(0, 8))
        show_best = ttk.Button(buttons, text="Show Best", command=self.on_review_hyperopt_show_best)
        show_best.pack(side="left")
        self._add_tooltips(
            "Context: Hyperopt result artifact to inspect in Review tab. Outcome: Used by Hyperopt List / Show commands; Latest auto-picks newest file when not set. "
            "Example: `user_data/hyperopt_results/strategy_2026-04-21.fthypt`.",
            hyper_file_label,
            hyper_file_entry,
            hyper_file_browse,
        )
        self._add_tooltips(
            "Context: Fast file selection shortcut. Outcome: Fills Hyperopt file with newest result in hyperopt_results folder. "
            "Example: Click Latest right after a completed hyperopt run.",
            hyper_file_latest,
        )
        self._add_tooltips(
            "Context: Output verbosity control for hyperopt-list. Outcome: Omits large per-row detail blocks for cleaner high-level scan. "
            "Example: Use when reviewing hundreds of epochs quickly.",
            no_details_check,
        )
        self._add_tooltips(
            "Context: Specific epoch selector for detailed inspection. Outcome: Show Epoch targets this index; blank defaults to current field value behavior. "
            "Example: Set `-1` to inspect latest epoch entry.",
            epoch_label,
            epoch_entry,
        )
        self._add_tooltips(
            "Context: Result filtering flag for hyperopt-list/show. Outcome: Limits output to best-scoring epoch only. "
            "Example: Combine with Print JSON for structured best-result export.",
            best_check,
        )
        self._add_tooltips(
            "Context: Profitability filter for epoch listing. Outcome: Shows only profitable epochs in output. "
            "Example: Remove negative-return epochs from manual review.",
            profitable_check,
        )
        self._add_tooltips(
            "Context: Output format selection. Outcome: Emits JSON-formatted output instead of table text. "
            "Example: Pipe review output into downstream scripts.",
            print_json_check,
        )
        self._add_tooltips(
            "Context: Compact text output preference. Outcome: Suppresses table headers in hyperopt output. "
            "Example: Easier diffing when comparing repeated runs.",
            no_header_check,
        )
        self._add_tooltips(
            "Context: Time-bucket grouping for review summaries. Outcome: Groups report rows by none/day/week/month/year/weekday. "
            "Example: Choose `week` to spot weekly drift in performance.",
            breakdown_label,
            breakdown_combo,
        )
        self._add_tooltips(
            "Context: Run hyperopt-list with current file and flags. Outcome: Prints epoch summary rows to Review output pane. "
            "Example: Audit parameter quality before picking a candidate epoch.",
            hyper_list,
        )
        self._add_tooltips(
            "Context: Deep inspection command for one epoch. Outcome: Prints detailed fields for selected epoch index. "
            "Example: Inspect epoch 42 settings and metrics.",
            show_epoch,
        )
        self._add_tooltips(
            "Context: Shortcut for best epoch detail view. Outcome: Prints details for best-scoring epoch in current file. "
            "Example: Confirm final selected candidate parameters.",
            show_best,
        )
        ttk.Label(hyper, text="Freqtrade uses the latest result automatically. If you specify a file, use one from userdir/hyperopt_results.", foreground="#666666", wraplength=980, justify="left").grid(row=5, column=0, columnspan=4, sticky="w", padx=8, pady=(0, 8))

        back = ttk.LabelFrame(root, text="Backtest review")
        back.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 8))
        for i in range(4):
            back.grid_columnconfigure(i, weight=1 if i in (1, 3) else 0)

        back_file_label, back_file_entry = self._labeled_entry(back, 0, 0, "Backtest file", self.review_backtest_file_var)
        back_file_browse = ttk.Button(back, text="Browse", command=self.browse_review_backtest_file)
        back_file_browse.grid(row=0, column=2, sticky="ew", padx=8, pady=8)
        back_file_latest = ttk.Button(back, text="Latest", command=self.set_latest_backtest_review_file)
        back_file_latest.grid(row=0, column=3, sticky="ew", padx=8, pady=8)
        directory_label, directory_entry, directory_button = self._path_row(back, 1, "Backtest directory", self.review_backtest_directory_var, directory=True, tooltip="Folder that contains backtest result files. Backtest analysis can also read a separate results directory when you point it here.")
        show_pair_list_check = ttk.Checkbutton(back, text="--show-pair-list", variable=self.review_backtest_show_pair_list_var)
        show_pair_list_check.grid(row=2, column=0, sticky="w", padx=8, pady=8)
        back_breakdown_label = ttk.Label(back, text="Breakdown")
        back_breakdown_label.grid(row=2, column=2, sticky="w", padx=8, pady=8)
        back_breakdown_combo = ttk.Combobox(back, textvariable=self.review_backtest_breakdown_var, values=REVIEW_BREAKDOWN_VALUES, width=16, state="readonly")
        back_breakdown_combo.grid(row=2, column=3, sticky="w", padx=8, pady=8)
        analysis_groups_label, analysis_groups_entry = self._labeled_entry(back, 3, 0, "Analysis groups", self.review_analysis_groups_var)
        indicators_label, indicators_entry = self._labeled_entry(back, 3, 2, "Indicators", self.review_indicator_list_var)
        entry_reasons_label, entry_reasons_entry = self._labeled_entry(back, 4, 0, "Entry reasons", self.review_enter_reasons_var)
        exit_reasons_label, exit_reasons_entry = self._labeled_entry(back, 4, 2, "Exit reasons", self.review_exit_reasons_var)
        entry_only_check = ttk.Checkbutton(back, text="--entry-only", variable=self.review_entry_only_var)
        entry_only_check.grid(row=5, column=0, sticky="w", padx=8, pady=8)
        exit_only_check = ttk.Checkbutton(back, text="--exit-only", variable=self.review_exit_only_var)
        exit_only_check.grid(row=5, column=1, sticky="w", padx=8, pady=8)
        rejected_check = ttk.Checkbutton(back, text="--rejected-signals", variable=self.review_rejected_signals_var)
        rejected_check.grid(row=5, column=2, sticky="w", padx=8, pady=8)
        analysis_csv_check = ttk.Checkbutton(back, text="--analysis-to-csv", variable=self.review_analysis_to_csv_var)
        analysis_csv_check.grid(row=5, column=3, sticky="w", padx=8, pady=8)
        csv_path_label, csv_path_entry = self._labeled_entry(back, 6, 0, "CSV path", self.review_analysis_csv_path_var)
        self._add_tooltips(
            "Context: Backtest result archive used by Show/Analysis commands. Outcome: Selected file is passed as --backtest-filename for review. "
            "Example: `backtest-result-2026-04-20_14-30-00.zip`.",
            back_file_label,
            back_file_entry,
            back_file_browse,
        )
        self._add_tooltips(
            "Context: Fast file selection shortcut for backtest review. Outcome: Auto-fills Backtest file with newest result archive. "
            "Example: Click after finishing a backtest run.",
            back_file_latest,
        )
        self._add_tooltips(
            "Context: Optional results folder reference. Outcome: Helps review helpers discover related result files in one location. "
            "Example: Point to `user_data/backtest_results`.",
            directory_label,
            directory_entry,
            directory_button,
        )
        self._add_tooltips(
            "Context: Backtesting-analysis output scope toggle. Outcome: Includes pair-list details in generated report tables. "
            "Example: See per-pair coverage in one report run.",
            show_pair_list_check,
        )
        self._add_tooltips(
            "Context: Time-bucket grouping for backtest review tables. Outcome: Aggregates rows by selected cadence. "
            "Example: Choose `weekday` to detect day-of-week behavior.",
            back_breakdown_label,
            back_breakdown_combo,
        )
        self._add_tooltips(
            "Context: Select analysis modules by numeric IDs. Outcome: Adds --analysis-groups tokens to analysis command. "
            "Example: `0 1 2 5`.",
            analysis_groups_label,
            analysis_groups_entry,
        )
        self._add_tooltips(
            "Context: Limit analysis output to chosen indicator columns. Outcome: Adds --indicator-list tokens to output. "
            "Example: `ema_fast rsi volume_mean`.",
            indicators_label,
            indicators_entry,
        )
        self._add_tooltips(
            "Context: Filter report rows by entry reason tags. Outcome: Adds --enter-reason-list filter. "
            "Example: `buy_signal dip_recovery`.",
            entry_reasons_label,
            entry_reasons_entry,
        )
        self._add_tooltips(
            "Context: Filter report rows by exit reason tags. Outcome: Adds --exit-reason-list filter. "
            "Example: `roi stop_loss trailing_stop_loss`.",
            exit_reasons_label,
            exit_reasons_entry,
        )
        self._add_tooltips(
            "Context: Focus mode for entry-side diagnostics. Outcome: Limits analysis output to entry-focused rows only. "
            "Example: Use when debugging signal quality at entry.",
            entry_only_check,
        )
        self._add_tooltips(
            "Context: Focus mode for exit-side diagnostics. Outcome: Limits analysis output to exit-focused rows only. "
            "Example: Use when validating custom exit logic.",
            exit_only_check,
        )
        self._add_tooltips(
            "Context: Capacity/constraint diagnostics switch. Outcome: Prints rejected entry signals caused by constraints like max-open-trades. "
            "Example: Explain why entries were skipped in strong trends.",
            rejected_check,
        )
        self._add_tooltips(
            "Context: Report output target mode. Outcome: Writes large analysis tables to CSV files instead of console text. "
            "Example: Enable for spreadsheet or pandas workflows.",
            analysis_csv_check,
        )
        self._add_tooltips(
            "Context: Output directory for CSV analysis exports. Outcome: Used when --analysis-to-csv is enabled; blank uses default behavior. "
            "Example: `research/review_exports`.",
            csv_path_label,
            csv_path_entry,
        )
        back_buttons = ttk.Frame(back)
        back_buttons.grid(row=7, column=0, columnspan=4, sticky="ew", padx=8, pady=8)
        back_show = ttk.Button(back_buttons, text="Backtesting Show", command=self.on_review_backtest_show)
        back_show.pack(side="left", padx=(0, 8))
        back_analysis = ttk.Button(back_buttons, text="Backtesting Analysis", command=self.on_review_backtest_analysis)
        back_analysis.pack(side="left")
        self._add_tooltips(
            "Context: Summary-style backtest inspection command. Outcome: Runs `backtesting show` on selected result archive. "
            "Example: Quick check of headline metrics before deep analysis.",
            back_show,
        )
        self._add_tooltips(
            "Context: Deep analysis command for backtest signals/trades. Outcome: Runs `backtesting-analysis` with current filters and toggles. "
            "Example: Export rejected signals and indicator slices to CSV.",
            back_analysis,
        )

        review_console_frame = ttk.LabelFrame(root, text="Review output")
        review_console_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=(0, 8))
        review_console_frame.grid_columnconfigure(0, weight=1)
        review_console_frame.grid_rowconfigure(0, weight=1)
        self.review_console = scrolledtext.ScrolledText(review_console_frame, wrap="word", height=20)
        self.review_console.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.review_console.configure(state="disabled")

    def _bind_variable_traces(self) -> None:
        trace_vars: list[tk.Variable] = [
            self.run_type_var,
            self.backend_var,
            self.project_root_var,
            self.python_exe_var,
            self.userdir_var,
            self.datadir_var,
            self.logfile_var,
            self.verbose_var,
            self.log_level_var,
            self.no_color_var,
            self.strategy_file_var,
            self.strategy_class_var,
            self.recursive_strategy_search_var,
            self.pair_mode_var,
            self.pair_reference_format_var,
            self.db_url_var,
            self.fee_var,
            self.dry_run_wallet_var,
            self.stake_amount_var,
            self.max_open_trades_var,
            self.timeframe_var,
            self.timerange_var,
            self.timeframe_detail_var,
            self.enable_protections_var,
            self.enable_position_stacking_var,
            self.backtest_export_var,
            self.backtest_breakdown_var,
            self.lookahead_min_trade_amount_var,
            self.lookahead_targeted_trade_amount_var,
            self.lookahead_export_filename_var,
            self.lookahead_allow_limit_orders_var,
            self.hyperopt_epochs_var,
            self.hyperopt_spaces_var,
            self.hyperopt_jobs_var,
            self.hyperopt_random_state_var,
            self.hyperopt_min_trades_var,
            self.hyperopt_loss_var,
            self.hyperopt_early_stop_var,
            self.hyperopt_ignore_missing_spaces_var,
            self.hyperopt_disable_param_export_var,
            self.hyperopt_analyze_per_epoch_var,
            self.hyperopt_print_all_var,
            self.hyperopt_monitor_results_var,
            self.hyperopt_results_file_var,
            self.hyperopt_results_poll_ms_var,
            self.explorer_sampling_seed_var,
            self.explorer_max_loops_var,
            self.explorer_epochs_var,
            self.explorer_random_state_var,
            self.explorer_backtest_workers_var,
            self.explorer_selection_mode_var,
            self.explorer_target_namespace_var,
            self.explorer_target_count_var,
            self.explorer_min_param_count_var,
            self.explorer_keeper_enabled_var,
            self.explorer_keeper_save_dir_var,
            self.explorer_keeper_win_numerator_var,
            self.explorer_keeper_win_denominator_var,
            self.explorer_keeper_min_profit_per_window_var,
            self.download_exchange_var,
            self.download_pairs_file_var,
            self.download_timeframes_var,
            self.download_days_var,
            self.download_new_pairs_days_var,
            self.download_timerange_var,
            self.download_trading_mode_var,
            self.download_candle_types_var,
            self.download_data_format_ohlcv_var,
            self.download_data_format_trades_var,
            self.download_include_inactive_var,
            self.download_no_parallel_var,
            self.download_dl_trades_var,
            self.download_convert_var,
            self.download_erase_var,
            self.download_prepend_var,
            self.frequi_url_var,
            self.frequi_open_after_launch_var,
            self.news_config_path_var,
            self.news_data_dir_var,
            self.news_db_path_var,
            self.news_interval_seconds_var,
            self.news_once_var,
            self.web_config_path_var,
            self.web_data_dir_var,
            self.web_db_path_var,
            self.web_interval_seconds_var,
            self.web_once_var,
            self.orderbook_config_path_var,
            self.orderbook_data_dir_var,
            self.orderbook_depth_levels_var,
            self.orderbook_stream_update_ms_var,
            self.orderbook_metric_interval_seconds_var,
            self.orderbook_snapshot_interval_seconds_var,
            self.orderbook_store_snapshots_var,
            self.orderbook_capacity_warning_mb_var,
            self.orderbook_capacity_critical_mb_var,
            self.orderbook_max_symbols_var,
        ]
        trace_vars.extend(self.explorer_backtest_count_vars.values())
        trace_vars.extend(self.explorer_12m_holdout_vars.values())
        for var in trace_vars:
            var.trace_add("write", lambda *_args: self._on_any_input_changed())
        for var in self.news_health_filter_vars.values():
            var.trace_add("write", lambda *_args: self._populate_news_source_health_tree())
        for var in self.web_health_filter_vars.values():
            var.trace_add("write", lambda *_args: self._populate_web_source_health_tree())

    def _on_any_input_changed(self) -> None:
        self.refresh_hyperopt_loss_dropdown()
        self._refresh_explorer_target_count_label()
        self._refresh_explorer_namespace_dropdown()
        self._refresh_explorer_summary_context()
        self.refresh_mode_options()
        self.refresh_all_command_previews()

    def refresh_all_command_previews(self) -> None:
        self.refresh_command_preview()
        self.refresh_news_command_preview()
        self.refresh_web_command_preview()
        self.refresh_orderbook_pair_preview()
        self.refresh_orderbook_estimate()

    def _ordered_catalog_namespaces(self, catalog: dict[str, Any] | None = None) -> list[str]:
        catalog = catalog or self.catalog_data or {}
        namespaces = catalog.get("namespaces") or {}
        preferred = list(catalog.get("preferred_namespace_order") or catalog.get("known_namespace_order") or [])
        ordered = [namespace for namespace in preferred if namespace in namespaces]
        ordered.extend(sorted(namespace for namespace in namespaces if namespace not in ordered))
        return ordered

    def _refresh_explorer_namespace_dropdown(self) -> None:
        combo = getattr(self, "explorer_target_namespace_combo", None)
        if combo is None:
            return
        try:
            catalog = self._load_catalog_data()
            namespaces = self._ordered_catalog_namespaces(catalog)
        except Exception:
            namespaces = []
        combo.configure(values=namespaces)
        current = self.explorer_target_namespace_var.get().strip()
        if namespaces and current not in namespaces:
            self.explorer_target_namespace_var.set("mode" if "mode" in namespaces else namespaces[0])
        self._refresh_explorer_mode_relevance()


    def load_explorer_market_windows(self) -> list[dict[str, Any]]:
        windows_file = app_path(EXPLORER_MARKET_WINDOWS_FILE)
        if not windows_file.exists():
            return []
        data: Any = None
        for encoding in ("utf-8", "utf-8-sig"):
            try:
                data = json.loads(windows_file.read_text(encoding=encoding))
                break
            except Exception:
                continue
        if data is None:
            return []
        if isinstance(data, list):
            windows = data
        elif isinstance(data, dict):
            windows = data.get("market_windows", data.get("windows", []))
        else:
            windows = []
        return windows if isinstance(windows, list) else []

    def selected_explorer_windows(self) -> list[str]:
        return [name for name, var in self.explorer_window_vars.items() if var.get()]

    def set_explorer_windows(self, enabled: bool) -> None:
        for var in self.explorer_window_vars.values():
            var.set(enabled)

    def set_explorer_windows_by_regime(self, regime: str) -> None:
        wanted: set[str] = set()
        for window in self.load_explorer_market_windows():
            if window.get("regime") == regime and window.get("name"):
                wanted.add(str(window["name"]))
        for name, var in self.explorer_window_vars.items():
            var.set(name in wanted)

    def _explorer_market_type_button_style(self, selected: bool) -> dict[str, Any]:
        if selected:
            return {
                "bg": "#2F80ED",
                "fg": "#FFFFFF",
                "activebackground": "#2F80ED",
                "activeforeground": "#FFFFFF",
                "relief": "sunken",
            }
        return {
            "bg": "#E9EEF6",
            "fg": "#1F2937",
            "activebackground": "#DCE6F6",
            "activeforeground": "#1F2937",
            "relief": "raised",
        }

    @staticmethod
    def _safe_nonnegative_int(value: Any) -> int:
        parsed = parse_int(str(value).strip()) if value is not None else None
        return max(0, parsed or 0)

    def _explorer_backtest_count_requests(self) -> dict[str, int]:
        return {
            regime: self._safe_nonnegative_int(self.explorer_backtest_count_vars[regime].get())
            for regime in EXPLORER_REGIME_VALUES
        }

    def _selected_12m_holdout_windows(self) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        for window in EXPLORER_12M_HOLDOUT_WINDOWS:
            label = str(window["label"])
            if self.explorer_12m_holdout_vars.get(label) and self.explorer_12m_holdout_vars[label].get():
                selected.append({key: value for key, value in window.items() if key != "tooltip"})
        return selected

    def _explorer_market_type_values(self, bucket: str) -> dict[str, tk.BooleanVar]:
        if bucket == "backtest":
            return self.explorer_backtest_market_type_vars
        return self.explorer_hyperopt_market_type_vars

    def _explorer_market_type_buttons(self, bucket: str) -> dict[str, tk.Button]:
        if bucket == "backtest":
            return self.explorer_backtest_market_type_buttons
        return self.explorer_hyperopt_market_type_buttons

    def _explorer_selected_market_types(self, bucket: str) -> list[str]:
        values = self._explorer_market_type_values(bucket)
        selected = [value for value in EXPLORER_REGIME_VALUES if values[value].get()]
        if not selected or len(selected) == len(EXPLORER_REGIME_VALUES):
            return []
        return selected

    def _explorer_effective_market_types(self, bucket: str) -> set[str]:
        values = self._explorer_market_type_values(bucket)
        selected = {value for value in EXPLORER_REGIME_VALUES if values[value].get()}
        if not selected or len(selected) == len(EXPLORER_REGIME_VALUES):
            return set(EXPLORER_REGIME_VALUES)
        return selected

    def _toggle_explorer_market_type(self, bucket: str, regime: str) -> None:
        values = self._explorer_market_type_values(bucket)
        var = values.get(regime)
        if var is None:
            return
        var.set(not var.get())
        self._refresh_explorer_market_type_tiles(bucket)

    def _refresh_explorer_market_type_tiles(self, bucket: str) -> None:
        values = self._explorer_market_type_values(bucket)
        buttons = self._explorer_market_type_buttons(bucket)
        for regime in EXPLORER_REGIME_VALUES:
            button = buttons.get(regime)
            if button is None:
                continue
            selected = bool(values[regime].get())
            button.configure(text=EXPLORER_MARKET_TYPE_COMPACT_LABELS.get(regime, regime.title()), **self._explorer_market_type_button_style(selected))

    def _explorer_selected_market_window_records(self) -> list[dict[str, Any]]:
        enabled = set(self.selected_explorer_windows())
        return [window for window in self.load_explorer_market_windows() if str(window.get("name") or "") in enabled]

    def _refresh_explorer_target_count_label(self) -> None:
        if self.explorer_selection_mode_var.get() == "Custom batches":
            tooltip_text = "Custom batches mode uses checked committed batches from the Catalog / Custom Batches sub-tab. Target count and namespace are ignored."
        elif self.explorer_selection_mode_var.get() == "Random tags":
            tooltip_text = "Total number of unique tag targets selected for this Explorer run. Blank is treated as 0. 0 means all matching tags, which can be very large."
        elif self.explorer_selection_mode_var.get() == "Random namespace values":
            tooltip_text = "Total number of unique values sampled from the selected namespace. For namespace mode, targets look like mode:breakout_volume_seed."
        else:
            tooltip_text = "Total number of unique family targets selected for this Explorer run. Blank is treated as 0. 0 means all families."
        for tooltip in getattr(self, "explorer_target_count_tooltips", []):
            tooltip.text = tooltip_text
        self._refresh_explorer_mode_relevance()

    def _refresh_explorer_mode_relevance(self) -> None:
        mode = self.explorer_selection_mode_var.get()
        custom_batches = mode == "Custom batches"
        namespace_mode = mode == "Random namespace values"

        target_entry = getattr(self, "explorer_target_count_entry_widget", None)
        if target_entry is not None:
            target_entry.configure(state="disabled" if custom_batches else "normal")

        target_label = getattr(self, "explorer_target_count_label_widget", None)
        if target_label is not None:
            target_label.configure(foreground="#777777" if custom_batches else "")

        namespace_label = getattr(self, "explorer_namespace_label_widget", None)
        if namespace_label is not None:
            namespace_label.configure(foreground="" if namespace_mode else "#777777")

        namespace_combo = getattr(self, "explorer_target_namespace_combo", None)
        if namespace_combo is not None:
            namespace_combo.configure(state="readonly" if namespace_mode else "disabled")

        min_params_entry = getattr(self, "explorer_min_params_entry_widget", None)
        if min_params_entry is not None:
            min_params_entry.configure(state="disabled" if custom_batches else "normal")

        min_params_label = getattr(self, "explorer_min_params_label_widget", None)
        if min_params_label is not None:
            min_params_label.configure(foreground="#777777" if custom_batches else "")

    def _apply_explorer_market_type_selection(self, bucket: str, values: Any) -> None:
        vars_map = self._explorer_market_type_values(bucket)
        raw_values = values if isinstance(values, list) else token_list(str(values or ""))
        selected = {
            str(value).lower()
            for value in raw_values
            if str(value).lower() in EXPLORER_REGIME_VALUES
        }
        if not selected or len(selected) == len(EXPLORER_REGIME_VALUES):
            for regime in EXPLORER_REGIME_VALUES:
                vars_map[regime].set(True)
        else:
            for regime in EXPLORER_REGIME_VALUES:
                vars_map[regime].set(regime in selected)
        self._refresh_explorer_market_type_tiles(bucket)

    def _build_market_type_filter_box(self, parent: ttk.Frame, column: int, title: str, bucket: str) -> None:
        box = ttk.LabelFrame(parent, text=title)
        box.grid(row=0, column=column, sticky="nsew", padx=5, pady=6)
        box.grid_columnconfigure(0, weight=1)

        if bucket == "hyperopt":
            filter_tip = "Limits which selected market windows can be chosen for the HyperOpt search. None selected is treated as all. All selected is also treated as all."
        else:
            filter_tip = "Limits which selected market windows can be used for champion/challenger backtesting. None selected is treated as all. All selected is also treated as all."
        ToolTip(box, filter_tip)

        tile_frame = ttk.Frame(box)
        tile_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=(4, 5))
        for idx in range(6):
            tile_frame.grid_columnconfigure(idx, weight=1)

        buttons = self._explorer_market_type_buttons(bucket)
        for idx, regime in enumerate(EXPLORER_REGIME_VALUES):
            row = 0
            col = idx
            btn = tk.Button(
                tile_frame,
                text=EXPLORER_MARKET_TYPE_COMPACT_LABELS.get(regime, regime.title()),
                command=lambda r=regime, b=bucket: self._toggle_explorer_market_type(b, r),
                width=6,
                padx=2,
                pady=1,
                borderwidth=1,
                relief="raised",
                cursor="hand2",
            )
            btn.grid(row=row, column=col, sticky="ew", padx=2, pady=2)
            buttons[regime] = btn
            ToolTip(btn, filter_tip)

        self._refresh_explorer_market_type_tiles(bucket)

    def _build_backtest_count_box(self, parent: ttk.Frame, column: int) -> None:
        box = ttk.LabelFrame(parent, text="Backtest sample counts")
        box.grid(row=0, column=column, sticky="nsew", padx=5, pady=6)
        for col in range(4):
            box.grid_columnconfigure(col, weight=1 if col in (1, 3) else 0)
        tip = "Blank means 0. Positive values request up to that many normal market windows of the selected type. Holdout windows are selected separately."
        ToolTip(box, tip)

        for idx, regime in enumerate(EXPLORER_REGIME_VALUES):
            row = idx // 2
            col = (idx % 2) * 2
            label = EXPLORER_MARKET_TYPE_LABELS.get(regime, regime.title())
            ttk.Label(box, text=f"{label}:").grid(row=row, column=col, sticky="w", padx=(6, 2), pady=3)
            entry = ttk.Entry(box, textvariable=self.explorer_backtest_count_vars[regime], width=5)
            entry.grid(row=row, column=col + 1, sticky="ew", padx=(0, 6), pady=3)
            ToolTip(entry, tip)

    # Widget helpers
    # ------------------------------------------------------------------
    def _path_row(
        self,
        parent: ttk.Frame,
        row: int,
        label: str,
        variable: tk.StringVar,
        *,
        directory: bool,
        executable: bool = False,
        callback: callable[[], None] | None = None,
        tooltip: str | None = None,
    ) -> tuple[ttk.Label, ttk.Entry, ttk.Button]:
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=0, sticky="w", padx=8, pady=8)
        entry = ttk.Entry(parent, textvariable=variable)
        entry.grid(row=row, column=1, sticky="ew", padx=8, pady=8)
        button_cmd = callback if callback else (lambda v=variable, d=directory, e=executable: self._browse_path(v, d, e))
        button = ttk.Button(parent, text="Browse", command=button_cmd)
        button.grid(row=row, column=2, sticky="ew", padx=8, pady=8)
        if tooltip:
            ToolTip(label_widget, tooltip)
            ToolTip(entry, tooltip)
            ToolTip(button, tooltip)
        return label_widget, entry, button

    def _labeled_entry(
        self,
        parent: ttk.Frame,
        row: int,
        col: int,
        label: str,
        variable: tk.StringVar,
    ) -> tuple[ttk.Label, ttk.Entry]:
        label_widget = ttk.Label(parent, text=label)
        label_widget.grid(row=row, column=col, sticky="w", padx=8, pady=8)
        entry_widget = ttk.Entry(parent, textvariable=variable)
        entry_widget.grid(row=row, column=col + 1, sticky="ew", padx=8, pady=8)
        return label_widget, entry_widget

    def _add_tooltips(self, text: str, *widgets: tk.Widget) -> None:
        for widget in widgets:
            ToolTip(widget, text)

    def _parse_minutes_to_seconds(self, value: str, default_minutes: int) -> int:
        text = value.strip()
        minutes = default_minutes
        if text:
            try:
                minutes = int(round(float(text)))
            except Exception:
                minutes = default_minutes
        if minutes < 1:
            minutes = 1
        return minutes * 60

    def _seconds_to_minutes_text(self, value: Any, default_seconds: int) -> str:
        try:
            seconds = int(round(float(value)))
        except Exception:
            seconds = default_seconds
        minutes = max(1, int(round(seconds / 60.0)))
        return str(minutes)

    def _browse_path(self, variable: tk.StringVar, directory: bool, executable: bool = False) -> None:
        if directory:
            path = filedialog.askdirectory()
        else:
            filetypes = [("All files", "*.*")]
            if executable:
                filetypes = [("Python", "python*"), ("All files", "*.*")]
            path = filedialog.askopenfilename(filetypes=filetypes)
        if path:
            variable.set(path)

    def refresh_hyperopt_loss_dropdown(self, force: bool = False) -> None:
        project_root = self.project_root_var.get().strip() or str(Path.cwd())
        userdir = self.userdir_var.get().strip()
        scan_key = (project_root, userdir, str(Path(__file__).resolve().parent))
        if not force and scan_key == self.hyperopt_loss_scan_key:
            return

        self.hyperopt_loss_scan_key = scan_key
        losses = discover_hyperopt_losses(Path(project_root), Path(userdir) if userdir else None)
        self.hyperopt_loss_info_by_name = {loss.name: loss for loss in losses}
        names = [loss.name for loss in losses]
        current = self.hyperopt_loss_var.get().strip()
        if current and current not in names:
            names.append(current)
            self.hyperopt_loss_info_by_name[current] = HyperoptLossInfo(
                name=current,
                source="Manual entry",
                description="This name is not currently discoverable from the scanned loss folders, but it will still be passed to Freqtrade.",
                path=None,
            )
        if self.hyperopt_loss_combo is not None:
            self.hyperopt_loss_combo.configure(values=names)
        if self.explorer_hyperopt_loss_combo is not None:
            self.explorer_hyperopt_loss_combo.configure(values=names)

    def show_hyperopt_loss_popup(self) -> None:
        self.refresh_hyperopt_loss_dropdown()
        name = self.hyperopt_loss_var.get().strip()
        if not name:
            messagebox.showinfo(APP_TITLE, "Select a Hyperopt loss first.")
            return
        info = self.hyperopt_loss_info_by_name.get(name)
        if info is None:
            info = HyperoptLossInfo(
                name=name,
                source="Manual entry",
                description="This loss is not currently in the discovered list. Check the class name and the scanned folders.",
                path=None,
            )

        popup = tk.Toplevel(self)
        popup.title(f"Hyperopt loss: {info.name}")
        popup.transient(self)
        popup.geometry("680x420")
        popup.minsize(560, 320)
        popup.grid_columnconfigure(0, weight=1)
        popup.grid_rowconfigure(1, weight=1)

        ttk.Label(popup, text=info.name, font=("", 12, "bold")).grid(row=0, column=0, sticky="w", padx=12, pady=(12, 4))
        text = scrolledtext.ScrolledText(popup, wrap="word", height=14)
        text.grid(row=1, column=0, sticky="nsew", padx=12, pady=8)
        details = [f"Source: {info.source}"]
        if info.path:
            details.append(f"File: {info.path}")
        details.extend(["", info.description.strip() or "No description was found in the class or module docstring."])
        text.insert("1.0", "\n".join(details))
        text.configure(state="disabled")
        ttk.Button(popup, text="Close", command=popup.destroy).grid(row=2, column=0, sticky="e", padx=12, pady=(0, 12))
        popup.grab_set()

    def on_browse_strategy(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Python files", "*.py"), ("All files", "*.*")])
        if not path:
            return
        self.strategy_file_var.set(path)
        class_names = extract_class_names(path)
        self.strategy_class_combo["values"] = class_names
        if class_names:
            if self.strategy_class_var.get() not in class_names:
                self.strategy_class_var.set(class_names[0])
        else:
            self.strategy_class_var.set(derive_module_stem(path))
        self.refresh_command_preview()

    def browse_hyperopt_result_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Hyperopt results", "*.fthypt *.pickle"), ("All files", "*.*")])
        if path:
            self.hyperopt_results_file_var.set(path)

    def browse_review_hyperopt_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Hyperopt results", "*.fthypt *.pickle"), ("All files", "*.*")])
        if path:
            self.review_hyperopt_file_var.set(path)

    def browse_review_backtest_file(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Backtest results", "*.zip *.json"), ("All files", "*.*")])
        if path:
            self.review_backtest_file_var.set(path)

    def _hyperopt_results_dir(self) -> Path:
        userdir = self.userdir_var.get().strip()
        if userdir:
            return Path(userdir) / "hyperopt_results"
        return Path.cwd() / "user_data" / "hyperopt_results"

    def _backtest_results_dir(self) -> Path:
        custom = self.review_backtest_directory_var.get().strip()
        if custom:
            return Path(custom)
        userdir = self.userdir_var.get().strip()
        if userdir:
            return Path(userdir) / "backtest_results"
        return Path.cwd() / "user_data" / "backtest_results"

    def _latest_file(self, directory: Path, patterns: list[str]) -> Path | None:
        if not directory.exists():
            return None
        candidates: list[Path] = []
        for pattern in patterns:
            candidates.extend([p for p in directory.glob(pattern) if p.is_file()])
        if not candidates:
            return None
        return max(candidates, key=lambda p: p.stat().st_mtime)

    def set_latest_hyperopt_review_file(self) -> None:
        latest = self._latest_file(self._hyperopt_results_dir(), ["*.fthypt", "*.pickle"])
        if latest:
            self.review_hyperopt_file_var.set(str(latest))
        else:
            messagebox.showinfo(APP_TITLE, "No hyperopt result file found in userdir/hyperopt_results.")

    def set_latest_backtest_review_file(self, *, silent: bool = False) -> None:
        latest = self._latest_file(
            self._backtest_results_dir(),
            [
                "backtest-result-*.zip",
                "backtest-result-*.meta.json",
                "backtest-result-*.json",
                "*.zip",
                "*.json",
            ],
        )
        if latest:
            self.review_backtest_file_var.set(str(latest))
        else:
            if not silent:
                messagebox.showinfo(APP_TITLE, "No backtest result file found in the selected results directory.")

    # ------------------------------------------------------------------
    # Presets
    # ------------------------------------------------------------------
    def _load_presets_from_disk(self) -> None:
        try:
            if os.path.exists(self.presets_path):
                with open(self.presets_path, "r", encoding="utf-8-sig") as handle:
                    loaded = json.load(handle)
                    if isinstance(loaded, dict):
                        self.presets = loaded
                        if AUTO_LAST_USED_PRESET in self.presets:
                            self.presets.pop(AUTO_LAST_USED_PRESET, None)
                            self._save_presets_to_disk()
                            messagebox.showwarning(
                                APP_TITLE,
                                "Removed legacy auto-save state from the preset file. "
                                "Runtime launcher state now lives under user_data/runtime.",
                            )
                    else:
                        self.presets = {}
                        messagebox.showwarning(APP_TITLE, "Preset file format was invalid and has been reset.")
            else:
                self.presets = {}
        except Exception as exc:
            messagebox.showwarning(APP_TITLE, f"Failed to load presets:\n{exc}")
            self.presets = {}
        self._refresh_preset_combo()
        self._load_last_used_state()

    def _save_presets_to_disk(self) -> None:
        with open(self.presets_path, "w", encoding="utf-8") as handle:
            json.dump(self.presets, handle, indent=2)

    def _refresh_preset_combo(self) -> None:
        names = sorted(self.presets.keys())
        self.preset_combo["values"] = names
        if self.preset_var.get() not in names:
            self.preset_var.set(names[0] if names else "")

    def _save_last_used_state(self) -> None:
        try:
            Path(self.last_used_state_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.last_used_state_path, "w", encoding="utf-8") as handle:
                json.dump(self.collect_settings(), handle, indent=2)
        except Exception as exc:
            messagebox.showwarning(APP_TITLE, f"Failed to auto-save last used settings:\n{exc}")

    def _load_last_used_state(self) -> None:
        if not os.path.exists(self.last_used_state_path):
            return
        try:
            with open(self.last_used_state_path, "r", encoding="utf-8-sig") as handle:
                payload = json.load(handle)
        except Exception as exc:
            messagebox.showwarning(APP_TITLE, f"Failed to load last used settings:\n{exc}")
            return
        if not isinstance(payload, dict):
            return
        self.apply_settings(payload, source_label="last used state")
        self.preset_var.set("")
        self._append_console("Auto-loaded last used settings.\n")

    def _collect_apply_settings_warnings(self, data: dict[str, Any]) -> list[str]:
        warnings: list[str] = []

        raw_windows = data.get("explorer_market_windows", data.get("explorer_source_windows", data.get("batch_windows")))
        if isinstance(raw_windows, list):
            current_windows = set(self.explorer_window_vars.keys())
            missing_windows = [str(name) for name in raw_windows if str(name) not in current_windows]
            if missing_windows:
                preview = ", ".join(missing_windows[:8])
                suffix = " ..." if len(missing_windows) > 8 else ""
                warnings.append(f"Explorer market windows no longer exist: {preview}{suffix}")

        raw_holdouts = data.get("explorer_12m_holdout_windows")
        if isinstance(raw_holdouts, list):
            current_holdouts = set(self.explorer_12m_holdout_vars.keys())
            missing_holdouts = [str(label) for label in raw_holdouts if str(label) not in current_holdouts]
            if missing_holdouts:
                preview = ", ".join(missing_holdouts[:8])
                suffix = " ..." if len(missing_holdouts) > 8 else ""
                warnings.append(f"12-month holdout windows no longer exist: {preview}{suffix}")

        selection_mode = str(data.get("explorer_selection_mode") or "").strip()
        if selection_mode and selection_mode not in set(EXPLORER_SELECTION_MODES) | {"random_families", "random_tags", "random_namespace", "custom_batches"}:
            warnings.append(f"Explorer selection mode '{selection_mode}' is unavailable; using default.")

        for bucket_name, key_options in (
            ("HyperOpt market types", ("explorer_hyperopt_market_types", "hyperopt_market_types", "batch_hyperopt_market_types")),
            ("Backtest market types", ("explorer_backtest_market_types", "backtest_market_types", "batch_backtest_market_types")),
        ):
            raw_value: Any = None
            for key in key_options:
                if key in data:
                    raw_value = data.get(key)
                    break
            if raw_value is None:
                continue
            values = raw_value if isinstance(raw_value, list) else token_list(str(raw_value))
            invalid = sorted({str(value).lower() for value in values if str(value).lower() and str(value).lower() not in EXPLORER_REGIME_VALUES})
            if invalid:
                warnings.append(f"{bucket_name} included unknown values: {', '.join(invalid)}")

        return warnings

    def save_preset(self) -> None:
        name = self.preset_var.get().strip()
        if not name:
            self.save_preset_as()
            return
        self.presets[name] = self.collect_settings()
        self._save_presets_to_disk()
        self._refresh_preset_combo()
        self._append_console(f"Saved preset: {name}\n")

    def save_preset_as(self) -> None:
        name = simpledialog.askstring(APP_TITLE, "Preset name")
        if not name:
            return
        name = name.strip()
        if not name:
            return
        self.preset_var.set(name)
        self.save_preset()

    def load_selected_preset(self) -> None:
        name = self.preset_var.get().strip()
        if not name:
            messagebox.showinfo(APP_TITLE, "Select a preset first.")
            return
        if name not in self.presets:
            messagebox.showerror(APP_TITLE, f"Preset not found: {name}")
            return
        self.apply_settings(self.presets[name], source_label=f"preset '{name}'")
        self._append_console(f"Loaded preset: {name}\n")

    def delete_selected_preset(self) -> None:
        name = self.preset_var.get().strip()
        if not name:
            return
        if name not in self.presets:
            return
        if not messagebox.askyesno(APP_TITLE, f"Delete preset '{name}'?"):
            return
        del self.presets[name]
        self._save_presets_to_disk()
        self._refresh_preset_combo()
        self._append_console(f"Deleted preset: {name}\n")

    # ------------------------------------------------------------------
    # Settings serialization
    # ------------------------------------------------------------------
    def collect_settings(self) -> dict[str, Any]:
        target_count = self.explorer_target_count_var.get().strip() or "0"
        normalized_hyperopt_jobs = self._normalized_job_workers_value(self.hyperopt_jobs_var.get())
        hyperopt_epochs_total = self._safe_effective_epochs_text(
            self.hyperopt_epochs_var.get(),
            normalized_hyperopt_jobs,
        )
        explorer_epochs_total = self._safe_effective_epochs_text(
            self.explorer_epochs_var.get(),
            normalized_hyperopt_jobs,
        )
        return {
            "run_type": self.run_type_var.get(),
            "backend": self.backend_var.get(),
            "project_root": self.project_root_var.get(),
            "python_exe": self.python_exe_var.get(),
            "userdir": self.userdir_var.get(),
            "datadir": self.datadir_var.get(),
            "logfile": self.logfile_var.get(),
            "verbose": self.verbose_var.get(),
            "log_level": self.log_level_var.get(),
            "no_color": self.no_color_var.get(),
            "config_files": self.config_editor.get_items(),
            "strategy_file": self.strategy_file_var.get(),
            "strategy_class": self.strategy_class_var.get(),
            "recursive_strategy_search": self.recursive_strategy_search_var.get(),
            "pair_mode": self.pair_mode_var.get(),
            "pairs": self.pairs_text.get("1.0", tk.END),
            "blacklist": self.blacklist_text.get("1.0", tk.END),
            "db_url": self.db_url_var.get(),
            "fee": self.fee_var.get(),
            "dry_run_wallet": self.dry_run_wallet_var.get(),
            "stake_amount": self.stake_amount_var.get(),
            "max_open_trades": self.max_open_trades_var.get(),
            "timeframe": self.timeframe_var.get(),
            "timerange": self.timerange_var.get(),
            "timeframe_detail": self.timeframe_detail_var.get(),
            "enable_protections": self.enable_protections_var.get(),
            "enable_position_stacking": self.enable_position_stacking_var.get(),
            "backtest_export": self.backtest_export_var.get(),
            "backtest_breakdown": self.backtest_breakdown_var.get(),
            "lookahead_min_trade_amount": self.lookahead_min_trade_amount_var.get(),
            "lookahead_targeted_trade_amount": self.lookahead_targeted_trade_amount_var.get(),
            "lookahead_export_filename": self.lookahead_export_filename_var.get(),
            "lookahead_allow_limit_orders": self.lookahead_allow_limit_orders_var.get(),
            "hyperopt_epoch_multiplier": self.hyperopt_epochs_var.get(),
            "hyperopt_epochs": hyperopt_epochs_total,
            "hyperopt_spaces": self.hyperopt_spaces_var.get(),
            "hyperopt_jobs": normalized_hyperopt_jobs,
            "hyperopt_random_state": self.hyperopt_random_state_var.get(),
            "hyperopt_min_trades": self.hyperopt_min_trades_var.get(),
            "hyperopt_loss": self.hyperopt_loss_var.get(),
            "hyperopt_early_stop": self.hyperopt_early_stop_var.get(),
            "hyperopt_ignore_missing_spaces": self.hyperopt_ignore_missing_spaces_var.get(),
            "hyperopt_disable_param_export": self.hyperopt_disable_param_export_var.get(),
            "hyperopt_analyze_per_epoch": self.hyperopt_analyze_per_epoch_var.get(),
            "hyperopt_print_all": self.hyperopt_print_all_var.get(),
            "hyperopt_monitor_results": self.hyperopt_monitor_results_var.get(),
            "hyperopt_results_file": self.hyperopt_results_file_var.get(),
            "hyperopt_results_poll_ms": self.hyperopt_results_poll_ms_var.get(),
            "explorer_sampling_seed": self.explorer_sampling_seed_var.get(),
            "explorer_max_loops": self.explorer_max_loops_var.get(),
            "explorer_epoch_multiplier": self.explorer_epochs_var.get(),
            "explorer_epochs": explorer_epochs_total,
            "explorer_random_state": self.explorer_random_state_var.get(),
            "explorer_backtest_workers": self.explorer_backtest_workers_var.get(),
            "explorer_market_windows": self.selected_explorer_windows(),
            "explorer_target_count": target_count,
            "explorer_hyperopt_market_types": self._explorer_selected_market_types("hyperopt"),
            "explorer_backtest_market_types": self._explorer_selected_market_types("backtest"),
            "explorer_backtest_count_bull": self.explorer_backtest_count_vars["bull"].get(),
            "explorer_backtest_count_bear": self.explorer_backtest_count_vars["bear"].get(),
            "explorer_backtest_count_chop": self.explorer_backtest_count_vars["chop"].get(),
            "explorer_backtest_count_crash": self.explorer_backtest_count_vars["crash"].get(),
            "explorer_backtest_count_crossover": self.explorer_backtest_count_vars["crossover"].get(),
            "explorer_12m_holdout_windows": [window["label"] for window in self._selected_12m_holdout_windows()],
            "explorer_selection_mode": self.explorer_selection_mode_var.get(),
            "explorer_target_namespace": self.explorer_target_namespace_var.get(),
            "explorer_selected_custom_batch_ids": sorted(self.custom_batch_selected_run_ids),
            "explorer_min_param_count": self.explorer_min_param_count_var.get(),
            "explorer_keeper_enabled": self.explorer_keeper_enabled_var.get(),
            "explorer_keeper_save_dir": self.explorer_keeper_save_dir_var.get(),
            "explorer_keeper_win_numerator": self.explorer_keeper_win_numerator_var.get(),
            "explorer_keeper_win_denominator": self.explorer_keeper_win_denominator_var.get(),
            "explorer_keeper_min_profit_per_window": self.explorer_keeper_min_profit_per_window_var.get(),
            "download_exchange": self.download_exchange_var.get(),
            "download_pairs_file": self.download_pairs_file_var.get(),
            "download_pairs": self.download_pairs_text.get("1.0", tk.END) if self.download_pairs_text is not None else "",
            "download_timeframes": self.download_timeframes_var.get(),
            "download_days": self.download_days_var.get(),
            "download_new_pairs_days": self.download_new_pairs_days_var.get(),
            "download_timerange": self.download_timerange_var.get(),
            "download_trading_mode": self.download_trading_mode_var.get(),
            "download_candle_types": self.download_candle_types_var.get(),
            "download_data_format_ohlcv": self.download_data_format_ohlcv_var.get(),
            "download_data_format_trades": self.download_data_format_trades_var.get(),
            "download_include_inactive": self.download_include_inactive_var.get(),
            "download_no_parallel": self.download_no_parallel_var.get(),
            "download_dl_trades": self.download_dl_trades_var.get(),
            "download_convert": self.download_convert_var.get(),
            "download_erase": self.download_erase_var.get(),
            "download_prepend": self.download_prepend_var.get(),
            "frequi_url": self.frequi_url_var.get(),
            "frequi_open_after_launch": self.frequi_open_after_launch_var.get(),
            "news_config_path": self.news_config_path_var.get(),
            "news_data_dir": self.news_data_dir_var.get(),
            "news_interval_seconds": str(self._parse_minutes_to_seconds(self.news_interval_seconds_var.get(), 15)),
            "news_once": self.news_once_var.get(),
            "web_config_path": self.web_config_path_var.get(),
            "web_data_dir": self.web_data_dir_var.get(),
            "web_interval_seconds": str(self._parse_minutes_to_seconds(self.web_interval_seconds_var.get(), 360)),
            "web_once": self.web_once_var.get(),
            "orderbook_config_path": self.orderbook_config_path_var.get(),
            "orderbook_data_dir": self.orderbook_data_dir_var.get(),
            "orderbook_exchange": self.orderbook_exchange_var.get(),
            "orderbook_market_type": self.orderbook_market_type_var.get(),
            "orderbook_depth_levels": self.orderbook_depth_levels_var.get(),
            "orderbook_stream_update_ms": self.orderbook_stream_update_ms_var.get(),
            "orderbook_metric_interval_seconds": self.orderbook_metric_interval_seconds_var.get(),
            "orderbook_snapshot_interval_seconds": self.orderbook_snapshot_interval_seconds_var.get(),
            "orderbook_store_snapshots": self.orderbook_store_snapshots_var.get(),
            "orderbook_capacity_warning_mb": self.orderbook_capacity_warning_mb_var.get(),
            "orderbook_capacity_critical_mb": self.orderbook_capacity_critical_mb_var.get(),
            "orderbook_max_symbols": self.orderbook_max_symbols_var.get(),
            "review_hyperopt_file": self.review_hyperopt_file_var.get(),
            "review_hyperopt_limit": self.review_hyperopt_limit_var.get(),
            "review_hyperopt_index": self.review_hyperopt_index_var.get(),
            "review_hyperopt_best": self.review_hyperopt_best_var.get(),
            "review_hyperopt_profitable": self.review_hyperopt_profitable_var.get(),
            "review_hyperopt_print_json": self.review_hyperopt_print_json_var.get(),
            "review_hyperopt_no_header": self.review_hyperopt_no_header_var.get(),
            "review_hyperopt_no_details": self.review_hyperopt_no_details_var.get(),
            "review_hyperopt_breakdown": self.review_hyperopt_breakdown_var.get(),
            "review_backtest_file": self.review_backtest_file_var.get(),
            "review_backtest_directory": self.review_backtest_directory_var.get(),
            "review_backtest_show_pair_list": self.review_backtest_show_pair_list_var.get(),
            "review_backtest_breakdown": self.review_backtest_breakdown_var.get(),
            "review_analysis_groups": self.review_analysis_groups_var.get(),
            "review_enter_reasons": self.review_enter_reasons_var.get(),
            "review_exit_reasons": self.review_exit_reasons_var.get(),
            "review_indicator_list": self.review_indicator_list_var.get(),
            "review_entry_only": self.review_entry_only_var.get(),
            "review_exit_only": self.review_exit_only_var.get(),
            "review_rejected_signals": self.review_rejected_signals_var.get(),
            "review_analysis_to_csv": self.review_analysis_to_csv_var.get(),
            "review_analysis_csv_path": self.review_analysis_csv_path_var.get(),
        }

    def apply_settings(self, data: dict[str, Any], source_label: str = "preset") -> None:
        def set_text(widget: scrolledtext.ScrolledText, value: str) -> None:
            widget.delete("1.0", tk.END)
            widget.insert("1.0", value or "")

        load_warnings = self._collect_apply_settings_warnings(data)

        loaded_run_type = str(data.get("run_type", self.run_type_var.get()) or self.run_type_var.get())
        if loaded_run_type not in RUN_TYPES:
            loaded_run_type = "Backtest"
        self.run_type_var.set(loaded_run_type)
        self.backend_var.set(data.get("backend", self.backend_var.get()))
        self.project_root_var.set(data.get("project_root", ""))
        self.python_exe_var.set(data.get("python_exe", sys.executable))
        self.userdir_var.set(data.get("userdir", ""))
        self.datadir_var.set(data.get("datadir", ""))
        self.logfile_var.set(data.get("logfile", ""))
        self.verbose_var.set(str(data.get("verbose", "0")))
        loaded_log_level = data.get("log_level")
        if loaded_log_level:
            self.log_level_var.set(str(loaded_log_level))
            self.verbose_var.set(LOG_LEVEL_TO_VERBOSE.get(str(loaded_log_level), self.verbose_var.get()))
        else:
            self.log_level_var.set(VERBOSE_TO_LOG_LEVEL.get(self.verbose_var.get(), "INFO"))
        self.no_color_var.set(bool(data.get("no_color", False)))
        self.config_editor.set_items(data.get("config_files", []))
        self.strategy_file_var.set(data.get("strategy_file", ""))
        self.strategy_class_var.set(data.get("strategy_class", ""))
        self.recursive_strategy_search_var.set(bool(data.get("recursive_strategy_search", False)))
        self.pair_mode_var.set(data.get("pair_mode", "config"))
        set_text(self.pairs_text, data.get("pairs", ""))
        set_text(self.blacklist_text, data.get("blacklist", ""))
        self.db_url_var.set(data.get("db_url", ""))
        self.fee_var.set(data.get("fee", ""))
        self.dry_run_wallet_var.set(data.get("dry_run_wallet", ""))
        self.stake_amount_var.set(data.get("stake_amount", ""))
        self.max_open_trades_var.set(data.get("max_open_trades", ""))
        self.timeframe_var.set(data.get("timeframe", ""))
        self.timerange_var.set(data.get("timerange", ""))
        self.timeframe_detail_var.set(data.get("timeframe_detail", ""))
        self.enable_protections_var.set(bool(data.get("enable_protections", False)))
        self.enable_position_stacking_var.set(bool(data.get("enable_position_stacking", False)))
        self.backtest_export_var.set(data.get("backtest_export", "default"))
        self.backtest_breakdown_var.set(data.get("backtest_breakdown", "none"))
        self.lookahead_min_trade_amount_var.set(str(data.get("lookahead_min_trade_amount", "") or ""))
        self.lookahead_targeted_trade_amount_var.set(str(data.get("lookahead_targeted_trade_amount", "") or ""))
        self.lookahead_export_filename_var.set(str(data.get("lookahead_export_filename", "") or ""))
        self.lookahead_allow_limit_orders_var.set(bool(data.get("lookahead_allow_limit_orders", False)))
        normalized_hyperopt_jobs = self._normalized_job_workers_value(data.get("hyperopt_jobs", ""))
        self.hyperopt_jobs_var.set(normalized_hyperopt_jobs)
        saved_hyperopt_multiplier = data.get("hyperopt_epoch_multiplier")
        if saved_hyperopt_multiplier is None:
            saved_hyperopt_multiplier = self._derive_multiplier_from_total_epochs(
                data.get("hyperopt_epochs", ""),
                normalized_hyperopt_jobs,
                "100",
            )
        self.hyperopt_epochs_var.set(str(saved_hyperopt_multiplier))
        self.hyperopt_spaces_var.set(data.get("hyperopt_spaces", "default"))
        self.hyperopt_random_state_var.set(data.get("hyperopt_random_state", ""))
        self.hyperopt_min_trades_var.set(data.get("hyperopt_min_trades", ""))
        self.hyperopt_loss_var.set(data.get("hyperopt_loss", ""))
        self.hyperopt_early_stop_var.set(data.get("hyperopt_early_stop", ""))
        self.hyperopt_ignore_missing_spaces_var.set(bool(data.get("hyperopt_ignore_missing_spaces", False)))
        self.hyperopt_disable_param_export_var.set(bool(data.get("hyperopt_disable_param_export", False)))
        self.hyperopt_analyze_per_epoch_var.set(bool(data.get("hyperopt_analyze_per_epoch", False)))
        self.hyperopt_print_all_var.set(bool(data.get("hyperopt_print_all", False)))
        self.hyperopt_monitor_results_var.set(bool(data.get("hyperopt_monitor_results", False)))
        self.hyperopt_results_file_var.set(data.get("hyperopt_results_file", ""))
        self.hyperopt_results_poll_ms_var.set(data.get("hyperopt_results_poll_ms", "1000"))
        explorer_sampling_seed = data.get("explorer_sampling_seed")
        if explorer_sampling_seed is None:
            explorer_sampling_seed = data.get("batch_sampling_seed")
        self.explorer_sampling_seed_var.set("" if explorer_sampling_seed is None else str(explorer_sampling_seed))
        self.explorer_max_loops_var.set(data.get("explorer_max_loops", data.get("batch_max_runs", "")))
        saved_explorer_multiplier = data.get("explorer_epoch_multiplier")
        if saved_explorer_multiplier is None:
            saved_explorer_multiplier = self._derive_multiplier_from_total_epochs(
                data.get("explorer_epochs", data.get("batch_epochs", "")),
                normalized_hyperopt_jobs,
                "",
            )
        self.explorer_epochs_var.set(str(saved_explorer_multiplier))
        self.explorer_random_state_var.set(data.get("explorer_random_state", data.get("batch_random_state", "")))
        self.explorer_backtest_workers_var.set(str(data.get("explorer_backtest_workers", "12") or "12"))
        explorer_windows = data.get("explorer_market_windows", data.get("explorer_source_windows", data.get("batch_windows")))
        if isinstance(explorer_windows, list):
            selected = {str(name) for name in explorer_windows}
            for name, var in self.explorer_window_vars.items():
                var.set(name in selected)
        self._apply_explorer_market_type_selection(
            "hyperopt",
            data.get(
                "explorer_hyperopt_market_types",
                data.get("hyperopt_market_types", data.get("batch_hyperopt_market_types", [])),
            ),
        )
        self._apply_explorer_market_type_selection(
            "backtest",
            data.get(
                "explorer_backtest_market_types",
                data.get("backtest_market_types", data.get("batch_backtest_market_types", [])),
            ),
        )
        for regime in EXPLORER_REGIME_VALUES:
            key = f"explorer_backtest_count_{regime}"
            self.explorer_backtest_count_vars[regime].set(str(data.get(key, "0")))
        holdout_values = data.get("explorer_12m_holdout_windows")
        if isinstance(holdout_values, list):
            selected_holdouts = {str(value) for value in holdout_values}
        else:
            selected_holdouts = {"20240101-20250101"}
        for label, var in self.explorer_12m_holdout_vars.items():
            var.set(label in selected_holdouts)
        saved_selection_mode = str(data.get("explorer_selection_mode", "Random families") or "Random families")
        selection_mode_aliases = {
            "random_families": "Random families",
            "random_tags": "Random tags",
            "random_namespace": "Random namespace values",
            "custom_batches": "Custom batches",
        }
        self.explorer_selection_mode_var.set(selection_mode_aliases.get(saved_selection_mode, saved_selection_mode if saved_selection_mode in EXPLORER_SELECTION_MODES else "Random families"))
        self.explorer_target_namespace_var.set(str(data.get("explorer_target_namespace", "mode") or "mode"))
        if "explorer_target_count" in data:
            count_value = data.get("explorer_target_count", "0")
        elif self.explorer_selection_mode_var.get() == "Random tags":
            count_value = data.get("explorer_tag_count", "0")
        else:
            count_value = data.get("explorer_family_count", "0")
        normalized_count = str(count_value).strip() or "0"
        self.explorer_target_count_var.set(normalized_count)
        self.explorer_family_count_var.set(normalized_count)
        self.explorer_tag_count_var.set(normalized_count)
        self.custom_batch_selected_run_ids = {str(batch_id) for batch_id in data.get("explorer_selected_custom_batch_ids", []) if str(batch_id)}
        self.explorer_min_param_count_var.set(data.get("explorer_min_param_count", "0"))
        self.explorer_keeper_enabled_var.set(bool(data.get("explorer_keeper_enabled", True)))
        self.explorer_keeper_save_dir_var.set(str(data.get("explorer_keeper_save_dir", "") or ""))
        self.explorer_keeper_win_numerator_var.set(str(data.get("explorer_keeper_win_numerator", "5") or "5"))
        self.explorer_keeper_win_denominator_var.set(str(data.get("explorer_keeper_win_denominator", "6") or "6"))
        self.explorer_keeper_min_profit_per_window_var.set(str(data.get("explorer_keeper_min_profit_per_window", "200") or "200"))
        self._refresh_explorer_namespace_dropdown()
        self._refresh_explorer_target_count_label()
        self.download_exchange_var.set(data.get("download_exchange", ""))
        self.download_pairs_file_var.set(data.get("download_pairs_file", ""))
        if self.download_pairs_text is not None:
            set_text(self.download_pairs_text, data.get("download_pairs", ""))
        self.download_timeframes_var.set(data.get("download_timeframes", ""))
        self.download_days_var.set(data.get("download_days", ""))
        self.download_new_pairs_days_var.set(data.get("download_new_pairs_days", ""))
        self.download_timerange_var.set(data.get("download_timerange", ""))
        self.download_trading_mode_var.set(data.get("download_trading_mode", ""))
        self.download_candle_types_var.set(data.get("download_candle_types", ""))
        self.download_data_format_ohlcv_var.set(data.get("download_data_format_ohlcv", ""))
        self.download_data_format_trades_var.set(data.get("download_data_format_trades", ""))
        self.download_include_inactive_var.set(bool(data.get("download_include_inactive", False)))
        self.download_no_parallel_var.set(bool(data.get("download_no_parallel", False)))
        self.download_dl_trades_var.set(bool(data.get("download_dl_trades", False)))
        self.download_convert_var.set(bool(data.get("download_convert", False)))
        self.download_erase_var.set(bool(data.get("download_erase", False)))
        self.download_prepend_var.set(bool(data.get("download_prepend", False)))
        self.frequi_url_var.set(data.get("frequi_url", "http://127.0.0.1:8080"))
        self.frequi_open_after_launch_var.set(bool(data.get("frequi_open_after_launch", True)))
        self.news_config_path_var.set(str(data.get("news_config_path", self.news_config_path_var.get()) or self.news_config_path_var.get()))
        self.news_data_dir_var.set(str(data.get("news_data_dir", self.news_data_dir_var.get()) or self.news_data_dir_var.get()))
        if "news_db_path" in data:
            self.news_db_path_var.set(str(data.get("news_db_path") or self.news_db_path_var.get()))
        else:
            self.news_db_path_var.set(str(Path(self.news_data_dir_var.get() or str(app_path(NEWS_DATA_DIR))) / NEWS_DB_FILE))
        self.news_interval_seconds_var.set(
            self._seconds_to_minutes_text(data.get("news_interval_seconds", self.news_interval_seconds_var.get()), 15 * 60)
        )
        self.news_once_var.set(bool(data.get("news_once", self.news_once_var.get())))
        self.web_config_path_var.set(str(data.get("web_config_path", self.web_config_path_var.get()) or self.web_config_path_var.get()))
        self.web_data_dir_var.set(str(data.get("web_data_dir", self.web_data_dir_var.get()) or self.web_data_dir_var.get()))
        if "web_db_path" in data:
            self.web_db_path_var.set(str(data.get("web_db_path") or self.web_db_path_var.get()))
        else:
            self.web_db_path_var.set(str(Path(self.web_data_dir_var.get() or str(app_path(WEB_DATA_DIR))) / WEB_DB_FILE))
        self.web_interval_seconds_var.set(
            self._seconds_to_minutes_text(data.get("web_interval_seconds", self.web_interval_seconds_var.get()), 360 * 60)
        )
        self.web_once_var.set(bool(data.get("web_once", self.web_once_var.get())))
        self.orderbook_config_path_var.set(str(data.get("orderbook_config_path", self.orderbook_config_path_var.get()) or self.orderbook_config_path_var.get()))
        self.orderbook_data_dir_var.set(str(data.get("orderbook_data_dir", self.orderbook_data_dir_var.get()) or self.orderbook_data_dir_var.get()))
        self.orderbook_exchange_var.set(str(data.get("orderbook_exchange", self.orderbook_exchange_var.get()) or self.orderbook_exchange_var.get()))
        self.orderbook_market_type_var.set(str(data.get("orderbook_market_type", self.orderbook_market_type_var.get()) or self.orderbook_market_type_var.get()))
        self.orderbook_depth_levels_var.set(str(data.get("orderbook_depth_levels", self.orderbook_depth_levels_var.get()) or self.orderbook_depth_levels_var.get()))
        self.orderbook_stream_update_ms_var.set(str(data.get("orderbook_stream_update_ms", self.orderbook_stream_update_ms_var.get()) or self.orderbook_stream_update_ms_var.get()))
        self.orderbook_metric_interval_seconds_var.set(str(data.get("orderbook_metric_interval_seconds", self.orderbook_metric_interval_seconds_var.get()) or self.orderbook_metric_interval_seconds_var.get()))
        self.orderbook_snapshot_interval_seconds_var.set(str(data.get("orderbook_snapshot_interval_seconds", self.orderbook_snapshot_interval_seconds_var.get()) or self.orderbook_snapshot_interval_seconds_var.get()))
        self.orderbook_store_snapshots_var.set(bool(data.get("orderbook_store_snapshots", self.orderbook_store_snapshots_var.get())))
        self.orderbook_capacity_warning_mb_var.set(str(data.get("orderbook_capacity_warning_mb", self.orderbook_capacity_warning_mb_var.get()) or self.orderbook_capacity_warning_mb_var.get()))
        self.orderbook_capacity_critical_mb_var.set(str(data.get("orderbook_capacity_critical_mb", self.orderbook_capacity_critical_mb_var.get()) or self.orderbook_capacity_critical_mb_var.get()))
        self.orderbook_max_symbols_var.set(str(data.get("orderbook_max_symbols", self.orderbook_max_symbols_var.get()) or self.orderbook_max_symbols_var.get()))
        self.review_hyperopt_file_var.set(data.get("review_hyperopt_file", ""))
        self.review_hyperopt_limit_var.set(data.get("review_hyperopt_limit", "20"))
        self.review_hyperopt_index_var.set(data.get("review_hyperopt_index", "-1"))
        self.review_hyperopt_best_var.set(bool(data.get("review_hyperopt_best", False)))
        self.review_hyperopt_profitable_var.set(bool(data.get("review_hyperopt_profitable", False)))
        self.review_hyperopt_print_json_var.set(bool(data.get("review_hyperopt_print_json", False)))
        self.review_hyperopt_no_header_var.set(bool(data.get("review_hyperopt_no_header", False)))
        self.review_hyperopt_no_details_var.set(bool(data.get("review_hyperopt_no_details", False)))
        self.review_hyperopt_breakdown_var.set(data.get("review_hyperopt_breakdown", "none"))
        self.review_backtest_file_var.set(data.get("review_backtest_file", ""))
        self.review_backtest_directory_var.set(data.get("review_backtest_directory", ""))
        self.review_backtest_show_pair_list_var.set(bool(data.get("review_backtest_show_pair_list", False)))
        self.review_backtest_breakdown_var.set(data.get("review_backtest_breakdown", "none"))
        self.review_analysis_groups_var.set(data.get("review_analysis_groups", "0 1 2 5"))
        self.review_enter_reasons_var.set(data.get("review_enter_reasons", ""))
        self.review_exit_reasons_var.set(data.get("review_exit_reasons", ""))
        self.review_indicator_list_var.set(data.get("review_indicator_list", ""))
        self.review_entry_only_var.set(bool(data.get("review_entry_only", False)))
        self.review_exit_only_var.set(bool(data.get("review_exit_only", False)))
        self.review_rejected_signals_var.set(bool(data.get("review_rejected_signals", False)))
        self.review_analysis_to_csv_var.set(bool(data.get("review_analysis_to_csv", False)))
        self.review_analysis_csv_path_var.set(data.get("review_analysis_csv_path", ""))

        strategy_file = self.strategy_file_var.get().strip()
        if strategy_file and os.path.isfile(strategy_file):
            classes = extract_class_names(strategy_file)
            self.strategy_class_combo["values"] = classes

        self.refresh_mode_options()
        self.refresh_all_command_previews()
        if hasattr(self, "catalog_tree"):
            self.refresh_custom_batch_catalog()
        if load_warnings:
            warning_text = "\n".join(f"- {item}" for item in load_warnings)
            self._append_console(f"Warning while loading {source_label}:\n{warning_text}\n")
            messagebox.showwarning(APP_TITLE, f"Some settings from {source_label} are no longer available and were skipped/defaulted.\n\n{warning_text}")

    def _normalized_job_workers_value(self, raw_value: Any) -> str:
        value = str(raw_value or "").strip()
        if not value:
            return ""
        try:
            parsed = int(value)
        except ValueError:
            return value
        if parsed >= 1:
            return str(parsed)
        cpu_total = max(1, int(os.cpu_count() or 1))
        # Legacy presets may store Freqtrade's relative syntax (-1=all, -2=all-1, ...).
        return str(max(1, cpu_total + 1 + parsed))

    def _parse_job_workers_or_none(self, raw_value: str) -> int | None:
        value = parse_int(raw_value.strip() or "")
        if value is None:
            return None
        if value < 1:
            raise ValueError("Job workers must be an integer >= 1 (direct core count).")
        return value

    def _effective_worker_count_for_epochs(self, explicit_workers: int | None) -> int:
        return explicit_workers if explicit_workers is not None else max(1, int(os.cpu_count() or 1))

    def _resolve_effective_epochs(self, multiplier_raw: str, effective_workers: int, field_label: str) -> int | None:
        multiplier = parse_int(multiplier_raw.strip() or "")
        if multiplier is None:
            return None
        if multiplier < 1:
            raise ValueError(f"{field_label} must be an integer >= 1.")
        return min(MAX_HYPEROPT_EPOCHS, effective_workers * multiplier)

    def _derive_multiplier_from_total_epochs(self, total_epochs_value: Any, workers_value: str, default_value: str) -> str:
        total_text = str(total_epochs_value or "").strip()
        try:
            total_epochs = int(total_text)
        except ValueError:
            return default_value
        if total_epochs < 1:
            return default_value
        try:
            explicit_workers = self._parse_job_workers_or_none(workers_value)
        except ValueError:
            explicit_workers = None
        effective_workers = self._effective_worker_count_for_epochs(explicit_workers)
        multiplier = max(1, (total_epochs + effective_workers - 1) // effective_workers)
        return str(multiplier)

    def _safe_effective_epochs_text(self, multiplier_raw: str, workers_raw: str) -> str:
        try:
            explicit_workers = self._parse_job_workers_or_none(workers_raw)
        except ValueError:
            explicit_workers = None
        try:
            epochs = self._resolve_effective_epochs(
                multiplier_raw,
                self._effective_worker_count_for_epochs(explicit_workers),
                "Epoch multiplier",
            )
        except ValueError:
            return ""
        return str(epochs) if epochs is not None else ""

    # ------------------------------------------------------------------
    # Command building
    # ------------------------------------------------------------------
    def refresh_mode_options(self) -> None:
        run_type = self.run_type_var.get()
        self.trade_frame.grid_remove()
        self.backtest_frame.grid_remove()
        self.lookahead_frame.grid_remove()
        self.hyperopt_frame.grid_remove()
        self.other_mode_frame.grid_remove()

        if run_type in ("Live", "Dry-run"):
            self.trade_frame.grid()
        elif run_type == "Backtest":
            self.backtest_frame.grid()
        elif run_type == "Lookahead Analysis":
            self.lookahead_frame.grid()
        elif run_type == "Hyperopt":
            self.hyperopt_frame.grid()
        else:
            self.other_mode_frame.grid()

        self.stop_button.configure(state="normal" if self.current_process else "disabled")
        if hasattr(self, "explorer_run_button"):
            self.explorer_run_button.configure(state="disabled" if (self.worker_thread and self.worker_thread.is_alive()) else "normal")
        if self.explorer_tab_run_button is not None:
            self.explorer_tab_run_button.configure(state="disabled" if (self.worker_thread and self.worker_thread.is_alive()) else "normal")
        if hasattr(self, "frequi_launch_button"):
            self.frequi_launch_button.configure(state="disabled" if (self.worker_thread and self.worker_thread.is_alive()) else "normal")

    def refresh_command_preview(self) -> None:
        try:
            build = self.build_command(preview_only=True)
            self.command_preview_var.set(shell_join(build.preview_command))
        except Exception as exc:
            self.command_preview_var.set(f"Invalid configuration: {exc}")

    def refresh_news_command_preview(self) -> None:
        try:
            command = self.build_news_collector_command(preview_only=True)
            self.news_command_preview_var.set(shell_join(command))
        except Exception as exc:
            self.news_command_preview_var.set(f"Invalid News Lab configuration: {exc}")

    def refresh_web_command_preview(self) -> None:
        try:
            command = self.build_web_collector_command(preview_only=True)
            self.web_command_preview_var.set(shell_join(command))
        except Exception as exc:
            self.web_command_preview_var.set(f"Invalid Web Lab configuration: {exc}")

    def build_explorer_command(self, preview_only: bool) -> tuple[list[str], str | None]:
        runner = app_path(EXPLORER_RUNNER_FILE)
        if not runner.exists():
            raise FileNotFoundError(runner)

        preset_name = "__launcher_explorer_current__"
        temp_preset_path: str | None
        if preview_only:
            temp_preset_path = "<temp_explorer_preset.json>"
        else:
            fd, temp_preset_path = tempfile.mkstemp(prefix="ft_explorer_preset_", suffix=".json")
            os.close(fd)
            payload = {preset_name: self.collect_settings()}
            with open(temp_preset_path, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)

        selected_windows = self.selected_explorer_windows()
        selected_records = self._explorer_selected_market_window_records()
        if not selected_windows:
            raise ValueError("Explorer needs at least one selected normal market window for HyperOpt.")
        hyperopt_market_types = self._explorer_effective_market_types("hyperopt")
        backtest_market_types = self._explorer_effective_market_types("backtest")
        hyperopt_candidates = [window for window in selected_records if str(window.get("regime") or "") in hyperopt_market_types]
        selected_holdouts = self._selected_12m_holdout_windows()
        if not hyperopt_candidates:
            raise ValueError("No eligible HyperOpt windows match the selected HyperOpt market types.")
        if not selected_holdouts:
            raise ValueError("Select at least one 12-month holdout backtest window. Holdout windows are required and are never used for HyperOpt.")

        python_exe = self.python_exe_var.get().strip() or sys.executable
        try:
            keeper_win_numerator = parse_int(self.explorer_keeper_win_numerator_var.get().strip() or "5")
            keeper_win_denominator = parse_int(self.explorer_keeper_win_denominator_var.get().strip() or "6")
            keeper_min_profit_per_window = float(self.explorer_keeper_min_profit_per_window_var.get().strip() or "200")
            backtest_workers = parse_int(self.explorer_backtest_workers_var.get().strip() or "12")
            hyperopt_jobs = self._parse_job_workers_or_none(self.hyperopt_jobs_var.get())
        except ValueError as exc:
            raise ValueError(f"Explorer runtime settings are invalid: {exc}") from exc
        if keeper_win_numerator is None or keeper_win_numerator <= 0:
            raise ValueError("Keeper win ratio numerator must be greater than 0.")
        if keeper_win_denominator is None or keeper_win_denominator <= 0:
            raise ValueError("Keeper win ratio denominator must be greater than 0.")
        if keeper_min_profit_per_window < 0:
            raise ValueError("Keeper min profit per backtest window must be >= 0.")
        if backtest_workers is None or backtest_workers < 1:
            raise ValueError("Backtest workers must be an integer >= 1.")
        effective_hyperopt_workers = self._effective_worker_count_for_epochs(hyperopt_jobs)
        explorer_epochs = self._resolve_effective_epochs(
            self.explorer_epochs_var.get(),
            effective_hyperopt_workers,
            "Explorer epoch multiplier",
        )

        if self.explorer_selection_mode_var.get() == "Custom batches":
            selection_mode = "custom_batches"
            if not self.custom_batch_selected_run_ids:
                raise ValueError("No custom batches selected. Select at least one committed batch before running Explorer.")
        elif self.explorer_selection_mode_var.get() == "Random tags":
            selection_mode = "random_tags"
        elif self.explorer_selection_mode_var.get() == "Random namespace values":
            selection_mode = "random_namespace"
            if not self.explorer_target_namespace_var.get().strip():
                raise ValueError("Select a namespace for Random namespace values mode.")
        else:
            selection_mode = "random_families"
        hyperopt_market_types_arg = ",".join(value for value in EXPLORER_REGIME_VALUES if value in hyperopt_market_types)
        backtest_market_types_arg = ",".join(value for value in EXPLORER_REGIME_VALUES if value in backtest_market_types)
        target_count = self.explorer_target_count_var.get().strip() or "0"
        normalized_target_count = str(parse_int(target_count) or 0)
        backtest_count_requests = self._explorer_backtest_count_requests()
        strategy_file = Path(self.strategy_file_var.get().strip() or str(app_path(DEFAULT_STRATEGY_FILE)))
        if not strategy_file.is_absolute():
            project_root = Path(self.project_root_var.get().strip() or Path.cwd())
            strategy_file = project_root / strategy_file
        strategy_param_file = strategy_file.with_suffix(".json").resolve()
        command = [
            python_exe,
            "-u",
            str(runner),
            "--preset-file",
            temp_preset_path,
            "--preset",
            preset_name,
            "--market-windows-file",
            str(app_path(EXPLORER_MARKET_WINDOWS_FILE).resolve()),
            "--strategy-param-file",
            str(strategy_param_file),
            "--metadata-file",
            str((app_path(EXPLORER_METADATA_DIR) / EXPLORER_LATEST_SUMMARY_FILE).resolve()),
            "--selection-mode",
            selection_mode,
            "--source-windows",
            *selected_windows,
        ]
        if hyperopt_market_types != set(EXPLORER_REGIME_VALUES):
            append_if_value(command, "--hyperopt-market-types", hyperopt_market_types_arg)
        if backtest_market_types != set(EXPLORER_REGIME_VALUES):
            append_if_value(command, "--backtest-market-types", backtest_market_types_arg)
        append_if_value(command, "--backtest-counts-json", json.dumps(backtest_count_requests, separators=(",", ":")))
        append_if_value(command, "--holdout-windows-json", json.dumps(selected_holdouts, separators=(",", ":")))
        if selection_mode == "custom_batches":
            append_if_value(command, "--custom-batches-file", str(self._custom_batches_path().resolve()))
            append_if_value(command, "--custom-batch-ids", ",".join(sorted(self.custom_batch_selected_run_ids)))
        elif selection_mode == "random_tags":
            append_if_value(command, "--tag-count", normalized_target_count)
        elif selection_mode == "random_namespace":
            append_if_value(command, "--target-namespace", self.explorer_target_namespace_var.get().strip())
            append_if_value(command, "--tag-count", normalized_target_count)
        else:
            append_if_value(command, "--family-count", normalized_target_count)
        if selection_mode != "custom_batches":
            append_if_value(command, "--min-param-count-per-hyper-run", self.explorer_min_param_count_var.get())
        append_if_value(command, "--sampling-seed", self.explorer_sampling_seed_var.get())
        append_if_value(command, "--max-loops", self.explorer_max_loops_var.get())
        if explorer_epochs is not None:
            command.extend(["--epochs", str(explorer_epochs)])
        append_if_value(command, "--random-state", self.explorer_random_state_var.get())
        command.extend(["--backtest-workers", str(backtest_workers)])
        command.extend(["--temp-backtest-root", str(app_path("../runtime/tempbacktest").resolve())])
        if not self.explorer_keeper_enabled_var.get():
            command.append("--keeper-disable")
        append_if_value(command, "--keeper-save-dir", self.explorer_keeper_save_dir_var.get())
        command.extend(["--keeper-win-numerator", str(keeper_win_numerator)])
        command.extend(["--keeper-win-denominator", str(keeper_win_denominator)])
        command.extend(["--keeper-min-profit-per-window", str(keeper_min_profit_per_window)])
        return command, None if preview_only else temp_preset_path

    def build_frequi_command(self, preview_only: bool = False) -> BuildResult:
        args: list[str] = ["webserver"]

        config_files = [config_file.strip() for config_file in self.config_editor.get_items() if config_file.strip()]
        self._validate_config_files(config_files)
        self._validate_frequi_config(config_files)
        has_config_file = bool(config_files)
        if not preview_only and not has_config_file and not self.userdir_var.get().strip():
            raise ValueError("FreqUI needs at least one config file, or a user data dir where Freqtrade can find config.")

        self.verbose_var.set(LOG_LEVEL_TO_VERBOSE.get(self.log_level_var.get(), self.verbose_var.get()))
        for _ in range(parse_int(self.verbose_var.get()) or 0):
            args.append("-v")
        if self.no_color_var.get():
            args.append("--no-color")
        if self.logfile_var.get().strip():
            args.extend(["--logfile", self.logfile_var.get().strip()])

        for config_file in config_files:
            args.extend(["-c", config_file])

        if self.datadir_var.get().strip():
            args.extend(["--datadir", self.datadir_var.get().strip()])
        if self.userdir_var.get().strip():
            args.extend(["--userdir", self.userdir_var.get().strip()])

        strategy_file = self.strategy_file_var.get().strip()
        strategy_class = self.strategy_class_var.get().strip()
        if strategy_file:
            args.extend(["--strategy-path", str(Path(strategy_file).resolve().parent)])
        if strategy_class:
            args.extend(["--strategy", strategy_class])
        if self.recursive_strategy_search_var.get():
            args.append("--recursive-strategy-search")

        temp_config_path: str | None = None

        python_for_preview = self.python_exe_var.get().strip() or sys.executable
        preview_command = [python_for_preview, "-u", "-m", "freqtrade", *args]
        return BuildResult(command_args=args, preview_command=preview_command, temp_config_path=temp_config_path)

    def _validate_config_files(self, config_files: list[str]) -> None:
        from freqtrade.configuration.load_config import load_config_file

        for config_file in config_files:
            path = Path(config_file).expanduser()
            if not path.is_file():
                raise ValueError(f"Config file does not exist: {path}")
            try:
                load_config_file(str(path))
            except Exception as exc:
                raise ValueError(f"Invalid config file {path}: {exc}") from exc

    def _validate_frequi_config(self, config_files: list[str]) -> None:
        from freqtrade.configuration.load_config import load_from_files

        if not config_files:
            return

        try:
            merged_config = load_from_files(config_files)
        except Exception as exc:
            raise ValueError(f"Could not load FreqUI config files: {exc}") from exc

        api_server = merged_config.get("api_server")
        if not isinstance(api_server, dict):
            raise ValueError(
                "FreqUI requires an `api_server` section in the merged config. "
                "Add `api_server.enabled`, `listen_ip_address`, `listen_port`, `username`, and `password`."
            )

        required_fields = ["enabled", "listen_ip_address", "listen_port", "username", "password"]
        missing_fields = [field for field in required_fields if field not in api_server]
        if missing_fields:
            raise ValueError(
                "FreqUI config is missing required `api_server` fields: "
                + ", ".join(missing_fields)
            )

    def build_command(self, preview_only: bool) -> BuildResult:

        run_type = self.run_type_var.get()
        if run_type not in RUN_TYPE_TO_COMMAND:
            run_type = "Backtest"
            self.run_type_var.set(run_type)
        command = RUN_TYPE_TO_COMMAND[run_type]
        args: list[str] = [command]
        config_files = [config_file.strip() for config_file in self.config_editor.get_items() if config_file.strip()]

        self.verbose_var.set(LOG_LEVEL_TO_VERBOSE.get(self.log_level_var.get(), self.verbose_var.get()))
        for _ in range(parse_int(self.verbose_var.get()) or 0):
            args.append("-v")
        if self.no_color_var.get():
            args.append("--no-color")
        if self.logfile_var.get().strip():
            args.extend(["--logfile", self.logfile_var.get().strip()])

        for config_file in config_files:
            args.extend(["-c", config_file])

        if self.datadir_var.get().strip():
            args.extend(["--datadir", self.datadir_var.get().strip()])
        if self.userdir_var.get().strip():
            args.extend(["--userdir", self.userdir_var.get().strip()])

        strategy_file = self.strategy_file_var.get().strip()
        strategy_class = self.strategy_class_var.get().strip()
        if run_type != "Download Data":
            if strategy_file:
                args.extend(["--strategy-path", str(Path(strategy_file).resolve().parent)])
            if strategy_class:
                args.extend(["--strategy", strategy_class])
            if self.recursive_strategy_search_var.get():
                args.append("--recursive-strategy-search")

        temp_config_path: str | None = None
        manual_pairs = self.pair_mode_var.get() == "manual"
        pairs = parse_token_list(self.pairs_text.get("1.0", tk.END))
        blacklist = parse_token_list(self.blacklist_text.get("1.0", tk.END))
        download_pairs = parse_token_list(self.download_pairs_text.get("1.0", tk.END)) if self.download_pairs_text is not None else []

        if run_type in ("Live", "Dry-run"):
            if self.db_url_var.get().strip():
                args.extend(["--db-url", self.db_url_var.get().strip()])
            if run_type == "Dry-run":
                args.append("--dry-run")
            append_if_value(args, "--dry-run-wallet", self.dry_run_wallet_var.get())
            append_if_value(args, "--fee", self.fee_var.get())

            if manual_pairs and pairs:
                overlay = {
                    "exchange": {
                        "pair_whitelist": pairs,
                    }
                }
                if blacklist:
                    overlay["exchange"]["pair_blacklist"] = blacklist

                if preview_only:
                    temp_config_path = "<temp_pair_override.json>"
                else:
                    fd, temp_config_path = tempfile.mkstemp(prefix="ft_pairs_", suffix=".json")
                    os.close(fd)
                    with open(temp_config_path, "w", encoding="utf-8") as handle:
                        json.dump(overlay, handle, indent=2)
                args.extend(["-c", temp_config_path])

        elif run_type == "Backtest":
            append_if_value(args, "-i", self.timeframe_var.get())
            append_if_value(args, "--timerange", self.timerange_var.get())
            append_if_value(args, "--timeframe-detail", self.timeframe_detail_var.get())
            append_if_value(args, "--max-open-trades", self.max_open_trades_var.get())
            append_if_value(args, "--stake-amount", self.stake_amount_var.get())
            append_if_value(args, "--fee", self.fee_var.get())
            append_if_value(args, "--dry-run-wallet", self.dry_run_wallet_var.get())
            effective_download_pairs = download_pairs or (pairs if manual_pairs else [])
            if effective_download_pairs:
                args.extend(["-p", *effective_download_pairs])
            if self.enable_protections_var.get():
                args.append("--enable-protections")
            if self.enable_position_stacking_var.get():
                args.append("--eps")
            export_value = self.backtest_export_var.get().strip()
            if export_value and export_value != "default":
                args.extend(["--export", export_value])
            breakdown_value = self.backtest_breakdown_var.get().strip()
            if breakdown_value and breakdown_value != "none":
                args.extend(["--breakdown", breakdown_value])

        elif run_type == "Lookahead Analysis":
            append_if_value(args, "-i", self.timeframe_var.get())
            append_if_value(args, "--timerange", self.timerange_var.get())
            append_if_value(args, "--timeframe-detail", self.timeframe_detail_var.get())
            append_if_value(args, "--max-open-trades", self.max_open_trades_var.get())
            append_if_value(args, "--stake-amount", self.stake_amount_var.get())
            append_if_value(args, "--fee", self.fee_var.get())
            append_if_value(args, "--dry-run-wallet", self.dry_run_wallet_var.get())
            if pairs:
                args.extend(["-p", *pairs])
            if self.enable_protections_var.get():
                args.append("--enable-protections")
            if self.enable_position_stacking_var.get():
                args.append("--eps")
            append_if_value(args, "--minimum-trade-amount", self.lookahead_min_trade_amount_var.get())
            append_if_value(args, "--targeted-trade-amount", self.lookahead_targeted_trade_amount_var.get())
            append_if_value(args, "--lookahead-analysis-exportfilename", self.lookahead_export_filename_var.get())
            if self.lookahead_allow_limit_orders_var.get():
                args.append("--allow-limit-orders")

        elif run_type == "Hyperopt":
            append_if_value(args, "-i", self.timeframe_var.get())
            append_if_value(args, "--timerange", self.timerange_var.get())
            append_if_value(args, "--timeframe-detail", self.timeframe_detail_var.get())
            append_if_value(args, "--max-open-trades", self.max_open_trades_var.get())
            append_if_value(args, "--stake-amount", self.stake_amount_var.get())
            append_if_value(args, "--fee", self.fee_var.get())
            append_if_value(args, "--dry-run-wallet", self.dry_run_wallet_var.get())
            if manual_pairs and pairs:
                args.extend(["-p", *pairs])
            if self.enable_protections_var.get():
                args.append("--enable-protections")
            if self.enable_position_stacking_var.get():
                args.append("--eps")
            hyperopt_jobs = self._parse_job_workers_or_none(self.hyperopt_jobs_var.get())
            effective_hyperopt_workers = self._effective_worker_count_for_epochs(hyperopt_jobs)
            effective_epochs = self._resolve_effective_epochs(
                self.hyperopt_epochs_var.get(),
                effective_hyperopt_workers,
                "Epoch multiplier",
            )
            if effective_epochs is not None:
                args.extend(["-e", str(effective_epochs)])
            spaces = parse_token_list(self.hyperopt_spaces_var.get())
            if spaces:
                args.extend(["--spaces", *spaces])
            if hyperopt_jobs is not None:
                args.extend(["-j", str(hyperopt_jobs)])
            append_if_value(args, "--random-state", self.hyperopt_random_state_var.get())
            append_if_value(args, "--min-trades", self.hyperopt_min_trades_var.get())
            append_if_value(args, "--hyperopt-loss", self.hyperopt_loss_var.get())
            append_if_value(args, "--early-stop", self.hyperopt_early_stop_var.get())
            if self.hyperopt_ignore_missing_spaces_var.get():
                args.append("--ignore-missing-spaces")
            if self.hyperopt_disable_param_export_var.get():
                args.append("--disable-param-export")
            if self.hyperopt_analyze_per_epoch_var.get():
                args.append("--analyze-per-epoch")
            if self.hyperopt_print_all_var.get():
                args.append("--print-all")

        elif run_type == "Download Data":
            append_if_value(args, "--exchange", self.download_exchange_var.get())
            append_if_value(args, "--pairs-file", self.download_pairs_file_var.get())
            timeframes = parse_token_list(self.download_timeframes_var.get())
            if timeframes:
                args.extend(["-t", *timeframes])
            append_if_value(args, "--days", self.download_days_var.get())
            append_if_value(args, "--new-pairs-days", self.download_new_pairs_days_var.get())
            append_if_value(args, "--timerange", self.download_timerange_var.get())
            append_if_value(args, "--trading-mode", self.download_trading_mode_var.get())
            candle_types = parse_token_list(self.download_candle_types_var.get())
            if candle_types:
                args.extend(["--candle-types", *candle_types])
            append_if_value(args, "--data-format-ohlcv", self.download_data_format_ohlcv_var.get())
            append_if_value(args, "--data-format-trades", self.download_data_format_trades_var.get())
            effective_download_pairs = download_pairs or (pairs if manual_pairs else [])
            if effective_download_pairs:
                args.extend(["-p", *effective_download_pairs])
            if self.download_include_inactive_var.get():
                args.append("--include-inactive-pairs")
            if self.download_no_parallel_var.get():
                args.append("--no-parallel-download")
            if self.download_dl_trades_var.get():
                args.append("--dl-trades")
            if self.download_convert_var.get():
                args.append("--convert")
            if self.download_erase_var.get():
                args.append("--erase")
            if self.download_prepend_var.get():
                args.append("--prepend")

        python_for_preview = self.python_exe_var.get().strip() or sys.executable
        preview_command = [python_for_preview, "-m", "freqtrade", *args]
        return BuildResult(command_args=args, preview_command=preview_command, temp_config_path=temp_config_path)

    def _collector_config_path(self, profile: CollectorProfile) -> Path:
        config_var = getattr(self, f"{profile.key}_config_path_var")
        return Path(config_var.get().strip() or app_path(profile.sources_file))

    def _collector_data_dir(self, profile: CollectorProfile) -> Path:
        data_dir_var = getattr(self, f"{profile.key}_data_dir_var")
        return Path(data_dir_var.get().strip() or app_path(profile.data_dir_name))

    def _collector_db_path(self, profile: CollectorProfile) -> Path:
        db_var = getattr(self, f"{profile.key}_db_path_var")
        text = db_var.get().strip()
        if text:
            return Path(text)
        return self._collector_data_dir(profile) / profile.db_file

    def _collector_status_path(self, profile: CollectorProfile) -> Path:
        return self._collector_data_dir(profile) / profile.status_file

    def _collector_pid_path(self, profile: CollectorProfile) -> Path:
        return self._collector_data_dir(profile) / profile.pid_file

    def _collector_stop_path(self, profile: CollectorProfile) -> Path:
        return self._collector_data_dir(profile) / profile.stop_file

    def _collector_log_path(self, profile: CollectorProfile) -> Path:
        return self._collector_data_dir(profile) / "logs" / profile.log_file

    def _news_config_path(self) -> Path:
        return self._collector_config_path(NEWS_COLLECTOR_PROFILE)

    def _news_data_dir(self) -> Path:
        return self._collector_data_dir(NEWS_COLLECTOR_PROFILE)

    def _news_db_path(self) -> Path:
        return self._collector_db_path(NEWS_COLLECTOR_PROFILE)

    def _news_status_path(self) -> Path:
        return self._collector_status_path(NEWS_COLLECTOR_PROFILE)

    def _news_pid_path(self) -> Path:
        return self._collector_pid_path(NEWS_COLLECTOR_PROFILE)

    def _news_stop_path(self) -> Path:
        return self._collector_stop_path(NEWS_COLLECTOR_PROFILE)

    def _news_log_path(self) -> Path:
        return self._collector_log_path(NEWS_COLLECTOR_PROFILE)

    def _web_config_path(self) -> Path:
        return self._collector_config_path(WEB_COLLECTOR_PROFILE)

    def _web_data_dir(self) -> Path:
        return self._collector_data_dir(WEB_COLLECTOR_PROFILE)

    def _web_db_path(self) -> Path:
        return self._collector_db_path(WEB_COLLECTOR_PROFILE)

    def _web_status_path(self) -> Path:
        return self._collector_status_path(WEB_COLLECTOR_PROFILE)

    def _web_pid_path(self) -> Path:
        return self._collector_pid_path(WEB_COLLECTOR_PROFILE)

    def _web_stop_path(self) -> Path:
        return self._collector_stop_path(WEB_COLLECTOR_PROFILE)

    def _web_log_path(self) -> Path:
        return self._collector_log_path(WEB_COLLECTOR_PROFILE)

    def _orderbook_paths(self) -> dict[str, Path]:
        data_dir = Path(self.orderbook_data_dir_var.get().strip() or app_path(ORDERBOOK_DATA_DIR))
        config_path = Path(self.orderbook_config_path_var.get().strip() or app_path(ORDERBOOK_SOURCES_FILE))
        return {
            "collector": app_path(ORDERBOOK_COLLECTOR_FILE),
            "config": config_path,
            "data_dir": data_dir,
            "db": data_dir / ORDERBOOK_DB_FILE,
            "status": data_dir / ORDERBOOK_STATUS_FILE,
            "pid": data_dir / ORDERBOOK_PID_FILE,
            "stop": data_dir / ORDERBOOK_STOP_FILE,
            "log": data_dir / "logs" / ORDERBOOK_LOG_FILE,
        }

    def _orderbook_raw_pairs(self) -> list[str]:
        return parse_token_list(self.pairs_text.get("1.0", tk.END))

    def _orderbook_resolved_pairs(self) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        raw_pairs = self._orderbook_raw_pairs()
        max_symbols = max(1, parse_int(self.orderbook_max_symbols_var.get().strip() or "12") or 12)
        valid = normalize_whitelist_pairs(raw_pairs, max_symbols=max_symbols)
        seen = {item["symbol"] for item in valid}
        preview_rows: list[dict[str, str]] = []
        added_valid = set()
        for pair in raw_pairs:
            symbol = normalize_freqtrade_pair_to_binance_symbol(pair)
            if not symbol:
                preview_rows.append({"pair": pair, "symbol": "-", "status": "invalid"})
                continue
            if symbol in added_valid:
                preview_rows.append({"pair": pair, "symbol": symbol, "status": "duplicate"})
                continue
            if symbol not in seen:
                preview_rows.append({"pair": pair, "symbol": symbol, "status": "max_symbols_limit"})
                continue
            preview_rows.append({"pair": pair, "symbol": symbol, "status": "ok"})
            added_valid.add(symbol)
        return valid, preview_rows

    def refresh_orderbook_pair_preview(self) -> None:
        valid, preview_rows = self._orderbook_resolved_pairs()
        tree = self.orderbook_pair_preview_tree
        if tree is not None:
            tree.delete(*tree.get_children())
            for idx, row in enumerate(preview_rows):
                tree.insert("", "end", iid=f"ob_pair_{idx}", values=(row["pair"], row["symbol"], row["status"]))
        self.orderbook_preview_pair_count_var.set(f"Whitelist pairs: {len(self._orderbook_raw_pairs())}")
        self.orderbook_preview_symbol_count_var.set(f"Active symbols: {len(valid)}")
        if valid:
            self.orderbook_pair_warning_var.set("")
        else:
            self.orderbook_pair_warning_var.set("No usable whitelist pairs found. Add pairs on the Pairs tab first.")
        self.refresh_orderbook_estimate()

    def on_orderbook_normalize_whitelist(self) -> None:
        self.normalize_pair_text(self.pairs_text)
        self.refresh_orderbook_pair_preview()

    def refresh_orderbook_estimate(self) -> None:
        valid, _ = self._orderbook_resolved_pairs()
        interval = max(1, parse_int(self.orderbook_metric_interval_seconds_var.get().strip() or "1") or 1)
        snapshot = max(1, parse_int(self.orderbook_snapshot_interval_seconds_var.get().strip() or "60") or 60)
        depth = max(1, parse_int(self.orderbook_depth_levels_var.get().strip() or "20") or 20)
        estimate = estimate_storage_usage(
            pair_count=len(valid),
            metric_interval_seconds=interval,
            snapshot_interval_seconds=snapshot,
            depth_levels=depth,
            store_snapshots=bool(self.orderbook_store_snapshots_var.get()),
        )
        self.orderbook_estimated_metric_rows_var.set(f"{estimate['metric_rows_per_day']:.0f}")
        self.orderbook_estimated_snapshot_rows_var.set(f"{estimate['snapshot_rows_per_day']:.0f}")
        self.orderbook_estimated_mb_per_day_var.set(f"{estimate['estimated_total_mb_per_day']:.2f}")
        warning = float(parse_int(self.orderbook_capacity_warning_mb_var.get().strip() or "500") or 500)
        mb_day = float(estimate["estimated_total_mb_per_day"])
        if mb_day <= 0:
            self.orderbook_estimated_days_to_warning_var.set("-")
        else:
            self.orderbook_estimated_days_to_warning_var.set(f"{warning / mb_day:.1f}")

    def build_orderbook_collector_command(self) -> list[str]:
        paths = self._orderbook_paths()
        collector = paths["collector"]
        if not collector.exists():
            raise FileNotFoundError(collector)
        valid, _ = self._orderbook_resolved_pairs()
        if not valid:
            raise ValueError("No usable whitelist pairs found. Add pairs on the Pairs tab first.")
        command = [
            self.python_exe_var.get().strip() or sys.executable,
            "-u",
            str(collector),
            "--config",
            str(paths["config"]),
            "--data-dir",
            str(paths["data_dir"]),
            "--db",
            str(paths["db"]),
            "--status-file",
            str(paths["status"]),
            "--pid-file",
            str(paths["pid"]),
            "--stop-file",
            str(paths["stop"]),
            "--log-file",
            str(paths["log"]),
            "--pairs",
            ",".join(item["pair"] for item in valid),
        ]
        return command

    def _build_orderbook_runtime_config_payload(self) -> dict[str, Any]:
        exchange_text = (self.orderbook_exchange_var.get().strip() or "Binance USD-M Futures").lower()
        exchange = "binance_usdm_futures" if "binance" in exchange_text else "binance_usdm_futures"
        market_type = (self.orderbook_market_type_var.get().strip() or "futures").lower()
        if market_type not in {"futures"}:
            market_type = "futures"

        depth_levels = parse_int(self.orderbook_depth_levels_var.get().strip() or "20")
        if depth_levels not in {5, 10, 20}:
            raise ValueError("Depth must be one of: 5, 10, 20.")
        stream_update_ms = parse_int(self.orderbook_stream_update_ms_var.get().strip() or "500")
        if stream_update_ms not in {100, 250, 500}:
            raise ValueError("Stream update ms must be one of: 100, 250, 500.")

        metric_interval_seconds = parse_int(self.orderbook_metric_interval_seconds_var.get().strip() or "1")
        if metric_interval_seconds is None or metric_interval_seconds <= 0:
            raise ValueError("Metric interval seconds must be a positive integer.")
        snapshot_interval_seconds = parse_int(self.orderbook_snapshot_interval_seconds_var.get().strip() or "60")
        if snapshot_interval_seconds is None or snapshot_interval_seconds <= 0:
            raise ValueError("Snapshot interval seconds must be a positive integer.")
        max_symbols = parse_int(self.orderbook_max_symbols_var.get().strip() or "12")
        if max_symbols is None or max_symbols <= 0:
            raise ValueError("Max symbols must be a positive integer.")
        warning_mb = parse_int(self.orderbook_capacity_warning_mb_var.get().strip() or "500")
        if warning_mb is None or warning_mb <= 0:
            raise ValueError("Capacity warning MB must be a positive integer.")
        critical_mb = parse_int(self.orderbook_capacity_critical_mb_var.get().strip() or "2000")
        if critical_mb is None or critical_mb <= 0:
            raise ValueError("Capacity critical MB must be a positive integer.")

        return {
            "exchange": exchange,
            "market_type": market_type,
            "stream_mode": "partial_depth",
            "depth_levels": int(depth_levels),
            "stream_update_ms": int(stream_update_ms),
            "metric_interval_seconds": int(metric_interval_seconds),
            "snapshot_interval_seconds": int(snapshot_interval_seconds),
            "store_snapshots": bool(self.orderbook_store_snapshots_var.get()),
            "max_symbols": int(max_symbols),
            "capacity_warning_mb": int(warning_mb),
            "capacity_critical_mb": int(critical_mb),
        }

    def _write_orderbook_runtime_config(self) -> Path:
        config_path = Path(self.orderbook_config_path_var.get().strip() or app_path(ORDERBOOK_SOURCES_FILE))
        payload: dict[str, Any] = {}
        if config_path.exists():
            try:
                loaded = json.loads(config_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    payload = loaded
            except Exception:
                payload = {}
        payload.setdefault("version", 1)
        payload.update(self._build_orderbook_runtime_config_payload())
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        self.orderbook_config_path_var.set(str(config_path))
        return config_path

    def start_orderbook_collector(self) -> None:
        try:
            self._write_orderbook_runtime_config()
            command = self.build_orderbook_collector_command()
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Cannot start Order Book Lab:\n{exc}")
            return
        paths = self._orderbook_paths()
        paths["data_dir"].mkdir(parents=True, exist_ok=True)
        paths["log"].parent.mkdir(parents=True, exist_ok=True)
        try:
            paths["stop"].unlink(missing_ok=True)
        except Exception:
            pass
        if paths["pid"].exists():
            try:
                stale_pid = int(paths["pid"].read_text(encoding="utf-8").strip())
                if self._is_process_running(stale_pid):
                    messagebox.showinfo(APP_TITLE, f"Order Book collector already running (PID {stale_pid}).")
                    return
            except Exception:
                pass
        try:
            with open(paths["log"], "ab") as log_handle:
                kwargs: dict[str, Any] = {
                    "stdin": subprocess.DEVNULL,
                    "stdout": log_handle,
                    "stderr": subprocess.STDOUT,
                    "cwd": str(Path(__file__).resolve().parent),
                    "env": utf8_subprocess_env(),
                    "close_fds": True,
                }
                if os.name == "nt":
                    kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
                else:
                    kwargs["start_new_session"] = True
                process = subprocess.Popen(command, **kwargs)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not start Order Book collector:\n{exc}")
            return
        self.orderbook_status_var.set(f"started detached PID {process.pid}")
        self.orderbook_pid_var.set(str(process.pid))
        self._append_console(f"Started Order Book collector detached: PID {process.pid}\n")
        self.refresh_orderbook_status(show_popup=False)

    def request_orderbook_stop(self) -> None:
        stop_file = self._orderbook_paths()["stop"]
        try:
            stop_file.parent.mkdir(parents=True, exist_ok=True)
            stop_file.write_text(utc_now() + "\n", encoding="utf-8")
            self.orderbook_status_var.set("stop requested")
            self._append_console(f"Order Book stop requested via {stop_file}\n")
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not request Order Book stop:\n{exc}")

    def _read_orderbook_status_payload(self) -> dict[str, Any]:
        status_file = self._orderbook_paths()["status"]
        if not status_file.exists():
            return {}
        try:
            payload = json.loads(status_file.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _refresh_orderbook_latest_metrics_table(self) -> None:
        tree = self.orderbook_latest_metrics_tree
        if tree is None:
            return
        tree.delete(*tree.get_children())
        db_path = self._orderbook_paths()["db"]
        if not db_path.exists():
            return
        try:
            with sqlite3.connect(str(db_path), timeout=5.0) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT pair, symbol, status, best_bid, best_ask, spread_bps, imbalance_top20,
                           bid_pressure_ratio_60s, ask_pressure_ratio_60s,
                           nearest_bid_wall_distance_bps, nearest_ask_wall_distance_bps, last_metric_at
                    FROM stream_status
                    ORDER BY pair
                    """
                ).fetchall()
            for row in rows:
                tree.insert(
                    "",
                    "end",
                    values=(
                        row["pair"],
                        row["symbol"],
                        row["status"],
                        row["best_bid"],
                        row["best_ask"],
                        row["spread_bps"],
                        row["imbalance_top20"],
                        row["bid_pressure_ratio_60s"],
                        row["ask_pressure_ratio_60s"],
                        row["nearest_bid_wall_distance_bps"],
                        row["nearest_ask_wall_distance_bps"],
                        row["last_metric_at"],
                    ),
                )
        except Exception:
            return

    def _maybe_orderbook_capacity_popup(self, status: dict[str, Any], show_popup: bool) -> None:
        if not show_popup:
            return
        level = str(status.get("capacity_level") or "ok").lower()
        if level not in {"warning", "critical"}:
            if level == "ok":
                self._orderbook_last_capacity_popup_level = "ok"
            return
        if level == self._orderbook_last_capacity_popup_level:
            return
        data_dir_mb = status.get("data_dir_mb")
        estimated = status.get("estimated_mb_per_day")
        if level == "warning":
            messagebox.showwarning(
                APP_TITLE,
                "Order Book Lab data size is above the configured warning threshold. "
                f"Current data dir: {data_dir_mb} MB. Estimated growth: {estimated} MB/day.",
            )
        else:
            messagebox.showwarning(
                APP_TITLE,
                "Order Book Lab data size is above the configured critical threshold. "
                "Consider stopping the collector or increasing retention/storage settings.",
            )
        self._orderbook_last_capacity_popup_level = level

    def refresh_orderbook_status(self, show_popup: bool = True) -> None:
        status = self._read_orderbook_status_payload()
        pid_path = self._orderbook_paths()["pid"]
        pid_text = "-"
        if pid_path.exists():
            try:
                pid_text = pid_path.read_text(encoding="utf-8").strip() or "-"
            except Exception:
                pid_text = "-"
        elif status.get("pid") is not None:
            pid_text = str(status.get("pid"))
        status_text = str(status.get("status") or "unknown")
        if pid_text not in {"", "-"}:
            try:
                pid_int = int(pid_text)
                if not self._is_process_running(pid_int) and status_text.lower() == "running":
                    status_text = "stale/unknown"
            except Exception:
                pass
        self.orderbook_status_var.set(status_text)
        self.orderbook_pid_var.set(pid_text)
        self.orderbook_started_at_var.set(str(status.get("started_at") or "-"))
        self.orderbook_heartbeat_at_var.set(str(status.get("heartbeat_at") or "-"))
        self.orderbook_last_message_at_var.set(str(status.get("last_message_at") or "-"))
        self.orderbook_last_metric_at_var.set(str(status.get("last_metric_at") or "-"))
        self.orderbook_pair_count_var.set(str(status.get("pair_count") if status.get("pair_count") is not None else "-"))
        self.orderbook_active_streams_var.set(str(status.get("active_streams") if status.get("active_streams") is not None else "-"))
        self.orderbook_message_count_var.set(str(status.get("message_count_total") if status.get("message_count_total") is not None else "-"))
        self.orderbook_metric_count_var.set(str(status.get("metric_count_total") if status.get("metric_count_total") is not None else "-"))
        self.orderbook_db_mb_var.set(str(status.get("db_mb") if status.get("db_mb") is not None else "-"))
        self.orderbook_data_dir_mb_var.set(str(status.get("data_dir_mb") if status.get("data_dir_mb") is not None else "-"))
        self.orderbook_estimated_mb_per_day_var.set(str(status.get("estimated_mb_per_day") if status.get("estimated_mb_per_day") is not None else self.orderbook_estimated_mb_per_day_var.get()))
        self.orderbook_capacity_level_var.set(str(status.get("capacity_level") or "-"))
        self.orderbook_last_error_var.set(str(status.get("last_error") or "-"))
        self._refresh_orderbook_latest_metrics_table()
        self.refresh_orderbook_estimate()
        self._maybe_orderbook_capacity_popup(status, show_popup=show_popup)

    def open_orderbook_data_dir(self) -> None:
        path = self._orderbook_paths()["data_dir"].resolve()
        path.mkdir(parents=True, exist_ok=True)
        try:
            webbrowser.open(path.as_uri())
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not open Order Book data folder:\n{exc}")

    def open_orderbook_log(self) -> None:
        log_path = self._orderbook_paths()["log"]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.touch(exist_ok=True)
        try:
            if os.name == "nt":
                os.startfile(str(log_path))  # type: ignore[attr-defined]
            else:
                webbrowser.open(log_path.resolve().as_uri())
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not open Order Book log:\n{exc}")

    def export_orderbook_latest_metrics_csv(self) -> None:
        db_path = self._orderbook_paths()["db"]
        if not db_path.exists():
            messagebox.showerror(APP_TITLE, "Order Book database not found.")
            return
        export_dir = self._orderbook_paths()["data_dir"] / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        output_path = export_dir / f"orderbook_latest_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            with sqlite3.connect(str(db_path), timeout=5.0) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """
                    SELECT pair, symbol, status, best_bid, best_ask, spread_bps, imbalance_top20,
                           bid_pressure_ratio_60s, ask_pressure_ratio_60s,
                           nearest_bid_wall_distance_bps, nearest_ask_wall_distance_bps, last_metric_at
                    FROM stream_status
                    ORDER BY pair
                    """
                ).fetchall()
            with open(output_path, "w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "pair",
                        "symbol",
                        "status",
                        "best_bid",
                        "best_ask",
                        "spread_bps",
                        "imbalance_top20",
                        "bid_pressure_ratio_60s",
                        "ask_pressure_ratio_60s",
                        "nearest_bid_wall_distance_bps",
                        "nearest_ask_wall_distance_bps",
                        "last_metric_at",
                    ]
                )
                for row in rows:
                    writer.writerow([row[key] for key in row.keys()])
            self._append_console(f"Exported Order Book latest metrics to {output_path}\n")
            messagebox.showinfo(APP_TITLE, f"Exported {len(rows)} rows.\n\n{output_path}")
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not export Order Book metrics:\n{exc}")

    def build_collector_command(self, profile: CollectorProfile, preview_only: bool) -> list[str]:
        _ = preview_only
        collector = app_path(profile.collector_file)
        if not collector.exists():
            raise FileNotFoundError(collector)
        python_exe = self.python_exe_var.get().strip() or sys.executable
        interval_var = getattr(self, f"{profile.key}_interval_seconds_var")
        once_var = getattr(self, f"{profile.key}_once_var")
        command = [
            python_exe,
            "-u",
            str(collector),
            "--config",
            str(self._collector_config_path(profile)),
            "--data-dir",
            str(self._collector_data_dir(profile)),
            "--db",
            str(self._collector_db_path(profile)),
            "--status-file",
            str(self._collector_status_path(profile)),
            "--pid-file",
            str(self._collector_pid_path(profile)),
            "--log-file",
            str(self._collector_log_path(profile)),
            "--stop-file",
            str(self._collector_stop_path(profile)),
            "--interval-seconds",
            str(self._parse_minutes_to_seconds(interval_var.get(), profile.default_interval_minutes)),
        ]
        if once_var.get():
            command.append("--once")
        return command

    def build_news_collector_command(self, preview_only: bool) -> list[str]:
        return self.build_collector_command(NEWS_COLLECTOR_PROFILE, preview_only)

    def build_web_collector_command(self, preview_only: bool) -> list[str]:
        return self.build_collector_command(WEB_COLLECTOR_PROFILE, preview_only)

    # ------------------------------------------------------------------
    # Launch / stop
    # ------------------------------------------------------------------
    def _stop_monitor(self) -> None:
        stop_event = getattr(self, "monitor_stop_event", None)
        if stop_event is not None:
            stop_event.set()
        self.monitor_stop_event = None
        self.monitor_thread = None

    def _subprocess_launch_kwargs(self) -> dict[str, Any]:
        if os.name == "nt":
            return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
        return {"start_new_session": True}

    def _terminate_process_tree(self, process: subprocess.Popen[Any], label: str = "subprocess") -> None:
        pid = process.pid
        self._stop_monitor()
        self.stop_button.configure(state="disabled")
        self._append_console(f"Stopping {label} process tree (PID {pid})...\n")

        if process.poll() is not None:
            self._append_console(f"{label.capitalize()} already exited.\n")
            return

        if os.name == "nt":
            try:
                result = subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=20,
                )
                output = (result.stdout or "").strip()
                if output:
                    self._append_console(output + "\n")
                if result.returncode != 0 and process.poll() is None:
                    self._append_console(f"taskkill returned exit code {result.returncode}; falling back to kill().\n")
                    process.kill()
            except Exception as exc:
                self._append_console(f"Process-tree stop failed: {exc}\n")
                try:
                    process.kill()
                except Exception as kill_exc:
                    self._append_console(f"Fallback kill failed: {kill_exc}\n")
            return

        try:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            self._append_console("Process tree did not exit after SIGTERM; sending SIGKILL.\n")
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
            except Exception as exc:
                self._append_console(f"SIGKILL failed: {exc}\n")
        except Exception as exc:
            self._append_console(f"Process-tree stop failed: {exc}\n")
            try:
                process.kill()
            except Exception as kill_exc:
                self._append_console(f"Fallback kill failed: {kill_exc}\n")

    @staticmethod
    def _parse_iso_timestamp(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00"))
        except Exception:
            return None

    def _research_lab_config(self, kind: str) -> dict[str, Any]:
        if kind == "news":
            return {
                "tree": self.news_source_health_tree,
                "db_path": self._news_db_path(),
                "filters": self.news_health_filter_vars,
                "summary_vars": self.news_summary_vars,
                "note_var": self.news_scoring_note_var,
                "status_var": self.news_status_var,
            }
        return {
            "tree": self.web_source_health_tree,
            "db_path": self._web_db_path(),
            "filters": self.web_health_filter_vars,
            "summary_vars": self.web_summary_vars,
            "note_var": self.web_scoring_note_var,
            "status_var": self.web_status_var,
        }

    @staticmethod
    def _research_lab_has_tables(conn: sqlite3.Connection, table_names: set[str]) -> bool:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        available = {str(row[0]) for row in rows}
        return table_names.issubset(available)

    @staticmethod
    def _research_lab_format_value(value: Any) -> str:
        if value is None:
            return "-"
        if isinstance(value, float):
            if value != value:
                return "-"
            return f"{value:.2f}"
        return str(value)

    def _research_lab_source_priority(self, tag_pairs: list[tuple[str, str]]) -> str:
        priority_order = {"critical": 3, "high": 2, "medium": 1, "low": 0}
        best = ""
        best_rank = -1
        for namespace, value in tag_pairs:
            if namespace != "source_priority":
                continue
            rank = priority_order.get(value, -1)
            if rank > best_rank:
                best = value
                best_rank = rank
        return best

    def _research_lab_tag_set(self, tag_pairs: list[tuple[str, str]], namespace: str) -> set[str]:
        return {value for ns, value in tag_pairs if ns == namespace}

    def _research_lab_is_crypto_source(self, source_group: str, tag_pairs: list[tuple[str, str]]) -> bool:
        if source_group in RESEARCH_LAB_CRYPTO_GROUPS:
            return True
        source_types = self._research_lab_tag_set(tag_pairs, "source_type")
        asset_classes = self._research_lab_tag_set(tag_pairs, "asset_class")
        return bool(source_types & {"crypto_exchange", "crypto_media", "aggregator"} or "crypto" in asset_classes)

    def _research_lab_is_official_source(self, tag_pairs: list[tuple[str, str]]) -> bool:
        return bool(self._research_lab_tag_set(tag_pairs, "collector_confidence") & RESEARCH_LAB_OFFICIAL_CONFIDENCE)

    @staticmethod
    def _research_lab_is_disable_candidate(row: dict[str, Any], duplicate_ratio: float, usefulness_score: float) -> bool:
        if not bool(row.get("enabled")):
            return False
        last_failure_at = str(row.get("last_failure_at") or "")
        last_error = str(row.get("last_error") or "")
        items_last_fetch = int(row.get("items_last_fetch") or 0)
        inserted_last_fetch = int(row.get("inserted_last_fetch") or 0)
        return bool(
            last_failure_at
            or last_error
            or usefulness_score < 20.0
            or duplicate_ratio >= 0.5
            or (items_last_fetch > 0 and inserted_last_fetch == 0)
        )

    def _research_lab_source_matches_search(self, row: dict[str, Any], tag_pairs: list[tuple[str, str]], search_text: str) -> bool:
        if not search_text:
            return True
        haystack = " ".join(
            [
                str(row.get("source_id") or ""),
                str(row.get("source_group") or ""),
                str(row.get("market_relevance") or ""),
                str(row.get("source_priority") or ""),
                str(row.get("last_error") or ""),
                str(row.get("last_success_at") or ""),
                str(row.get("last_failure_at") or ""),
                " ".join(f"{namespace}:{value}" for namespace, value in tag_pairs),
            ]
        ).lower()
        return search_text in haystack

    def _research_lab_collect_state(
        self, kind: str
    ) -> tuple[list[dict[str, Any]], dict[str, list[tuple[str, str]]], dict[str, str], str, bool]:
        config = self._research_lab_config(kind)
        db_path: Path = config["db_path"]
        summary = {key: "-" for key in config["summary_vars"]}
        note = ""
        tag_map: dict[str, list[tuple[str, str]]] = {}
        if not db_path.exists():
            return [], tag_map, summary, note, False

        try:
            with sqlite3.connect(str(db_path), timeout=5.0) as conn:
                conn.row_factory = sqlite3.Row
                tables = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                has_scoring = {"tags", "source_tags", "source_stats_daily", "article_scores", "articles", "sources"}.issubset(tables)

                if has_scoring:
                    tag_rows = conn.execute(
                        """
                        SELECT st.source_id, t.namespace, t.value
                        FROM source_tags st
                        JOIN tags t ON t.id = st.tag_id
                        """
                    ).fetchall()
                    for row in tag_rows:
                        tag_map.setdefault(str(row["source_id"]), []).append((str(row["namespace"]), str(row["value"])))

                    stats_rows = conn.execute(
                        """
                        SELECT ssd.source_id, ssd.stat_date, ssd.articles_seen, ssd.relevant_articles,
                               ssd.high_priority_articles, ssd.duplicate_articles, ssd.avg_priority_score,
                               ssd.usefulness_score
                        FROM source_stats_daily ssd
                        JOIN (
                            SELECT source_id, MAX(stat_date) AS stat_date
                            FROM source_stats_daily
                            GROUP BY source_id
                        ) latest
                          ON latest.source_id = ssd.source_id AND latest.stat_date = ssd.stat_date
                        """
                    ).fetchall()
                    stats_map = {str(row["source_id"]): row for row in stats_rows}

                    rows: list[dict[str, Any]] = []
                    source_rows = conn.execute(
                        """
                        SELECT source_id, source_group, enabled, market_relevance, last_success_at, last_failure_at,
                               last_error, items_last_fetch, inserted_last_fetch, duplicates_last_fetch
                        FROM sources
                        ORDER BY source_group, source_id
                        """
                    ).fetchall()
                    for row in source_rows:
                        row_dict = dict(row)
                        tags = tag_map.get(str(row_dict.get("source_id") or ""), [])
                        stats = stats_map.get(str(row_dict.get("source_id") or ""), {})
                        source_priority = self._research_lab_source_priority(tags)
                        usefulness_score = float(stats["usefulness_score"] if isinstance(stats, sqlite3.Row) else stats.get("usefulness_score", 0.0) if isinstance(stats, dict) else 0.0)
                        avg_priority_score = float(stats["avg_priority_score"] if isinstance(stats, sqlite3.Row) else stats.get("avg_priority_score", 0.0) if isinstance(stats, dict) else 0.0)
                        high_priority_articles = int(stats["high_priority_articles"] if isinstance(stats, sqlite3.Row) else stats.get("high_priority_articles", 0) if isinstance(stats, dict) else 0)
                        duplicate_articles = int(stats["duplicate_articles"] if isinstance(stats, sqlite3.Row) else stats.get("duplicate_articles", 0) if isinstance(stats, dict) else 0)
                        articles_seen = int(stats["articles_seen"] if isinstance(stats, sqlite3.Row) else stats.get("articles_seen", 0) if isinstance(stats, dict) else 0)
                        duplicate_ratio = (duplicate_articles / articles_seen) if articles_seen else 0.0
                        row_dict.update(
                            {
                                "source_priority": source_priority or str(row_dict.get("market_relevance") or "-"),
                                "usefulness_score": usefulness_score,
                                "avg_priority_score": avg_priority_score,
                                "high_priority_articles": high_priority_articles,
                                "duplicate_ratio": duplicate_ratio,
                            }
                        )
                        rows.append(row_dict)

                    summary = self._research_lab_summary(conn, has_scoring)
                    return rows, tag_map, summary, note, True

                rows = [
                    dict(row)
                    for row in conn.execute(
                        """
                        SELECT source_id, source_group, enabled, last_success_at, last_failure_at, last_error,
                               items_last_fetch, inserted_last_fetch, duplicates_last_fetch
                        FROM sources
                        ORDER BY source_group, source_id
                        """
                    ).fetchall()
                ]
                note = "Scoring tables not found; run updated collector once."
                summary = self._research_lab_summary(conn, False)
                return rows, tag_map, summary, note, False
        except Exception:
            return [], tag_map, summary, "Could not read the SQLite database.", False

    def _research_lab_summary(self, conn: sqlite3.Connection, has_scoring: bool) -> dict[str, str]:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        summary = {
            "total_articles": "0",
            "high_priority_articles": "0",
            "crypto_articles": "0",
            "macro_articles": "0",
            "official_source_articles": "0",
            "failing_sources": "0",
            "top_source_group": "-",
        }
        try:
            summary["total_articles"] = str(
                int(conn.execute("SELECT COUNT(*) FROM articles WHERE collected_at >= ?", (cutoff,)).fetchone()[0])
            )
        except Exception:
            summary["total_articles"] = "-"

        try:
            summary["failing_sources"] = str(
                int(
                    conn.execute(
                        "SELECT COUNT(*) FROM sources WHERE COALESCE(last_failure_at, '') <> '' OR COALESCE(last_error, '') <> ''"
                    ).fetchone()[0]
                )
            )
        except Exception:
            summary["failing_sources"] = "-"

        if has_scoring:
            try:
                summary["high_priority_articles"] = str(
                    int(
                        conn.execute(
                            """
                            SELECT COUNT(DISTINCT a.id)
                            FROM articles a
                            JOIN article_scores s ON s.article_id = a.id
                            WHERE a.collected_at >= ? AND s.final_priority_score >= 70
                            """,
                            (cutoff,),
                        ).fetchone()[0]
                    )
                )
            except Exception:
                summary["high_priority_articles"] = "-"

            try:
                summary["crypto_articles"] = str(
                    int(
                        conn.execute(
                            """
                            SELECT COUNT(DISTINCT a.id)
                            FROM articles a
                            JOIN sources s ON s.source_id = a.source_id
                            WHERE a.collected_at >= ? AND (
                                s.source_group IN ('crypto', 'crypto_media', 'exchange_announcements', 'protocol_foundations', 'defi_protocols', 'infrastructure', 'aggregators')
                            )
                            """,
                            (cutoff,),
                        ).fetchone()[0]
                    )
                )
            except Exception:
                summary["crypto_articles"] = "-"

            try:
                summary["macro_articles"] = str(
                    int(
                        conn.execute(
                            """
                            SELECT COUNT(DISTINCT a.id)
                            FROM articles a
                            JOIN sources s ON s.source_id = a.source_id
                            WHERE a.collected_at >= ? AND (
                                s.source_group IN ('macro_economics', 'official_data_releases', 'central_banks')
                            )
                            """,
                            (cutoff,),
                        ).fetchone()[0]
                    )
                )
            except Exception:
                summary["macro_articles"] = "-"

            try:
                summary["official_source_articles"] = str(
                    int(
                        conn.execute(
                            """
                            SELECT COUNT(DISTINCT a.id)
                            FROM articles a
                            JOIN source_tags st ON st.source_id = a.source_id
                            JOIN tags t ON t.id = st.tag_id
                            WHERE a.collected_at >= ? AND t.namespace = 'collector_confidence' AND t.value IN ('official_source', 'official_page')
                            """,
                            (cutoff,),
                        ).fetchone()[0]
                    )
                )
            except Exception:
                summary["official_source_articles"] = "-"

        try:
            top_row = conn.execute(
                """
                SELECT COALESCE(source_group, 'unknown') AS source_group, COUNT(*) AS article_count
                FROM articles
                WHERE collected_at >= ?
                GROUP BY COALESCE(source_group, 'unknown')
                ORDER BY article_count DESC, source_group ASC
                LIMIT 1
                """,
                (cutoff,),
            ).fetchone()
            if top_row:
                summary["top_source_group"] = f"{top_row[0]} ({top_row[1]})"
        except Exception:
            summary["top_source_group"] = "-"

        return summary

    def _research_lab_apply_summary(self, kind: str, summary: dict[str, str], note: str) -> None:
        config = self._research_lab_config(kind)
        for key, variable in config["summary_vars"].items():
            variable.set(summary.get(key, "-"))
        config["note_var"].set(note)

    def _research_lab_build_source_health_section(self, root: Any, kind: str, row: int, tree_tip: str) -> None:
        config = self._research_lab_config(kind)
        summary = ttk.LabelFrame(root, text="Last 24h summary")
        summary.grid(row=row, column=0, sticky="ew", padx=8, pady=(0, 8))
        summary.grid_columnconfigure(1, weight=1)
        summary.grid_columnconfigure(3, weight=1)
        summary_items = [
            ("Total articles", "total_articles"),
            ("High priority articles", "high_priority_articles"),
            ("Crypto articles", "crypto_articles"),
            ("Macro articles", "macro_articles"),
            ("Official-source articles", "official_source_articles"),
            ("Failing sources", "failing_sources"),
            ("Top source group", "top_source_group"),
        ]
        for index, (label_text, key) in enumerate(summary_items):
            row_index = index // 2
            col_index = (index % 2) * 2
            label_widget = ttk.Label(summary, text=f"{label_text}:")
            label_widget.grid(row=row_index, column=col_index, sticky="w", padx=8, pady=4)
            value_widget = ttk.Label(summary, textvariable=config["summary_vars"][key])
            value_widget.grid(row=row_index, column=col_index + 1, sticky="w", padx=8, pady=4)
            ToolTip(label_widget, f"{label_text} for the last 24 hours, read directly from SQLite.")
            ToolTip(value_widget, f"{label_text} for the last 24 hours, read directly from SQLite.")

        note_label = ttk.Label(summary, textvariable=config["note_var"], foreground="#777777", wraplength=1160, justify="left")
        note_label.grid(row=4, column=0, columnspan=4, sticky="w", padx=8, pady=(2, 8))
        ToolTip(
            note_label,
            "Context: Compatibility status note for source-health analytics. Outcome: Indicates fallback mode when newer scoring tables are unavailable. "
            "Example: Fresh/older DB schema shows legacy mode until upgraded data appears.",
        )

        filters = ttk.LabelFrame(root, text="Source health filters")
        filters.grid(row=row + 1, column=0, sticky="ew", padx=8, pady=(0, 8))
        filters.grid_columnconfigure(1, weight=1)
        search_label = ttk.Label(filters, text="Search")
        search_label.grid(row=0, column=0, sticky="w", padx=8, pady=6)
        search_entry = ttk.Entry(filters, textvariable=config["filters"]["text"])
        search_entry.grid(row=0, column=1, columnspan=4, sticky="ew", padx=8, pady=6)
        ToolTip(
            search_label,
            "Context: Source-health table text filter. Outcome: Matches source id/group/tags/priority/errors/timestamps and reduces visible rows. "
            "Example: Search `timeout` or `binance`.",
        )
        ToolTip(
            search_entry,
            "Context: Source-health table text filter. Outcome: Matches source id/group/tags/priority/errors/timestamps and reduces visible rows. "
            "Example: Search `last_failure_at` date fragments.",
        )

        filter_specs = [
            ("enabled_only", "Enabled only"),
            ("failing_only", "Failing only"),
            ("crypto_only", "Crypto only"),
            ("official_only", "Official only"),
            ("disable_candidates_only", "Disable candidates"),
        ]
        for column, (key, label_text) in enumerate(filter_specs):
            check = ttk.Checkbutton(filters, text=label_text, variable=config["filters"][key])
            check.grid(row=1, column=column, sticky="w", padx=8, pady=(0, 6))
            ToolTip(
                check,
                f"Context: Structured source-health filter. Outcome: Restricts rows to {label_text.lower()} view. "
                f"Example: Enable {label_text.lower()} to narrow triage scope.",
            )

        health = ttk.LabelFrame(root, text="Source health")
        health.grid(row=row + 2, column=0, sticky="nsew", padx=8, pady=(0, 8))
        health.grid_columnconfigure(0, weight=1)
        health.grid_rowconfigure(0, weight=1)
        tree_frame = ttk.Frame(health)
        tree_frame.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        tree_frame.grid_columnconfigure(0, weight=1)
        tree_frame.grid_rowconfigure(0, weight=1)
        tree = ttk.Treeview(tree_frame, columns=RESEARCH_LAB_SOURCE_HEALTH_BASE_COLUMNS, show="headings", height=10)
        setattr(self, f"{kind}_source_health_tree", tree)
        for column in RESEARCH_LAB_SOURCE_HEALTH_BASE_COLUMNS:
            tree.heading(column, text=column)
            tree.column(column, width=150 if column != "last_error" else 320, anchor="w")
        tree.grid(row=0, column=0, sticky="nsew")
        y_scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=tree.yview)
        y_scrollbar.grid(row=0, column=1, sticky="ns")
        x_scrollbar = ttk.Scrollbar(tree_frame, orient="horizontal", command=tree.xview)
        x_scrollbar.grid(row=1, column=0, sticky="ew")
        tree.configure(yscrollcommand=y_scrollbar.set, xscrollcommand=x_scrollbar.set)
        ToolTip(tree, tree_tip)
        root.grid_rowconfigure(row + 2, weight=1)

    def _research_lab_render_rows(self, kind: str) -> None:
        config = self._research_lab_config(kind)
        tree = config["tree"]
        if tree is None:
            return
        for item in tree.get_children():
            tree.delete(item)
        rows, source_tag_map, summary, note, has_scoring = self._research_lab_collect_state(kind)
        self._research_lab_apply_summary(kind, summary, note)

        filters = config["filters"]
        search_text = filters["text"].get().strip().lower()
        enabled_only = bool(filters["enabled_only"].get())
        failing_only = bool(filters["failing_only"].get())
        crypto_only = bool(filters["crypto_only"].get())
        official_only = bool(filters["official_only"].get())
        disable_candidates_only = bool(filters["disable_candidates_only"].get())

        display_rows: list[dict[str, Any]] = []
        for row in rows:
            source_id = str(row.get("source_id") or "")
            source_group = str(row.get("source_group") or "")
            tag_pairs: list[tuple[str, str]] = source_tag_map.get(source_id, [])
            source_priority = str(row.get("source_priority") or row.get("market_relevance") or "-")
            usefulness_score = float(row.get("usefulness_score") or 0.0)
            avg_priority_score = float(row.get("avg_priority_score") or 0.0)
            duplicate_ratio = float(row.get("duplicate_ratio") or 0.0)
            if not self._research_lab_source_matches_search(
                {
                    **row,
                    "source_priority": source_priority,
                },
                tag_pairs,
                search_text,
            ):
                continue
            if enabled_only and not bool(row.get("enabled")):
                continue
            if failing_only and not (str(row.get("last_failure_at") or "") or str(row.get("last_error") or "")):
                continue
            if crypto_only and not self._research_lab_is_crypto_source(source_group, tag_pairs):
                continue
            if official_only and not self._research_lab_is_official_source(tag_pairs):
                continue
            if disable_candidates_only and not self._research_lab_is_disable_candidate(row, duplicate_ratio, usefulness_score):
                continue

            if has_scoring:
                display_rows.append(
                    {
                        "source_id": source_id,
                        "source_group": source_group,
                        "enabled": "yes" if bool(row.get("enabled")) else "no",
                        "market_relevance": str(row.get("market_relevance") or "-"),
                        "source_priority": source_priority,
                        "usefulness_score": self._research_lab_format_value(usefulness_score),
                        "avg_priority_score": self._research_lab_format_value(avg_priority_score),
                        "high_priority_articles": self._research_lab_format_value(row.get("high_priority_articles") or 0),
                        "duplicate_ratio": self._research_lab_format_value(duplicate_ratio),
                        "last_success_at": str(row.get("last_success_at") or "-"),
                        "last_failure_at": str(row.get("last_failure_at") or "-"),
                        "last_error": str(row.get("last_error") or "-"),
                        "items_last_fetch": self._research_lab_format_value(row.get("items_last_fetch") or 0),
                        "inserted_last_fetch": self._research_lab_format_value(row.get("inserted_last_fetch") or 0),
                        "duplicates_last_fetch": self._research_lab_format_value(row.get("duplicates_last_fetch") or 0),
                    }
                )
            else:
                display_rows.append(
                    {
                        "source_id": source_id,
                        "source_group": source_group,
                        "enabled": "yes" if bool(row.get("enabled")) else "no",
                        "last_success_at": str(row.get("last_success_at") or "-"),
                        "last_failure_at": str(row.get("last_failure_at") or "-"),
                        "last_error": str(row.get("last_error") or "-"),
                        "items_last_fetch": self._research_lab_format_value(row.get("items_last_fetch") or 0),
                        "inserted_last_fetch": self._research_lab_format_value(row.get("inserted_last_fetch") or 0),
                        "duplicates_last_fetch": self._research_lab_format_value(row.get("duplicates_last_fetch") or 0),
                    }
                )

        columns = RESEARCH_LAB_SOURCE_HEALTH_SCORE_COLUMNS if has_scoring else RESEARCH_LAB_SOURCE_HEALTH_BASE_COLUMNS
        tree.configure(columns=columns)
        for column in columns:
            tree.heading(column, text=column)
            tree.column(column, width=140 if column != "last_error" else 320, anchor="w")
        for row in display_rows:
            tree.insert("", "end", values=[row.get(column, "-") for column in columns])

    def _populate_news_source_health_tree(self) -> None:
        self._populate_collector_source_health_tree(NEWS_COLLECTOR_PROFILE)

    def _populate_web_source_health_tree(self) -> None:
        self._populate_collector_source_health_tree(WEB_COLLECTOR_PROFILE)

    def _populate_collector_source_health_tree(self, profile: CollectorProfile) -> None:
        self._research_lab_render_rows(profile.key)

    def refresh_collector_status(self, profile: CollectorProfile) -> None:
        from research.collectors.news_research_store import load_status

        status = load_status(self._collector_status_path(profile))
        pid_path = self._collector_pid_path(profile)
        pid_file_exists = pid_path.exists()
        pid_text = pid_path.read_text(encoding="utf-8").strip() if pid_file_exists else str(status.get("pid") or "-")
        heartbeat = str(status.get("heartbeat_at") or status.get("last_heartbeat_at") or "-")
        started_at = str(status.get("started_at") or "-")
        last_fetch_at = str(status.get("last_fetch_at") or "-")
        total_articles = str(status.get("total_articles") if status.get("total_articles") is not None else "-")
        new_articles = str(status.get("new_articles_last_cycle") if status.get("new_articles_last_cycle") is not None else "-")
        raw_last_error = str(status.get("last_error") or "").strip()
        status_code = str(status.get("status") or "unknown").strip().lower() or "unknown"
        status_reason = str(status.get("status_reason") or "").strip()
        heartbeat_dt = self._parse_iso_timestamp(heartbeat)
        process_alive: bool | None = None
        pid_value: int | None = None
        try:
            pid_value = int(str(pid_text).strip())
            process_alive = self._is_process_running(pid_value)
        except Exception:
            pid_value = None
            process_alive = None

        if status_code == "running":
            if not pid_file_exists:
                status_text = "running status is stale; PID file missing"
            elif pid_value is not None and process_alive is False:
                status_text = "running status is stale; process not alive"
            elif heartbeat_dt is not None and (datetime.now(timezone.utc) - heartbeat_dt).total_seconds() > 600:
                status_text = "running but heartbeat stale"
            else:
                status_text = "running"
        else:
            status_text = status_code

        if status_reason and (
            status_text in {"stopped", "error", "unknown"}
            or (status_text == "running" and status_reason != "normal_exit")
        ):
            status_text = f"{status_text} ({status_reason})"
        if not status:
            status_text = "unknown"
        if status_reason and status_text == "unknown":
            status_text = f"unknown ({status_reason})"

        if status_code == "error" or raw_last_error:
            last_error = raw_last_error or "-"
        else:
            last_error = "-"

        getattr(self, f"{profile.key}_status_var").set(status_text)
        getattr(self, f"{profile.key}_pid_var").set(str(pid_text or "-"))
        getattr(self, f"{profile.key}_started_at_var").set(started_at)
        getattr(self, f"{profile.key}_heartbeat_at_var").set(heartbeat)
        getattr(self, f"{profile.key}_last_fetch_at_var").set(last_fetch_at)
        getattr(self, f"{profile.key}_total_articles_var").set(total_articles)
        getattr(self, f"{profile.key}_new_articles_last_cycle_var").set(new_articles)
        getattr(self, f"{profile.key}_last_error_var").set(last_error)
        self._populate_collector_source_health_tree(profile)

    def refresh_news_status(self) -> None:
        self.refresh_collector_status(NEWS_COLLECTOR_PROFILE)

    def refresh_web_status(self) -> None:
        self.refresh_collector_status(WEB_COLLECTOR_PROFILE)

    def open_collector_data_folder(self, profile: CollectorProfile) -> None:
        path = self._collector_data_dir(profile).resolve()
        path.mkdir(parents=True, exist_ok=True)
        try:
            webbrowser.open(path.as_uri())
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not open data folder:\n{exc}")

    def open_news_data_folder(self) -> None:
        self.open_collector_data_folder(NEWS_COLLECTOR_PROFILE)

    def open_web_data_folder(self) -> None:
        self.open_collector_data_folder(WEB_COLLECTOR_PROFILE)

    def open_collector_log(self, profile: CollectorProfile) -> None:
        log_path = self._collector_log_path(profile)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.touch(exist_ok=True)
        try:
            if os.name == "nt":
                os.startfile(str(log_path))  # type: ignore[attr-defined]
            else:
                webbrowser.open(log_path.resolve().as_uri())
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not open log file:\n{exc}")

    def open_news_log(self) -> None:
        self.open_collector_log(NEWS_COLLECTOR_PROFILE)

    def open_web_log(self) -> None:
        self.open_collector_log(WEB_COLLECTOR_PROFILE)

    def export_collector_articles_csv(self, profile: CollectorProfile) -> None:
        from research.collectors.news_research_store import export_articles_csv

        db_path = self._collector_db_path(profile)
        exports_dir = self._collector_data_dir(profile) / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        csv_path = exports_dir / f"{profile.key}_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            row_count = export_articles_csv(db_path, csv_path)
            if profile.key == "news":
                self._append_console(f"Exported {row_count} news articles to {csv_path}\n")
                messagebox.showinfo(APP_TITLE, f"Exported {row_count} news articles.\n\n{csv_path}")
            else:
                self._append_console(f"Exported {row_count} {profile.label} articles to {csv_path}\n")
                messagebox.showinfo(APP_TITLE, f"Exported {row_count} {profile.label} articles.\n\n{csv_path}")
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not export CSV:\n{exc}")

    def on_export_news_csv(self) -> None:
        self.export_collector_articles_csv(NEWS_COLLECTOR_PROFILE)

    def on_export_web_csv(self) -> None:
        self.export_collector_articles_csv(WEB_COLLECTOR_PROFILE)

    def export_collector_source_health_csv(self, profile: CollectorProfile) -> None:
        from research.collectors.news_research_store import export_source_health_csv

        db_path = self._collector_db_path(profile)
        exports_dir = self._collector_data_dir(profile) / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        filename = "source_health" if profile.key == "news" else "web_source_health"
        csv_path = exports_dir / f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        try:
            row_count = export_source_health_csv(db_path, csv_path)
            if profile.key == "news":
                self._append_console(f"Exported {row_count} source health rows to {csv_path}\n")
                messagebox.showinfo(APP_TITLE, f"Exported {row_count} source health rows.\n\n{csv_path}")
            else:
                self._append_console(f"Exported {row_count} {profile.label} source health rows to {csv_path}\n")
                messagebox.showinfo(APP_TITLE, f"Exported {row_count} {profile.label} source health rows.\n\n{csv_path}")
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not export source health CSV:\n{exc}")

    def on_export_source_health_csv(self) -> None:
        self.export_collector_source_health_csv(NEWS_COLLECTOR_PROFILE)

    def on_export_web_source_health_csv(self) -> None:
        self.export_collector_source_health_csv(WEB_COLLECTOR_PROFILE)

    def start_collector(self, profile: CollectorProfile) -> None:
        collector = app_path(profile.collector_file)
        if not collector.exists():
            messagebox.showerror(APP_TITLE, f"{profile.label} collector script not found:\n{collector}")
            return
        data_dir = self._collector_data_dir(profile)
        data_dir.mkdir(parents=True, exist_ok=True)
        (data_dir / "logs").mkdir(parents=True, exist_ok=True)
        stop_path = self._collector_stop_path(profile)
        try:
            stop_path.unlink(missing_ok=True)
        except Exception:
            pass

        command = self.build_collector_command(profile, preview_only=False)
        log_path = self._collector_log_path(profile)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(log_path, "ab") as log_handle:
                kwargs: dict[str, Any] = {
                    "stdin": subprocess.DEVNULL,
                    "stdout": log_handle,
                    "stderr": subprocess.STDOUT,
                    "cwd": str(Path(__file__).resolve().parent),
                    "env": utf8_subprocess_env(),
                    "close_fds": True,
                }
                if os.name == "nt":
                    flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(subprocess, "DETACHED_PROCESS", 0)
                    kwargs["creationflags"] = flags
                else:
                    kwargs["start_new_session"] = True
                process = subprocess.Popen(command, **kwargs)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not start {profile.label} collector:\n{exc}")
            return

        getattr(self, f"{profile.key}_status_var").set(f"started detached PID {process.pid}")
        getattr(self, f"{profile.key}_pid_var").set(str(process.pid))
        self._append_console(f"Started {profile.label} collector detached: PID {process.pid}\n")
        self.refresh_collector_status(profile)

    def on_start_news_collector(self) -> None:
        self.start_collector(NEWS_COLLECTOR_PROFILE)

    def on_start_web_collector(self) -> None:
        self.start_collector(WEB_COLLECTOR_PROFILE)

    def stop_collector(self, profile: CollectorProfile) -> None:
        stop_path = self._collector_stop_path(profile)
        try:
            stop_path.parent.mkdir(parents=True, exist_ok=True)
            stop_path.write_text(utc_now() + "\n", encoding="utf-8")
            getattr(self, f"{profile.key}_status_var").set("stop requested")
            self._append_console(f"{profile.label} stop requested via {stop_path}\n")
            self.refresh_collector_status(profile)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not request {profile.label} stop:\n{exc}")

    def on_stop_news_collector(self) -> None:
        self.stop_collector(NEWS_COLLECTOR_PROFILE)

    def on_stop_web_collector(self) -> None:
        self.stop_collector(WEB_COLLECTOR_PROFILE)

    def on_run(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo(APP_TITLE, "A run is already active.")
            return
        self._save_last_used_state()

        try:
            build = self.build_command(preview_only=False)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Cannot build command:\n{exc}")
            return

        self.current_temp_config = build.temp_config_path
        backend = self.backend_var.get()
        self.active_run_type = self.run_type_var.get()
        self.stdout_parse_buffer = ""
        self.adjust_events = []

        self._stop_monitor()
        if self.run_type_var.get() == "Hyperopt" and self.hyperopt_monitor_results_var.get():
            self.monitor_stop_event = threading.Event()
            run_started_at = time.time()
            self.monitor_thread = threading.Thread(target=self._tail_hyperopt_results_worker, args=(run_started_at, self.monitor_stop_event), daemon=True)
            self.monitor_thread.start()

        self._append_console("\n" + "=" * 100 + "\n")
        self._append_console(shell_join(build.preview_command) + "\n")
        if backend == "In-process (debug)" and (self.python_exe_var.get().strip() and Path(self.python_exe_var.get().strip()) != Path(sys.executable)):
            self._append_console(
                f"Note: in-process mode always uses the current interpreter: {sys.executable}\n"
            )

        self.run_button.configure(state="disabled")
        if hasattr(self, "explorer_run_button"):
            self.explorer_run_button.configure(state="disabled")
        if self.explorer_tab_run_button is not None:
            self.explorer_tab_run_button.configure(state="disabled")
        if hasattr(self, "frequi_launch_button"):
            self.frequi_launch_button.configure(state="disabled")
        self.stop_button.configure(state="normal" if backend == "Subprocess" else "disabled")

        if backend == "Subprocess":
            self.worker_thread = threading.Thread(target=self._run_subprocess, args=(build,), daemon=True)
        else:
            self.worker_thread = threading.Thread(target=self._run_in_process, args=(build,), daemon=True)
        self.worker_thread.start()

    def on_run_explorer(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo(APP_TITLE, "A run is already active.")
            return
        self._save_last_used_state()

        try:
            command, temp_preset_path = self.build_explorer_command(preview_only=False)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Cannot build explorer command:\n{exc}")
            return

        self.current_temp_config = temp_preset_path
        self.active_run_type = "Explorer"
        self.stdout_parse_buffer = ""
        self.adjust_events = []
        self._stop_monitor()

        self.explorer_total_runs = 0
        self.explorer_current_run = 0

        # Rough estimate: updated after the runner starts writing status lines.
        self.explorer_total_runs = 0
        self.explorer_current_run = 0
        self.explorer_last_status = {}
        self.explorer_summary_block = ""
        self.explorer_summary_section = ""
        self._reset_explorer_live_status()

        self._append_console("\n" + "=" * 100 + "\n")
        self._append_console("Explorer\n")
        self._append_console(shell_join(command) + "\n")
        self._append_explorer_summary("\n" + "=" * 100 + "\n")
        self._append_explorer_summary("Explorer\n")
        self.explorer_summary_section = "header"

        self.run_button.configure(state="disabled")
        if hasattr(self, "explorer_run_button"):
            self.explorer_run_button.configure(state="disabled")
        if self.explorer_tab_run_button is not None:
            self.explorer_tab_run_button.configure(state="disabled")
        if hasattr(self, "frequi_launch_button"):
            self.frequi_launch_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.worker_thread = threading.Thread(target=self._run_explorer_subprocess, args=(command,), daemon=True)
        self.worker_thread.start()

    def on_launch_frequi(self) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo(APP_TITLE, "A run is already active. Stop it before launching FreqUI.")
            return

        url = self.frequi_url_var.get().strip() or "http://127.0.0.1:8080"
        if not (url.startswith("http://") or url.startswith("https://")):
            normalized_url = "http://" + url
            if not messagebox.askyesno(APP_TITLE, f"FreqUI URL does not start with http:// or https://.\n\nUse this URL instead?\n{normalized_url}"):
                return
            self.frequi_url_var.set(normalized_url)

        self._save_last_used_state()

        try:
            build = self.build_frequi_command(preview_only=False)
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Cannot build FreqUI command:\n{exc}")
            return

        self.current_temp_config = build.temp_config_path
        self.active_run_type = "FreqUI"
        self.stdout_parse_buffer = ""
        self.adjust_events = []
        self._stop_monitor()

        self._append_console("\n" + "=" * 100 + "\n")
        self._append_console("FreqUI\n")
        self._append_console(shell_join(build.preview_command) + "\n")

        self.run_button.configure(state="disabled")
        if hasattr(self, "explorer_run_button"):
            self.explorer_run_button.configure(state="disabled")
        if self.explorer_tab_run_button is not None:
            self.explorer_tab_run_button.configure(state="disabled")
        if hasattr(self, "frequi_launch_button"):
            self.frequi_launch_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.frequi_status_var.set("Running")

        self.worker_thread = threading.Thread(target=self._run_subprocess, args=(build,), daemon=True)
        self.worker_thread.start()
        if self.frequi_open_after_launch_var.get():
            readiness_thread = threading.Thread(target=self._wait_for_frequi_ready_and_open, daemon=True)
            readiness_thread.start()

    def open_frequi_url(self, url: str | None = None) -> None:
        if not url:
            url = self.frequi_url_var.get().strip() or "http://127.0.0.1:8080"
        self.frequi_url_var.set(url)
        try:
            webbrowser.open(url)
            self._append_console(f"Opened FreqUI: {url}\n")
        except Exception as exc:
            messagebox.showerror(APP_TITLE, f"Could not open FreqUI URL:\n{exc}")

    def _frequi_ping_url(self) -> str:
        base_url = self.frequi_url_var.get().strip() or "http://127.0.0.1:8080"
        parsed = urllib_parse.urlparse(base_url)
        path = parsed.path or ""
        if not path.endswith("/api/v1/ping"):
            path = path.rstrip("/") + "/api/v1/ping"
        if not path.startswith("/"):
            path = "/" + path
        return urllib_parse.urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))

    def _is_frequi_ready(self, ping_url: str) -> bool:
        request = urllib_request.Request(ping_url, method="GET")
        try:
            with urllib_request.urlopen(request, timeout=2.0) as response:
                if response.status != 200:
                    return False
                raw = response.read().decode("utf-8", "replace").strip()
        except (urllib_error.URLError, urllib_error.HTTPError, TimeoutError, OSError):
            return False

        compact = raw.replace(" ", "").lower()
        if compact == '{"status":"pong"}' or '"status":"pong"' in compact:
            return True
        try:
            payload = json.loads(raw)
        except Exception:
            return "status" in compact and "pong" in compact
        return isinstance(payload, dict) and str(payload.get("status", "")).lower() == "pong"

    def _wait_for_frequi_ready_and_open(self) -> None:
        ping_url = self._frequi_ping_url()
        timeout_seconds = 25.0
        poll_seconds = 0.75
        deadline = time.time() + timeout_seconds

        while time.time() < deadline:
            process = self.current_process
            if process is not None and process.poll() is not None:
                self.output_queue.put(("monitor", "FreqUI webserver exited before becoming ready. Check console output.\n"))
                self.output_queue.put(("frequi_status", "Failed: webserver exited"))
                return
            if process is None and (not self.worker_thread or not self.worker_thread.is_alive()):
                self.output_queue.put(("monitor", "FreqUI webserver exited before becoming ready. Check console output.\n"))
                self.output_queue.put(("frequi_status", "Failed: webserver exited"))
                return

            if self._is_frequi_ready(ping_url):
                open_url = self.frequi_url_var.get().strip() or "http://127.0.0.1:8080"
                self.output_queue.put(("monitor", f"FreqUI webserver is ready at {ping_url}\n"))
                self.output_queue.put(("frequi_open_url", open_url))
                self.output_queue.put(("frequi_status", "Running"))
                return

            time.sleep(poll_seconds)

        process = self.current_process
        if process is not None and process.poll() is None:
            self.output_queue.put(
                (
                    "monitor",
                    "FreqUI webserver is still running but did not answer /api/v1/ping yet. Open manually when ready.\n",
                )
            )
            self.output_queue.put(("frequi_status", "Running (not ready yet)"))
            return

        self.output_queue.put(("monitor", "FreqUI webserver exited before becoming ready. Check console output.\n"))
        self.output_queue.put(("frequi_status", "Failed: webserver exited"))

    def on_stop(self) -> None:
        if self.current_process and self.current_process.poll() is None:
            self._terminate_process_tree(self.current_process)
        else:
            messagebox.showinfo(APP_TITLE, "Only subprocess runs can be stopped from the launcher.")

    def _run_subprocess(self, build: BuildResult) -> None:
        python_exe = self.python_exe_var.get().strip() or sys.executable
        project_root = self.project_root_var.get().strip() or None

        # Force Python child output to UTF-8 so Windows code-page differences do not
        # crash the launcher while reading logs.
        env = utf8_subprocess_env()

        if project_root:
            current_pp = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = project_root + (os.pathsep + current_pp if current_pp else "")

        command = [python_exe, "-u", "-m", "freqtrade", *build.command_args]
        try:
            self.current_process = subprocess.Popen(
                command,
                cwd=project_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=False,
                bufsize=0,
                **self._subprocess_launch_kwargs(),
            )
            assert self.current_process.stdout is not None
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            stdout_fd = self.current_process.stdout.fileno()
            while True:
                chunk = os.read(stdout_fd, 4096)
                if not chunk:
                    break
                text_chunk = decoder.decode(chunk)
                if text_chunk:
                    self.output_queue.put(("stdout", text_chunk.replace("\r", "\n")))
            final_text = decoder.decode(b"", final=True)
            if final_text:
                self.output_queue.put(("stdout", final_text.replace("\r", "\n")))
            returncode = self.current_process.wait()
            self.output_queue.put(("stdout", f"\nProcess finished with exit code {returncode}.\n"))
        except Exception:
            self.output_queue.put(("stdout", traceback.format_exc() + "\n"))
        finally:
            self.current_process = None
            self._cleanup_temp_config()
            self.output_queue.put(("status", "idle"))

    def _run_explorer_subprocess(self, command: list[str]) -> None:
        project_root = self.project_root_var.get().strip() or None

        env = utf8_subprocess_env()
        if project_root:
            current_pp = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = project_root + (os.pathsep + current_pp if current_pp else "")

        try:
            self.current_process = subprocess.Popen(
                command,
                cwd=project_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=False,
                bufsize=0,
                **self._subprocess_launch_kwargs(),
            )
            assert self.current_process.stdout is not None
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            stdout_fd = self.current_process.stdout.fileno()
            while True:
                chunk = os.read(stdout_fd, 4096)
                if not chunk:
                    break
                text_chunk = decoder.decode(chunk)
                if text_chunk:
                    self.output_queue.put(("stdout", text_chunk.replace("\r", "\n")))
            final_text = decoder.decode(b"", final=True)
            if final_text:
                self.output_queue.put(("stdout", final_text.replace("\r", "\n")))
            returncode = self.current_process.wait()
            self.output_queue.put(("stdout", f"\nExplorer process finished with exit code {returncode}.\n"))
        except Exception:
            self.output_queue.put(("stdout", traceback.format_exc() + "\n"))
        finally:
            self.current_process = None
            self._cleanup_temp_config()
            self.output_queue.put(("status", "idle"))

    def _run_in_process(self, build: BuildResult) -> None:
        project_root = self.project_root_var.get().strip()
        writer = QueueWriter(self.output_queue)
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        old_argv = sys.argv[:]
        old_cwd = os.getcwd()
        old_sys_path = sys.path[:]

        try:
            sys.stdout = writer
            sys.stderr = writer
            if project_root:
                os.chdir(project_root)
                if project_root not in sys.path:
                    sys.path.insert(0, project_root)

            sys.argv = ["freqtrade", *build.command_args]

            try:
                if importlib.util.find_spec("freqtrade.__main__") is not None:
                    runpy.run_module("freqtrade", run_name="__main__", alter_sys=True)
                else:
                    # Fallback for environments where the executable entrypoint is
                    # exposed via freqtrade.main only.
                    runpy.run_module("freqtrade.main", run_name="__main__", alter_sys=True)
                exit_code = 0
            except SystemExit as exc:
                exit_code = exc.code if isinstance(exc.code, int) else 0

            self.output_queue.put(("stdout", f"\nIn-process run finished with exit code {exit_code}.\n"))
        except Exception:
            self.output_queue.put(("stdout", traceback.format_exc() + "\n"))
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            sys.argv = old_argv
            sys.path[:] = old_sys_path
            try:
                os.chdir(old_cwd)
            except Exception:
                pass
            self._cleanup_temp_config()
            self.output_queue.put(("status", "idle"))

    def _cleanup_temp_config(self) -> None:
        temp_path = self.current_temp_config
        self.current_temp_config = None
        self._stop_monitor()
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass

    # ------------------------------------------------------------------
    # Console / status
    # ------------------------------------------------------------------
    def _default_explorer_tally_text(self) -> str:
        return (
            "Runs: A 0 | R 0 | Loops 0\n"
            "Champion Profit:    - -> -\n"
            "Champion Objective: - -> -\n"
            "Champion WinRatio:  -/- -> -/-"
        )

    def _reset_explorer_live_status(self) -> None:
        self.explorer_accepted_count = 0
        self.explorer_rejected_count = 0
        self.explorer_loop_count = 0
        self.explorer_loop_log_entries = 0
        self.explorer_current_loop_targets = {}
        self.explorer_current_hyperopt_window = "-"
        self.explorer_initial_champion_stats = {}
        self.explorer_current_champion_stats = {}
        self.explorer_tally_var.set(self._default_explorer_tally_text())
        self._refresh_explorer_summary_context()
        if hasattr(self, "explorer_loop_log"):
            self.explorer_loop_log.configure(state="normal")
            self.explorer_loop_log.delete("1.0", tk.END)
            self.explorer_loop_log.configure(state="disabled")

    def _format_explorer_metric(self, value: Any) -> str:
        try:
            return f"{float(value):.4f}"
        except Exception:
            return "-"

    def _win_ratio_text(self, stats: dict[str, Any]) -> str:
        try:
            wins = int(stats.get("winning_windows"))
            total = int(stats.get("total_windows"))
        except Exception:
            return "-/-"
        return f"{wins}/{total}" if total > 0 else "-/-"

    def _stats_from_explorer_event(self, event: dict[str, Any], role: str) -> dict[str, Any]:
        prefix = "challenger" if role == "challenger" else "champion"
        total_windows = int(event.get("backtest_window_count") or 0)
        try:
            loss_windows = int(event.get(f"{prefix}_loss_window_count") or event.get(f"{prefix}_loss_windows") or 0)
        except Exception:
            loss_windows = 0
        return {
            "profit_total": event.get(f"{prefix}_profit_total"),
            "objective_total": event.get(f"{prefix}_objective_total"),
            "loss_windows": loss_windows,
            "total_windows": total_windows,
            "winning_windows": max(0, total_windows - loss_windows) if total_windows > 0 else 0,
        }

    def _refresh_explorer_tally(self) -> None:
        initial = self.explorer_initial_champion_stats
        current = self.explorer_current_champion_stats
        self.explorer_tally_var.set(
            f"Runs: A {self.explorer_accepted_count} | R {self.explorer_rejected_count} | Loops {self.explorer_loop_count}\n"
            "Champion Profit:    "
            f"{self._format_explorer_metric(initial.get('profit_total'))} -> {self._format_explorer_metric(current.get('profit_total'))}\n"
            "Champion Objective: "
            f"{self._format_explorer_metric(initial.get('objective_total'))} -> {self._format_explorer_metric(current.get('objective_total'))}\n"
            "Champion WinRatio:  "
            f"{self._win_ratio_text(initial)} -> {self._win_ratio_text(current)}"
        )
        self._refresh_explorer_summary_context()

    def _refresh_explorer_summary_context(self) -> None:
        loss_name = self.hyperopt_loss_var.get().strip() or "-"
        base_window = str(self.explorer_current_hyperopt_window or "-").strip() or "-"
        self.explorer_summary_context_var.set(f"Loss: {loss_name} | HyperOpt window: {base_window}")

    def _append_explorer_loop_decision(self, event: dict[str, Any]) -> None:
        if not hasattr(self, "explorer_loop_log"):
            return
        loop_index = int(event.get("loop_index") or 0)
        target = dict(self.explorer_current_loop_targets.get(loop_index) or {})
        target.update({key: event.get(key) for key in ("target", "namespace", "namespace_value", "family", "tag") if event.get(key)})
        decision = str(event.get("decision") or "").upper() or "-"
        namespace = str(target.get("namespace") or "-")
        namespace_value = str(target.get("namespace_value") or "-")
        family = str(target.get("family") or "-")
        tag = str(target.get("tag") or target.get("target") or "-")
        target_label = str(target.get("target") or "-")
        line = (
            f"Loop {loop_index}: {decision}\n"
            f"  target={target_label}\n"
            f"  namespace={namespace}:{namespace_value} | family={family} | tag={tag}\n"
        )

        self.explorer_loop_log.configure(state="normal")
        if self.explorer_loop_log_entries >= 100:
            self.explorer_loop_log.delete("1.0", tk.END)
            self.explorer_loop_log_entries = 0
        self.explorer_loop_log.insert(tk.END, line)
        self.explorer_loop_log_entries += 1
        self.explorer_loop_log.see(tk.END)
        self.explorer_loop_log.configure(state="disabled")

    def _handle_explorer_status_marker(self, event: dict[str, Any]) -> None:
        self.explorer_last_status = dict(event)
        self._update_explorer_live_status(event)
        summary = self._format_explorer_status_summary(event)
        if summary:
            self._append_explorer_summary_section(summary, self._explorer_summary_section_key(summary))

    def _update_explorer_live_status(self, event: dict[str, Any]) -> None:
        event_name = str(event.get("event") or "")
        if event_name == "loop_start":
            hyperopt_window = str(event.get("hyperopt_window") or "").strip()
            if hyperopt_window:
                self.explorer_current_hyperopt_window = hyperopt_window
            try:
                loop_index = int(event.get("loop_index") or 0)
            except Exception:
                loop_index = 0
            if loop_index > 0:
                self.explorer_loop_count = max(self.explorer_loop_count, loop_index)
                self.explorer_current_loop_targets[loop_index] = {
                    "target": event.get("target"),
                    "namespace": event.get("namespace"),
                    "namespace_value": event.get("namespace_value"),
                    "family": event.get("family"),
                    "tag": event.get("tag"),
                }
                self._refresh_explorer_tally()
            else:
                self._refresh_explorer_summary_context()
            return

        if event_name != "challenger_decision":
            return

        decision = str(event.get("decision") or "").upper()
        if decision == "ACCEPTED":
            self.explorer_accepted_count += 1
        elif decision == "REJECTED":
            self.explorer_rejected_count += 1

        champion_stats = self._stats_from_explorer_event(event, "champion")
        challenger_stats = self._stats_from_explorer_event(event, "challenger")
        if not self.explorer_initial_champion_stats:
            self.explorer_initial_champion_stats = dict(champion_stats)
            self.explorer_current_champion_stats = dict(champion_stats)
        if decision == "ACCEPTED":
            self.explorer_current_champion_stats = dict(challenger_stats)

        self._append_explorer_loop_decision(event)
        self._refresh_explorer_tally()

    def _console_at_bottom(self) -> bool:
        try:
            return self.console.yview()[1] >= 0.999
        except Exception:
            return True

    def _explorer_summary_at_bottom(self) -> bool:
        try:
            return self.explorer_summary_console.yview()[1] >= 0.999
        except Exception:
            return True

    def _append_console(self, text: str) -> None:
        should_follow = self.console_follow_tail_var.get() and self._console_at_bottom()
        top_index = self.console.index("@0,0")
        self.console.configure(state="normal")
        self.console.insert(tk.END, text)
        if should_follow:
            self.console.see(tk.END)
        else:
            self.console.yview(top_index)
        self.console.configure(state="disabled")

    def _append_explorer_summary(self, text: str) -> None:
        should_follow = self.explorer_summary_follow_tail_var.get() and self._explorer_summary_at_bottom()
        top_index = self.explorer_summary_console.index("@0,0")
        self.explorer_summary_console.configure(state="normal")
        self.explorer_summary_console.insert(tk.END, text)
        if should_follow:
            self.explorer_summary_console.see(tk.END)
        else:
            self.explorer_summary_console.yview(top_index)
        self.explorer_summary_console.configure(state="disabled")

    def _explorer_summary_text(self) -> str:
        try:
            return self.explorer_summary_console.get("1.0", "end-1c")
        except Exception:
            return ""

    def _append_explorer_summary_section(self, text: str, section: str) -> None:
        section = section or "misc"
        existing = self._explorer_summary_text()
        inserted_gap = False
        if existing and self.explorer_summary_section and self.explorer_summary_section != section:
            self._append_explorer_summary("\n" if existing.endswith("\n") else "\n\n")
            inserted_gap = True
        if inserted_gap:
            text = text.lstrip("\n")
        self.explorer_summary_section = section
        self._append_explorer_summary(text)

    def _append_review_console(self, text: str) -> None:
        self.review_console.configure(state="normal")
        self.review_console.insert(tk.END, text)
        self.review_console.see(tk.END)
        self.review_console.configure(state="disabled")

    def _drain_output_queue(self) -> None:
        try:
            while True:
                kind, payload = self.output_queue.get_nowait()
                if kind == "stdout":
                    self._handle_stdout_payload(payload)
                elif kind == "monitor":
                    self._append_console(payload)
                elif kind == "review":
                    self._append_review_console(payload)
                elif kind == "frequi_open_url":
                    self.open_frequi_url(payload)
                elif kind == "frequi_status":
                    self.frequi_status_var.set(str(payload))
                elif kind == "status" and payload == "idle":
                    previous_run_type = self.active_run_type
                    self._flush_stdout_parse_buffer()
                    self.explorer_summary_block = ""
                    summary = self._format_adjust_summary()
                    if summary:
                        self._append_console(summary)
                    if self.active_run_type == "Backtest":
                        self.set_latest_backtest_review_file(silent=True)
                    self.active_run_type = ""
                    self.run_button.configure(state="normal")
                    if hasattr(self, "frequi_launch_button"):
                        self.frequi_launch_button.configure(state="normal")
                    if hasattr(self, "explorer_run_button"):
                        self.explorer_run_button.configure(state="normal")
                    if self.explorer_tab_run_button is not None:
                        self.explorer_tab_run_button.configure(state="normal")
                    self.stop_button.configure(state="disabled")
                    if hasattr(self, "frequi_status_var"):
                        current_status = self.frequi_status_var.get().strip().lower()
                        if not (previous_run_type == "FreqUI" and current_status.startswith("failed")):
                            self.frequi_status_var.set("Stopped")
                    self.refresh_mode_options()
        except queue.Empty:
            pass
        self.after(100, self._drain_output_queue)

    def focus_console_search(self, _event: tk.Event | None = None) -> str:
        self.console_search_entry.focus_set()
        self.console_search_entry.selection_range(0, tk.END)
        return "break"

    def scroll_console_to_bottom(self) -> None:
        self.console.see(tk.END)

    def scroll_explorer_summary_to_bottom(self) -> None:
        self.explorer_summary_console.see(tk.END)

    def clear_console_search(self) -> None:
        self.console.configure(state="normal")
        self.console.delete("1.0", tk.END)
        self.console.configure(state="disabled")
        self.console_search_var.set("")
        self.console_key_term_var.set("")
        self._clear_console_search_tags()
        self.console_search_status_var.set("")

    def clear_explorer_summary(self) -> None:
        self.explorer_summary_console.configure(state="normal")
        self.explorer_summary_console.delete("1.0", tk.END)
        self.explorer_summary_console.configure(state="disabled")
        self.explorer_summary_section = ""
        self.explorer_summary_block = ""
        self.clear_explorer_summary_search()

    def apply_console_key_term(self) -> None:
        term = self.console_key_term_var.get().strip()
        if not term:
            return
        self.console_search_var.set(term)
        self.after_idle(self.find_console_next)

    def _on_console_search_changed(self) -> None:
        if not self.console_search_var.get():
            self._clear_console_search_tags()
            self.console_search_status_var.set("")

    def _clear_console_search_tags(self) -> None:
        self.console.tag_remove("search_match", "1.0", tk.END)
        self.console.tag_remove("search_active", "1.0", tk.END)

    def _on_explorer_summary_search_changed(self) -> None:
        if not self.explorer_summary_search_var.get():
            self._clear_explorer_summary_search_tags()
            self.explorer_summary_search_status_var.set("")

    def _clear_explorer_summary_search_tags(self) -> None:
        self.explorer_summary_console.tag_remove("summary_search_match", "1.0", tk.END)
        self.explorer_summary_console.tag_remove("summary_search_active", "1.0", tk.END)

    def clear_explorer_summary_search(self) -> None:
        self.explorer_summary_search_var.set("")
        self._clear_explorer_summary_search_tags()
        self.explorer_summary_search_status_var.set("")

    def _highlight_console_matches(self) -> list[tuple[str, str]]:
        self._clear_console_search_tags()
        needle = self.console_search_var.get()
        if not needle:
            return []

        matches: list[tuple[str, str]] = []
        start = "1.0"
        count = tk.IntVar()
        nocase = not self.console_search_case_var.get()
        while True:
            index = self.console.search(needle, start, stopindex=tk.END, nocase=nocase, count=count)
            if not index or count.get() <= 0:
                break
            end = f"{index}+{count.get()}c"
            matches.append((index, end))
            self.console.tag_add("search_match", index, end)
            start = end
        return matches

    def _highlight_explorer_summary_matches(self) -> list[tuple[str, str]]:
        self._clear_explorer_summary_search_tags()
        needle = self.explorer_summary_search_var.get()
        if not needle:
            return []

        matches: list[tuple[str, str]] = []
        start = "1.0"
        count = tk.IntVar()
        nocase = not self.explorer_summary_search_case_var.get()
        while True:
            index = self.explorer_summary_console.search(needle, start, stopindex=tk.END, nocase=nocase, count=count)
            if not index or count.get() <= 0:
                break
            end = f"{index}+{count.get()}c"
            matches.append((index, end))
            self.explorer_summary_console.tag_add("summary_search_match", index, end)
            start = end
        return matches

    def _active_console_match_start(self) -> str | None:
        ranges = self.console.tag_ranges("search_active")
        if ranges:
            return str(ranges[0])
        return None

    def _active_explorer_summary_match_start(self) -> str | None:
        ranges = self.explorer_summary_console.tag_ranges("summary_search_active")
        if ranges:
            return str(ranges[0])
        return None

    def _activate_console_match(self, matches: list[tuple[str, str]], match_index: int) -> None:
        self.console.tag_remove("search_active", "1.0", tk.END)
        start, end = matches[match_index]
        self.console.tag_add("search_active", start, end)
        self.console.mark_set(tk.INSERT, start)
        self.console.see(start)
        self.console_search_status_var.set(f"{match_index + 1} of {len(matches)}")

    def _activate_explorer_summary_match(self, matches: list[tuple[str, str]], match_index: int) -> None:
        self.explorer_summary_console.tag_remove("summary_search_active", "1.0", tk.END)
        start, end = matches[match_index]
        self.explorer_summary_console.tag_add("summary_search_active", start, end)
        self.explorer_summary_console.mark_set(tk.INSERT, start)
        self.explorer_summary_console.see(start)
        self.explorer_summary_search_status_var.set(f"{match_index + 1} of {len(matches)}")

    def _find_console(self, backwards: bool) -> None:
        active_start = self._active_console_match_start()
        matches = self._highlight_console_matches()
        if not matches:
            self.console_search_status_var.set("No matches" if self.console_search_var.get() else "")
            return

        anchor = self.console.index(tk.INSERT)
        if active_start:
            anchor = self.console.index(f"{active_start} {'-1c' if backwards else '+1c'}")

        chosen: int | None = None
        if backwards:
            for index, (start, _end) in enumerate(matches):
                if self.console.compare(start, "<=", anchor):
                    chosen = index
                else:
                    break
            if chosen is None:
                chosen = len(matches) - 1
        else:
            for index, (start, _end) in enumerate(matches):
                if not self.console.compare(start, "<", anchor):
                    chosen = index
                    break
            if chosen is None:
                chosen = 0
        self._activate_console_match(matches, chosen)

    def _find_explorer_summary(self, backwards: bool) -> None:
        active_start = self._active_explorer_summary_match_start()
        matches = self._highlight_explorer_summary_matches()
        if not matches:
            self.explorer_summary_search_status_var.set("No matches" if self.explorer_summary_search_var.get() else "")
            return

        anchor = self.explorer_summary_console.index(tk.INSERT)
        if active_start:
            anchor = self.explorer_summary_console.index(f"{active_start} {'-1c' if backwards else '+1c'}")

        chosen: int | None = None
        if backwards:
            for index, (start, _end) in enumerate(matches):
                if self.explorer_summary_console.compare(start, "<=", anchor):
                    chosen = index
                else:
                    break
            if chosen is None:
                chosen = len(matches) - 1
        else:
            for index, (start, _end) in enumerate(matches):
                if not self.explorer_summary_console.compare(start, "<", anchor):
                    chosen = index
                    break
            if chosen is None:
                chosen = 0
        self._activate_explorer_summary_match(matches, chosen)

    def find_console_next(self) -> None:
        self._find_console(backwards=False)

    def find_console_previous(self) -> None:
        self._find_console(backwards=True)

    def find_explorer_summary_next(self) -> None:
        self._find_explorer_summary(backwards=False)

    def find_explorer_summary_previous(self) -> None:
        self._find_explorer_summary(backwards=True)

    def find_explorer_summary_accepted(self) -> None:
        self.explorer_summary_search_var.set("Decision: Accepted")
        self.explorer_summary_search_case_var.set(False)
        self.after_idle(self.find_explorer_summary_next)

    def _format_explorer_status_summary(self, event: dict[str, Any]) -> str:
        event_name = str(event.get("event") or "")
        if event_name == "loop_start":
            backtest_windows = [str(item) for item in event.get("backtest_windows") or [] if str(item)]
            if backtest_windows:
                return f"Generated/random backtest windows ({event.get('backtest_window_count', len(backtest_windows))}): {', '.join(backtest_windows)}\n"
            return ""

        if event_name == "challenger_decision":
            decision = str(event.get("decision") or "").upper()
            prefix = "\n" if decision == "ACCEPTED" else ""
            reason = str(event.get("decision_reason") or "").strip()
            reason_text = f" | reason={reason}" if reason and reason != "weighted_score_improved" else ""
            return (
                f"{prefix}Decision: {decision} | target={event.get('target')} | "
                f"params_changed={event.get('params_changed')} | "
                f"champion={event.get('champion_score')} | "
                f"challenger={event.get('challenger_score')} | "
                f"delta={event.get('weighted_score_delta')}"
                f"{reason_text}\n"
            )
        return ""

    def _is_explorer_summary_line(self, line: str) -> bool:
        stripped = line.strip()
        if not stripped:
            return False
        if stripped.startswith(EXPLORER_SUMMARY_PREFIXES):
            return True
        if stripped in EXPLORER_SUMMARY_TABLE_HEADERS:
            return True
        if stripped.startswith(EXPLORER_SUMMARY_TABLE_ROW_PREFIXES):
            return True
        if set(stripped) in ({"-"}, {"="}):
            return False
        return False

    def _explorer_summary_section_key(self, line: str) -> str:
        stripped = line.strip()
        if not stripped:
            return ""
        if stripped.startswith(("End of Explorer search run", "Start of Explorer search run")):
            return "search_run"
        lowered = stripped.lower()
        if stripped.startswith(("End of ", "Start of ")) and "run" in lowered and (("explorer" in lowered) or ("batch" in lowered)):
            return "search_run"
        if stripped.startswith(("Explorer loop", "HyperOpt window:", "Backtest windows:", "Generated/random backtest windows")):
            return "loop"
        if stripped.startswith(("Selected random tag pool for this Explorer launch:", "Selected random family pool for this Explorer launch:", "Selected custom batches for this Explorer launch:")):
            return "selection"
        if stripped.startswith(("Run-level selected target:", "Run-level selected targets:", "Explorer target", "Search run for", "Target:", "Mode: Custom batches", "Params:", "Spaces:", "Excluded:")):
            return "target"
        if stripped.startswith(("Active HyperOpt params:", "Min-param padding tags:", "Min-param padding families:")):
            return "params"
        if stripped.startswith("Warning:"):
            return "warning"
        if stripped.startswith(("Collected challenger:", "No challengers")):
            return "collection"
        if stripped.startswith((
            "Champion backtest baseline:",
            "Champion baseline reused from cache:",
            "Champion baseline backtest executed:",
            "Starting champion baseline backtest",
            "Champion backtest failed",
        )):
            return "champion"
        if stripped == "Backtest comparison" or stripped in EXPLORER_SUMMARY_TABLE_HEADERS:
            return "comparison"
        if stripped.startswith(EXPLORER_SUMMARY_TABLE_ROW_PREFIXES):
            return "comparison"
        if stripped.startswith("Acceptance:") or stripped.startswith(("Decision:", "Accepted", "Champion updated", "No challenger beat the champion baseline")):
            return "decision"
        if stripped.startswith((
            "=== Challenger",
            "HyperOpt loss:",
            "Starting backtest with challenger",
            "Backtest failed",
            "HyperOpt failed",
            "No hyperopt result file found",
            "Could not read loss",
        )):
            return "challenger"
        return "misc"

    def _handle_explorer_summary_line(self, line: str) -> None:
        stripped = line.strip()
        if not stripped:
            return

        if stripped == "Backtest comparison":
            self.explorer_summary_block = "comparison"
            self._append_explorer_summary_section(line, "comparison")
            return

        if stripped.startswith("Acceptance:"):
            self.explorer_summary_block = "acceptance"
            self._append_explorer_summary_section(line, "decision")
            return

        if self.explorer_summary_block == "comparison":
            if line.startswith("  "):
                self._append_explorer_summary_section(line, "comparison")
                return
            self.explorer_summary_block = ""

        if self.explorer_summary_block == "acceptance":
            if line.startswith("  "):
                self._append_explorer_summary_section(line, "decision")
                return
            self.explorer_summary_block = ""

        if self._is_explorer_summary_line(line):
            self._append_explorer_summary_section(line, self._explorer_summary_section_key(line))


    def _handle_stdout_payload(self, payload: str) -> None:
        self.stdout_parse_buffer += payload
        while "\n" in self.stdout_parse_buffer:
            line, self.stdout_parse_buffer = self.stdout_parse_buffer.split("\n", 1)
            self._handle_stdout_line(line + "\n")

    def _flush_stdout_parse_buffer(self) -> None:
        if self.stdout_parse_buffer:
            self._handle_stdout_line(self.stdout_parse_buffer)
            self.stdout_parse_buffer = ""

    def _handle_stdout_line(self, line: str) -> None:
        import re
        search_match = re.search(r"--- Search run (\d+)(?:/(\d+))?:", line)
        legacy_match = re.search(r"--- Run (\d+):", line) if not search_match else None

        if search_match or legacy_match:
            current_run = int((search_match or legacy_match).group(1))
            total_run_text = (search_match.group(2) if search_match else None)
            if total_run_text:
                try:
                    self.explorer_total_runs = int(total_run_text)
                except Exception:
                    pass
            self.explorer_current_run = current_run
            if self.explorer_current_run > 1:
                prev_run = self.explorer_current_run - 1
                msg = f"\nEnd of Explorer search run {prev_run}"
                if self.explorer_total_runs > 0:
                    msg += f" of {self.explorer_total_runs}"
                self._append_console(msg + "\n")
                self._append_explorer_summary_section(msg + "\n", "search_run")

            msg = f"\nStart of Explorer search run {self.explorer_current_run}"
            if self.explorer_total_runs > 0:
                msg += f" of {self.explorer_total_runs}"
            self._append_console(msg + "\n")
            self._append_explorer_summary_section(msg + "\n", "search_run")

        if "=== Batch Metadata Summary ===" in line and self.explorer_current_run > 0:
            msg = f"\nEnd of Explorer search run {self.explorer_current_run}"
            if self.explorer_total_runs > 0:
                msg += f" of {self.explorer_total_runs}"
            self._append_console(msg + "\n")
            self._append_explorer_summary_section(msg + "\n", "search_run")
            self.explorer_current_run = 0

        explorer_marker_index = line.find(EXPLORER_STATUS_MARKER)
        if explorer_marker_index != -1:
            json_text = line[explorer_marker_index + len(EXPLORER_STATUS_MARKER):].strip()
            try:
                event = json.loads(json_text)
            except Exception:
                self._append_console(line)
                return
            if isinstance(event, dict):
                self._handle_explorer_status_marker(event)
            return

        marker_index = line.find(ADJUST_EVENT_MARKER)
        if marker_index != -1:
            json_text = line[marker_index + len(ADJUST_EVENT_MARKER):].strip()
            try:
                event = json.loads(json_text)
            except Exception:
                self._append_console(line)
                return
            if isinstance(event, dict):
                self.adjust_events.append(event)
            return
        self._handle_explorer_summary_line(line)
        self._append_console(line)

    def _format_adjust_summary(self) -> str:
        if not self.adjust_events:
            return ""

        total = len(self.adjust_events)
        by_action: dict[str, int] = {}
        by_action_level: dict[str, dict[str, int]] = {}
        by_pair: dict[str, int] = {}
        by_trade: dict[str, int] = {}

        for event in self.adjust_events:
            action = str(event.get("action") or "unknown")
            level = str(event.get("level") or "unknown")
            pair = str(event.get("pair") or "unknown")
            trade_id = str(event.get("trade_id") if event.get("trade_id") is not None else "unknown")

            by_action[action] = by_action.get(action, 0) + 1
            by_action_level.setdefault(action, {})
            by_action_level[action][level] = by_action_level[action].get(level, 0) + 1
            by_pair[pair] = by_pair.get(pair, 0) + 1
            by_trade[trade_id] = by_trade.get(trade_id, 0) + 1

        lines = []
        lines.append("\n" + "-" * 100 + "\n")
        lines.append("Adjustment fill summary\n")
        lines.append(f"Total filled adjustments: {total}\n")

        if by_action:
            lines.append("By action:\n")
            for action in sorted(by_action):
                lines.append(f"  {action}: {by_action[action]}\n")

        if by_action_level:
            lines.append("By action and level:\n")
            for action in sorted(by_action_level):
                level_counts = by_action_level[action]
                parts = [f"{level}={level_counts[level]}" for level in sorted(level_counts)]
                lines.append(f"  {action}: " + ", ".join(parts) + "\n")

        if by_pair:
            lines.append("By pair:\n")
            for pair, count in sorted(by_pair.items(), key=lambda item: (-item[1], item[0])):
                lines.append(f"  {pair}: {count}\n")

        if by_trade:
            lines.append("Most active trades:\n")
            for trade_id, count in sorted(by_trade.items(), key=lambda item: (-item[1], item[0]))[:10]:
                lines.append(f"  trade_id={trade_id}: {count}\n")

        return "".join(lines)

    def _tail_hyperopt_results_worker(self, run_started_at: float, stop_event: threading.Event) -> None:
        explicit = self.hyperopt_results_file_var.get().strip()
        poll_ms = parse_int(self.hyperopt_results_poll_ms_var.get()) or 1000
        poll_s = max(0.2, poll_ms / 1000.0)
        target: Path | None = Path(explicit) if explicit else None
        position = 0
        announced = False
        decoder = codecs.getincrementaldecoder("utf-8")("replace")

        while not stop_event.is_set():
            if target is None:
                latest = self._latest_file(self._hyperopt_results_dir(), ["*.fthypt", "*.pickle"])
                if latest and latest.stat().st_mtime >= run_started_at - 10:
                    target = latest
                    position = latest.stat().st_size
            if target and target.exists():
                if not announced:
                    self.output_queue.put(("monitor", f"[results-monitor] Tailing {target}\n"))
                    announced = True
                try:
                    with open(target, "rb") as handle:
                        handle.seek(position)
                        chunk = handle.read()
                    if chunk:
                        position += len(chunk)
                        text_chunk = decoder.decode(chunk)
                        if text_chunk:
                            self.output_queue.put(("monitor", text_chunk.replace("\r", "\n")))
                except Exception as exc:
                    self.output_queue.put(("monitor", f"[results-monitor] Read failed: {exc}\n"))
                    return
            stop_event.wait(poll_s)

    def _review_common_args(self) -> list[str]:
        args: list[str] = []
        self.verbose_var.set(LOG_LEVEL_TO_VERBOSE.get(self.log_level_var.get(), self.verbose_var.get()))
        for _ in range(parse_int(self.verbose_var.get()) or 0):
            args.append("-v")
        if self.no_color_var.get():
            args.append("--no-color")
        if self.logfile_var.get().strip():
            args.extend(["--logfile", self.logfile_var.get().strip()])
        for config_file in self.config_editor.get_items():
            if config_file.strip():
                args.extend(["-c", config_file.strip()])
        if self.datadir_var.get().strip():
            args.extend(["--datadir", self.datadir_var.get().strip()])
        if self.userdir_var.get().strip():
            args.extend(["--userdir", self.userdir_var.get().strip()])
        return args

    def _review_hyperopt_filename(self) -> str | None:
        path = self.review_hyperopt_file_var.get().strip()
        if not path:
            return None
        return Path(path).name

    def _review_backtest_filename(self) -> str | None:
        path = self.review_backtest_file_var.get().strip()
        if not path:
            return None
        return str(Path(path))

    def _backtest_file_has_analysis_signals(self, filename: str) -> bool | None:
        path = Path(filename)
        if path.suffix.lower() != ".zip" or not path.is_file():
            return None
        try:
            with ZipFile(path) as zip_file:
                return any(name.endswith("_signals.pkl") for name in zip_file.namelist())
        except BadZipFile:
            return None

    def _run_review_command(self, command_args: list[str], title: str) -> None:
        if self.worker_thread and self.worker_thread.is_alive():
            messagebox.showinfo(APP_TITLE, "A run is already active. Wait for it to finish before launching a review command.")
            return
        self._append_review_console("\n" + "=" * 100 + "\n")
        preview = [self.python_exe_var.get().strip() or sys.executable, "-u", "-m", "freqtrade", *command_args]
        self._append_review_console(title + "\n")
        self._append_review_console(shell_join(preview) + "\n")
        self.worker_thread = threading.Thread(target=self._run_aux_subprocess, args=(command_args, "review"), daemon=True)
        self.worker_thread.start()

    def _run_aux_subprocess(self, command_args: list[str], queue_kind: str) -> None:
        python_exe = self.python_exe_var.get().strip() or sys.executable
        project_root = self.project_root_var.get().strip() or None
        env = utf8_subprocess_env()
        if project_root:
            current_pp = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = project_root + (os.pathsep + current_pp if current_pp else "")
        command = [python_exe, "-u", "-m", "freqtrade", *command_args]
        try:
            self.current_process = subprocess.Popen(
                command,
                cwd=project_root,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=False,
                bufsize=0,
                **self._subprocess_launch_kwargs(),
            )
            assert self.current_process.stdout is not None
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            stdout_fd = self.current_process.stdout.fileno()
            while True:
                chunk = os.read(stdout_fd, 4096)
                if not chunk:
                    break
                text_chunk = decoder.decode(chunk)
                if text_chunk:
                    self.output_queue.put((queue_kind, text_chunk.replace("\r", "\n")))
            final_text = decoder.decode(b"", final=True)
            if final_text:
                self.output_queue.put((queue_kind, final_text.replace("\r", "\n")))
            returncode = self.current_process.wait()
            self.output_queue.put((queue_kind, f"\nProcess finished with exit code {returncode}.\n"))
        except Exception:
            self.output_queue.put((queue_kind, traceback.format_exc() + "\n"))
        finally:
            self.current_process = None
            self.output_queue.put(("status", "idle"))

    def on_review_hyperopt_list(self) -> None:
        args = ["hyperopt-list", *self._review_common_args()]
        if self.review_hyperopt_best_var.get():
            args.append("--best")
        if self.review_hyperopt_profitable_var.get():
            args.append("--profitable")
        if self.review_hyperopt_print_json_var.get():
            args.append("--print-json")
        if self.review_hyperopt_no_details_var.get():
            args.append("--no-details")
        filename = self._review_hyperopt_filename()
        if filename:
            args.extend(["--hyperopt-filename", filename])
        self._run_review_command(args, "Hyperopt List")

    def on_review_hyperopt_show(self) -> None:
        args = ["hyperopt-show", *self._review_common_args()]
        if self.review_hyperopt_best_var.get():
            args.append("--best")
        if self.review_hyperopt_profitable_var.get():
            args.append("--profitable")
        if self.review_hyperopt_print_json_var.get():
            args.append("--print-json")
        if self.review_hyperopt_no_header_var.get():
            args.append("--no-header")
        filename = self._review_hyperopt_filename()
        if filename:
            args.extend(["--hyperopt-filename", filename])
        idx = self.review_hyperopt_index_var.get().strip()
        if idx:
            args.extend(["-n", idx])
        breakdown = self.review_hyperopt_breakdown_var.get().strip()
        if breakdown and breakdown != "none":
            args.extend(["--breakdown", breakdown])
        self._run_review_command(args, "Hyperopt Show")

    def on_review_hyperopt_show_best(self) -> None:
        self.review_hyperopt_best_var.set(True)
        if not self.review_hyperopt_index_var.get().strip():
            self.review_hyperopt_index_var.set("-1")
        self.on_review_hyperopt_show()

    def on_review_backtest_show(self) -> None:
        args = ["backtesting-show", *self._review_common_args()]
        filename = self._review_backtest_filename()
        if filename:
            args.extend(["--backtest-filename", filename])
        if self.review_backtest_show_pair_list_var.get():
            args.append("--show-pair-list")
        breakdown = self.review_backtest_breakdown_var.get().strip()
        if breakdown and breakdown != "none":
            args.extend(["--breakdown", breakdown])
        self._run_review_command(args, "Backtesting Show")

    def on_review_backtest_analysis(self) -> None:
        args = ["backtesting-analysis", *self._review_common_args()]
        filename = self._review_backtest_filename()
        if filename:
            has_signals = self._backtest_file_has_analysis_signals(filename)
            if has_signals is False:
                messagebox.showerror(
                    APP_TITLE,
                    "This backtest result does not contain signal data for Backtesting Analysis.\n\n"
                    "Run the backtest again with Export set to 'signals', then use the new result zip.",
                )
                return
            args.extend(["--backtest-filename", filename])
        groups = parse_token_list(self.review_analysis_groups_var.get())
        if groups:
            args.extend(["--analysis-groups", *groups])
        enters = parse_token_list(self.review_enter_reasons_var.get())
        if enters:
            args.extend(["--enter-reason-list", *enters])
        exits = parse_token_list(self.review_exit_reasons_var.get())
        if exits:
            args.extend(["--exit-reason-list", *exits])
        indicators = parse_token_list(self.review_indicator_list_var.get())
        if indicators:
            args.extend(["--indicator-list", *indicators])
        if self.review_entry_only_var.get():
            args.append("--entry-only")
        if self.review_exit_only_var.get():
            args.append("--exit-only")
        if self.review_rejected_signals_var.get():
            args.append("--rejected-signals")
        if self.review_analysis_to_csv_var.get():
            args.append("--analysis-to-csv")
        csv_path = self.review_analysis_csv_path_var.get().strip()
        if csv_path:
            args.extend(["--analysis-csv-path", csv_path])
        self._run_review_command(args, "Backtesting Analysis")

# ----------------------------------------------------------------------
# Utility helpers
# ----------------------------------------------------------------------
def discover_hyperopt_losses(project_root: Path, userdir: Path | None) -> list[HyperoptLossInfo]:
    """Discover HyperOpt loss classes without importing user code."""

    loss_dirs: list[tuple[Path, str]] = []
    root_candidates = [
        project_root,
        Path.cwd(),
        Path(__file__).resolve().parent.parent,
    ]
    seen_dirs: set[str] = set()

    def add_dir(path: Path, label: str) -> None:
        try:
            resolved = str(path.resolve())
        except Exception:
            resolved = str(path)
        if resolved in seen_dirs or not path.is_dir():
            return
        seen_dirs.add(resolved)
        loss_dirs.append((path, label))

    for root in root_candidates:
        add_dir(root / "freqtrade" / "optimize" / "hyperopt_loss", "Freqtrade loss folder")
        add_dir(root / "user_data" / "hyperopts", "User hyperopts")
    if userdir is not None:
        add_dir(userdir / "hyperopts", "User hyperopts")

    found: dict[str, HyperoptLossInfo] = {}
    for folder, source in loss_dirs:
        for file_path in sorted(folder.rglob("*.py")):
            for info in extract_hyperopt_loss_infos(file_path, source):
                found.setdefault(info.name, info)

    for name in HYPEROPT_LOSS_VALUES:
        found.setdefault(
            name,
            HyperoptLossInfo(
                name=name,
                source="Built-in",
                description=BUILTIN_HYPEROPT_LOSS_DESCRIPTIONS.get(name, "Built-in Freqtrade HyperOpt loss."),
                path=None,
            ),
        )

    builtin_order = {name: index for index, name in enumerate(HYPEROPT_LOSS_VALUES)}
    return sorted(
        found.values(),
        key=lambda item: (builtin_order.get(item.name, len(builtin_order)), item.name.lower()),
    )


def extract_hyperopt_loss_infos(file_path: Path, source: str) -> list[HyperoptLossInfo]:
    try:
        text = file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        try:
            text = file_path.read_text(encoding="utf-8-sig")
        except Exception:
            return []
    except Exception:
        return []

    try:
        tree = ast.parse(text, filename=str(file_path))
    except Exception:
        return []

    module_doc = ast.get_docstring(tree) or ""
    infos: list[HyperoptLossInfo] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if not looks_like_hyperopt_loss_class(node):
            continue
        class_doc = ast.get_docstring(node) or ""
        description = "\n\n".join(part.strip() for part in [class_doc, module_doc] if part.strip())
        if not description:
            description = BUILTIN_HYPEROPT_LOSS_DESCRIPTIONS.get(
                node.name,
                "Custom HyperOpt loss class. Add a class docstring to make this popup more helpful.",
            )
        elif node.name in BUILTIN_HYPEROPT_LOSS_DESCRIPTIONS:
            description = f"{BUILTIN_HYPEROPT_LOSS_DESCRIPTIONS[node.name]}\n\n{description}"
        infos.append(
            HyperoptLossInfo(
                name=node.name,
                source=source,
                description=description,
                path=str(file_path),
            )
        )
    return infos


def looks_like_hyperopt_loss_class(node: ast.ClassDef) -> bool:
    if node.name.startswith("_") or node.name == "IHyperOptLoss":
        return False
    if node.name.endswith("HyperOptLoss"):
        return True
    for base in node.bases:
        base_name = dotted_ast_name(base)
        if base_name.endswith("IHyperOptLoss"):
            return True
    return any(
        isinstance(item, ast.FunctionDef) and item.name == "hyperopt_loss_function"
        for item in node.body
    )


def dotted_ast_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = dotted_ast_name(node.value)
        return f"{parent}.{node.attr}" if parent else node.attr
    if isinstance(node, ast.Subscript):
        return dotted_ast_name(node.value)
    return ""


def extract_class_names(file_path: str) -> list[str]:
    """Return top-level class names from a Python strategy file.

    A full inheritance analysis is unnecessary here. The launcher only needs a
    quick class picker so the user does not have to type the class name every
    time.
    """

    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            tree = ast.parse(handle.read(), filename=file_path)
    except Exception:
        return []

    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            names.append(node.name)
    return names


def derive_module_stem(file_path: str) -> str:
    return Path(file_path).stem


def parse_token_list(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in text.replace(",", " ").split():
        token = raw.strip()
        if token:
            tokens.append(token)
    return tokens


def parse_int(text: str) -> int | None:
    text = text.strip()
    if not text:
        return None
    return int(text)


def append_if_value(args: list[str], option: str, value: str) -> None:
    value = value.strip()
    if value:
        args.extend([option, value])


def shell_join(parts: Iterable[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    app = FreqtradeLauncher()
    app.mainloop()


if __name__ == "__main__":
    main()

