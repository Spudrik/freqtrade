from __future__ import annotations

import argparse
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


USER_DATA_DIR = Path(__file__).resolve().parents[3]
REPORTS_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "reports"
DEFAULT_CANDIDATES = REPORTS_DIR / "trader_sieve_runtime_candidate_leads.csv"


def main() -> int:
    parser = argparse.ArgumentParser(description="Export BTC decision-hour trades/signals from borrowed sieve-runtime candidate archives.")
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--pair", default="BTC/USDT:USDT")
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--min-btc-trades", type=int, default=5)
    parser.add_argument("--tag", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    args = parser.parse_args()

    if not args.candidates.exists():
        raise FileNotFoundError(args.candidates)
    candidates = pd.read_csv(args.candidates)
    selected = select_candidates(candidates, top=args.top, min_btc_trades=args.min_btc_trades)
    trades = export_trades(selected, pair=args.pair)
    signals = export_signals(trades)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = safe_name(args.tag)
    selected_path = args.output_dir / f"trader_sieve_runtime_signal_exports_{tag}_selected.csv"
    trades_path = args.output_dir / f"trader_sieve_runtime_signal_exports_{tag}_trades.csv"
    signals_path = args.output_dir / f"trader_sieve_runtime_signal_exports_{tag}_signals.parquet"
    markdown_path = args.output_dir / f"trader_sieve_runtime_signal_exports_{tag}.md"
    meta_path = args.output_dir / f"trader_sieve_runtime_signal_exports_{tag}_meta.json"

    selected.to_csv(selected_path, index=False)
    trades.to_csv(trades_path, index=False)
    signals.to_parquet(signals_path, index=False)
    markdown_path.write_text(markdown_report(selected, trades, signals, args), encoding="utf-8")
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pair": args.pair,
        "candidate_rows": int(len(candidates)),
        "selected_candidates": int(len(selected)),
        "trade_rows": int(len(trades)),
        "signal_rows": int(len(signals)),
        "alignment": "signal_date/date is open_date minus one hour; open_date is the actual backtest entry candle.",
        "outputs": {
            "selected": str(selected_path),
            "trades": str(trades_path),
            "signals": str(signals_path),
            "markdown": str(markdown_path),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


def select_candidates(candidates: pd.DataFrame, top: int, min_btc_trades: int) -> pd.DataFrame:
    frame = candidates.copy()
    if "pair_trade_count" in frame.columns:
        frame = frame[pd.to_numeric(frame["pair_trade_count"], errors="coerce").fillna(0).ge(min_btc_trades)]
    frame = frame.sort_values(["candidate_score", "strategy"], ascending=[False, True])
    return frame.head(top).reset_index(drop=True)


def export_trades(candidates: pd.DataFrame, pair: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, candidate in candidates.iterrows():
        backtest_file = Path(str(candidate.get("backtest_file", "")))
        if not backtest_file.exists():
            continue
        for trade in read_pair_trades(backtest_file, pair=pair):
            open_date = pd.to_datetime(trade.get("open_date"), utc=True, errors="coerce")
            close_date = pd.to_datetime(trade.get("close_date"), utc=True, errors="coerce")
            if pd.isna(open_date):
                continue
            signal_date = open_date - pd.Timedelta(hours=1)
            direction = "short" if bool(trade.get("is_short")) else "long"
            strategy = str(candidate.get("strategy"))
            rows.append(
                {
                    "rule_id": strategy,
                    "strategy": strategy,
                    "signal_date": signal_date,
                    "open_date": open_date,
                    "close_date": close_date,
                    "pair": trade.get("pair"),
                    "direction": direction,
                    "net_return": floatish(trade.get("profit_ratio")),
                    "profit_ratio": floatish(trade.get("profit_ratio")),
                    "exit_reason": trade.get("exit_reason"),
                    "open_rate": floatish(trade.get("open_rate")),
                    "close_rate": floatish(trade.get("close_rate")),
                    "trade_duration_minutes": intish(trade.get("trade_duration")),
                    "candidate_score": floatish(candidate.get("candidate_score")),
                    "trader_question": candidate.get("trader_question"),
                    "visible_market_state": candidate.get("visible_market_state"),
                    "source_detail_blocks": candidate.get("source_detail_blocks"),
                    "evidence_paths": candidate.get("evidence_paths"),
                    "backtest_file": str(backtest_file),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.sort_values(["signal_date", "rule_id"]).reset_index(drop=True)


def export_signals(trades: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame(columns=["date", "rule_id", "enter_long", "enter_short", "direction", "signal_source"])
    signals = trades[["signal_date", "rule_id", "direction", "candidate_score"]].copy()
    signals = signals.rename(columns={"signal_date": "date"})
    signals["enter_long"] = signals["direction"].eq("long").astype(int)
    signals["enter_short"] = signals["direction"].eq("short").astype(int)
    signals["enter_tag"] = signals["rule_id"]
    signals["concept_id"] = signals["rule_id"]
    signals["threshold"] = 0.0
    signals["signal_score"] = pd.to_numeric(signals["candidate_score"], errors="coerce").fillna(0.0)
    signals["signal_source"] = "sieve_runtime_btc_backtest_archive"
    signals = signals.sort_values(["date", "rule_id"]).drop_duplicates(["date", "rule_id"], keep="first")
    return signals.reset_index(drop=True)


def read_pair_trades(backtest_file: Path, pair: str) -> list[dict[str, Any]]:
    try:
        with zipfile.ZipFile(backtest_file) as archive:
            result_name = first_result_json_name(archive)
            if not result_name:
                return []
            data = json.loads(archive.read(result_name))
    except (OSError, KeyError, json.JSONDecodeError, zipfile.BadZipFile):
        return []
    strategies = data.get("strategy", {})
    if not strategies:
        return []
    result = next(iter(strategies.values()))
    return [trade for trade in result.get("trades", []) if str(trade.get("pair")) == pair]


def first_result_json_name(archive: zipfile.ZipFile) -> str:
    names = []
    for name in archive.namelist():
        if not name.endswith(".json") or "_config" in name:
            continue
        stem = Path(name).stem
        if re.search(r"_[A-Z][A-Za-z0-9]+$", stem):
            continue
        names.append(name)
    if names:
        return sorted(names)[0]
    fallback = [name for name in archive.namelist() if name.endswith(".json") and "_config" not in name]
    return sorted(fallback)[0] if fallback else ""


def markdown_report(selected: pd.DataFrame, trades: pd.DataFrame, signals: pd.DataFrame, args: argparse.Namespace) -> str:
    lines = [
        "# Sieve Runtime BTC Signal Exports",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "Purpose: convert borrowed sieve BTC backtest trades into decision-hour signals for confluence testing.",
        "",
        "Alignment: `date`/`signal_date` is one hour before the backtest entry `open_date`, so confluence features describe what was visible before entry.",
        "",
        f"- Pair: {args.pair}",
        f"- Selected candidate strategies: {len(selected)}",
        f"- BTC trade rows: {len(trades)}",
        f"- Signal rows: {len(signals)}",
        "",
    ]
    if not selected.empty:
        cols = ["strategy", "direction", "pair_trade_count", "pair_winrate", "pair_profit_total", "pair_profit_factor", "candidate_score"]
        lines.extend(["## Selected Candidates", "", selected[[c for c in cols if c in selected.columns]].head(40).to_markdown(index=False), ""])
    if not trades.empty:
        by_rule = trades.groupby("rule_id").agg(
            trades=("net_return", "size"),
            win_rate=("net_return", lambda s: float((s > 0).mean())),
            avg_return=("net_return", "mean"),
            total_return_sum=("net_return", "sum"),
        ).reset_index().sort_values("trades", ascending=False)
        lines.extend(["## Exported Trade Summary", "", by_rule.head(40).to_markdown(index=False), ""])
    return "\n".join(lines) + "\n"


def safe_name(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean[:180] or "unknown"


def floatish(value: Any) -> float:
    try:
        if value is None or pd.isna(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def intish(value: Any) -> int:
    try:
        if value is None or pd.isna(value):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
