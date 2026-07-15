from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from production_alpha_backtest_audit import load_backtest, normalize_trades, safe_name


ROOT = Path(__file__).resolve().parents[4]
REPORTS_DIR = ROOT / "user_data" / "research_news_data" / "context_features" / "reports"
DEFAULT_FEATURES = REPORTS_DIR.parent / "confluence_cache" / "trader_confluence_1h_latest.parquet"
DEFAULT_BACKTEST = ROOT / "user_data" / "backtest_results" / "backtest-result-2026-06-07_00-46-44.zip"
DEFAULT_SIGNALS = REPORTS_DIR / "trading_lead_signals_20260607_current_strongest_no_poc_plus_complete_pattern_rare_current_priority.parquet"


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit ignored overlapping signals as add/hold/reduce/exit evidence.")
    parser.add_argument("--backtest-zip", type=Path, default=DEFAULT_BACKTEST)
    parser.add_argument("--signals", type=Path, default=DEFAULT_SIGNALS)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--tag", default=datetime.now(timezone.utc).strftime("%Y%m%d_signal_stacking"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    plan = {
        "mode": "setup-only unless --execute is supplied",
        "trader_question": "Do overlapping signals during an open trade contain useful add/hold/reduce/exit evidence?",
        "backtest_zip": str(args.backtest_zip),
        "signals": str(args.signals),
        "features": str(args.features),
    }
    if not args.execute:
        print(json.dumps(plan, indent=2))
        return 0

    for path in (args.backtest_zip, args.signals, args.features):
        if not path.exists():
            raise FileNotFoundError(path)

    trades = normalize_trades(pd.DataFrame(load_backtest(args.backtest_zip)["trades"]))
    signals = load_signals(args.signals)
    features = load_features(args.features)

    trade_audit, signal_audit = audit_stacking(trades, signals, features)
    trade_summary = summarize_trade_audit(trade_audit)
    signal_summary = summarize_signal_audit(signal_audit)
    management_summary = summarize_management_overlay(trade_audit, signal_audit, features)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = safe_name(args.tag)
    trade_path = args.output_dir / f"production_alpha_{tag}_trade_audit.csv"
    signal_path = args.output_dir / f"production_alpha_{tag}_signal_audit.csv"
    trade_summary_path = args.output_dir / f"production_alpha_{tag}_trade_summary.csv"
    signal_summary_path = args.output_dir / f"production_alpha_{tag}_signal_summary.csv"
    management_summary_path = args.output_dir / f"production_alpha_{tag}_management_summary.csv"
    md_path = args.output_dir / f"production_alpha_{tag}.md"
    meta_path = args.output_dir / f"production_alpha_{tag}_meta.json"

    trade_audit.to_csv(trade_path, index=False)
    signal_audit.to_csv(signal_path, index=False)
    trade_summary.to_csv(trade_summary_path, index=False)
    signal_summary.to_csv(signal_summary_path, index=False)
    management_summary.to_csv(management_summary_path, index=False)
    md_path.write_text(markdown_report(trade_summary, signal_summary, management_summary), encoding="utf-8")
    meta = {
        **plan,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trades": int(len(trades)),
        "signals": int(len(signals)),
        "overlap_events": int(len(signal_audit)),
        "outputs": {
            "trade_audit": str(trade_path),
            "signal_audit": str(signal_path),
            "trade_summary": str(trade_summary_path),
            "signal_summary": str(signal_summary_path),
            "management_summary": str(management_summary_path),
            "markdown": str(md_path),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


def load_signals(path: Path) -> DataFrame:
    frame = pd.read_parquet(path)
    required = {"date", "enter_long", "enter_short", "rule_id", "enter_tag", "signal_score"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Signal file missing columns: {missing}")
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    frame["signal_direction"] = np.where(pd.to_numeric(frame["enter_short"], errors="coerce").fillna(0).gt(0), "short", "long")
    frame["signal_score"] = pd.to_numeric(frame["signal_score"], errors="coerce").fillna(0.0)
    return frame


def load_features(path: Path) -> DataFrame:
    frame = pd.read_parquet(path, columns=["date", "open", "high", "low", "close"])
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    return frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def audit_stacking(trades: DataFrame, signals: DataFrame, features: DataFrame) -> tuple[DataFrame, DataFrame]:
    trades = trades.dropna(subset=["open_date", "close_date"]).sort_values("open_date").reset_index(drop=True)
    signal_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []
    for trade_id, trade in trades.iterrows():
        open_date = pd.to_datetime(trade["open_date"], utc=True)
        close_date = pd.to_datetime(trade["close_date"], utc=True)
        direction = "short" if bool(trade.get("is_short", False)) else "long"
        overlaps = signals.loc[(signals["date"] > open_date) & (signals["date"] < close_date)].copy()
        overlaps["relationship"] = np.where(overlaps["signal_direction"].eq(direction), "same_direction", "opposite_direction")
        for signal in overlaps.itertuples(index=False):
            signal_rows.append(signal_overlap_row(trade_id, trade, signal, features))
        same = overlaps.loc[overlaps["relationship"].eq("same_direction")]
        opposite = overlaps.loc[overlaps["relationship"].eq("opposite_direction")]
        trade_rows.append(
            {
                "trade_id": int(trade_id),
                "enter_tag": trade.get("enter_tag"),
                "direction": direction,
                "open_date": open_date,
                "close_date": close_date,
                "profit_ratio": float(trade.get("profit_ratio", 0.0)),
                "profit_abs": float(trade.get("profit_abs", 0.0)),
                "win": bool(trade.get("profit_ratio", 0.0) > 0.0),
                "same_signal_count": int(len(same)),
                "opposite_signal_count": int(len(opposite)),
                "same_score_sum": float(same["signal_score"].sum()) if not same.empty else 0.0,
                "opposite_score_sum": float(opposite["signal_score"].sum()) if not opposite.empty else 0.0,
                "net_same_minus_opposite_score": float(same["signal_score"].sum() - opposite["signal_score"].sum()),
                "first_same_hours_after_open": first_hours_after_open(open_date, same),
                "first_opposite_hours_after_open": first_hours_after_open(open_date, opposite),
            }
        )
    return pd.DataFrame(trade_rows), pd.DataFrame(signal_rows)


def signal_overlap_row(trade_id: int, trade: Any, signal: Any, features: DataFrame) -> dict[str, Any]:
    signal_date = pd.to_datetime(signal.date, utc=True)
    close_date = pd.to_datetime(trade["close_date"], utc=True)
    direction = "short" if bool(trade.get("is_short", False)) else "long"
    relationship = "same_direction" if signal.signal_direction == direction else "opposite_direction"
    return {
        "trade_id": int(trade_id),
        "trade_enter_tag": trade.get("enter_tag"),
        "trade_direction": direction,
        "trade_open_date": pd.to_datetime(trade["open_date"], utc=True),
        "trade_close_date": close_date,
        "trade_profit_ratio": float(trade.get("profit_ratio", 0.0)),
        "trade_win": bool(trade.get("profit_ratio", 0.0) > 0.0),
        "signal_date": signal_date,
        "signal_rule_id": str(signal.rule_id),
        "signal_direction": str(signal.signal_direction),
        "relationship": relationship,
        "signal_score": float(signal.signal_score),
        "hours_after_open": float((signal_date - pd.to_datetime(trade["open_date"], utc=True)).total_seconds() / 3600.0),
        "remaining_trade_return": direction_return(features, signal_date, close_date, direction),
        "future_3h_return": horizon_return(features, signal_date, direction, 3),
        "future_6h_return": horizon_return(features, signal_date, direction, 6),
        "future_12h_return": horizon_return(features, signal_date, direction, 12),
    }


def first_hours_after_open(open_date: pd.Timestamp, rows: DataFrame) -> float | None:
    if rows.empty:
        return None
    first = pd.to_datetime(rows["date"].min(), utc=True)
    return float((first - open_date).total_seconds() / 3600.0)


def direction_return(features: DataFrame, start: pd.Timestamp, end: pd.Timestamp, direction: str) -> float:
    indexed = features.set_index("date")
    if start not in indexed.index:
        rows = indexed.loc[indexed.index >= start]
        if rows.empty:
            return np.nan
        start = rows.index[0]
    if end not in indexed.index:
        rows = indexed.loc[indexed.index <= end]
        if rows.empty:
            return np.nan
        end = rows.index[-1]
    start_price = float(indexed.loc[start, "close"])
    end_price = float(indexed.loc[end, "close"])
    return close_return(direction, start_price, end_price)


def horizon_return(features: DataFrame, start: pd.Timestamp, direction: str, hours: int) -> float:
    end = start + pd.Timedelta(hours=hours)
    return direction_return(features, start, end, direction)


def close_return(direction: str, start_price: float, end_price: float) -> float:
    if not np.isfinite(start_price) or not np.isfinite(end_price) or start_price <= 0 or end_price <= 0:
        return np.nan
    if direction == "short":
        return start_price / end_price - 1.0
    return end_price / start_price - 1.0


def summarize_trade_audit(frame: DataFrame) -> DataFrame:
    rows: list[dict[str, Any]] = []
    for group_name, group in grouped_trade_views(frame):
        rows.append(summary_row(group_name, group))
    return pd.DataFrame(rows).sort_values(["scope", "relationship_bucket"]).reset_index(drop=True)


def grouped_trade_views(frame: DataFrame) -> list[tuple[tuple[str, str], DataFrame]]:
    views: list[tuple[tuple[str, str], DataFrame]] = []
    buckets = {
        "no_overlap": frame["same_signal_count"].eq(0) & frame["opposite_signal_count"].eq(0),
        "same_only": frame["same_signal_count"].gt(0) & frame["opposite_signal_count"].eq(0),
        "opposite_only": frame["same_signal_count"].eq(0) & frame["opposite_signal_count"].gt(0),
        "same_and_opposite": frame["same_signal_count"].gt(0) & frame["opposite_signal_count"].gt(0),
        "net_same_positive": frame["net_same_minus_opposite_score"].gt(0),
        "net_opposite_positive": frame["net_same_minus_opposite_score"].lt(0),
    }
    for bucket, mask in buckets.items():
        views.append((("all_trades", bucket), frame.loc[mask]))
    for direction in ("long", "short"):
        side = frame.loc[frame["direction"].eq(direction)]
        for bucket, mask in buckets.items():
            views.append(((direction, bucket), side.loc[mask.reindex(side.index, fill_value=False)]))
    return views


def summary_row(group_name: tuple[str, str], group: DataFrame) -> dict[str, Any]:
    scope, bucket = group_name
    wins = group.loc[group["profit_ratio"].gt(0), "profit_ratio"].sum()
    losses = -group.loc[group["profit_ratio"].lt(0), "profit_ratio"].sum()
    return {
        "scope": scope,
        "relationship_bucket": bucket,
        "trades": int(len(group)),
        "win_rate": float(group["win"].mean()) if len(group) else np.nan,
        "avg_profit_ratio": float(group["profit_ratio"].mean()) if len(group) else np.nan,
        "total_profit_ratio_sum": float(group["profit_ratio"].sum()) if len(group) else 0.0,
        "profit_factor_ratio_sum": float(wins / losses) if losses else np.nan,
        "avg_same_signal_count": float(group["same_signal_count"].mean()) if len(group) else np.nan,
        "avg_opposite_signal_count": float(group["opposite_signal_count"].mean()) if len(group) else np.nan,
        "avg_net_score": float(group["net_same_minus_opposite_score"].mean()) if len(group) else np.nan,
    }


def summarize_signal_audit(frame: DataFrame) -> DataFrame:
    if frame.empty:
        return DataFrame()
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(["trade_direction", "relationship"], dropna=False):
        rows.append(signal_summary_row(keys, group))
    for relationship, group in frame.groupby("relationship", dropna=False):
        rows.append(signal_summary_row(("all", relationship), group))
    return pd.DataFrame(rows).sort_values(["trade_direction", "relationship"]).reset_index(drop=True)


def summarize_management_overlay(trades: DataFrame, signals: DataFrame, features: DataFrame) -> DataFrame:
    first_same = first_signal_by_relationship(signals, "same_direction")
    first_opp = first_signal_by_relationship(signals, "opposite_direction")
    frame = trades.copy().sort_values("open_date").reset_index(drop=True)
    frame = frame.merge(first_same, on="trade_id", how="left", suffixes=("", "_same"))
    frame = frame.merge(first_opp, on="trade_id", how="left", suffixes=("", "_opposite"))
    rows: list[dict[str, Any]] = []
    specs = {
        "baseline_1x": lambda r: float(r["profit_ratio"]),
        "add_0p25_after_first_same": lambda r: add_after_same(r, 0.25),
        "add_0p50_after_first_same": lambda r: add_after_same(r, 0.50),
        "exit_at_first_opposite": lambda r: exit_at_opposite(r, features),
        "reduce_half_after_first_opposite": lambda r: reduce_after_opposite(r, features, 0.50),
        "add_0p25_same_reduce_half_opposite": lambda r: reduce_after_opposite_value(add_after_same(r, 0.25), r, features, 0.50),
        "add_0p50_same_reduce_half_opposite": lambda r: reduce_after_opposite_value(add_after_same(r, 0.50), r, features, 0.50),
    }
    baseline: dict[str, Any] | None = None
    for overlay_id, builder in specs.items():
        returns = frame.apply(builder, axis=1).replace([np.inf, -np.inf], np.nan).fillna(frame["profit_ratio"])
        row = overlay_summary(overlay_id, returns, frame)
        if baseline is None:
            baseline = row
        else:
            row["return_delta"] = float(row["compounded_return"]) - float(baseline["compounded_return"])
            row["drawdown_delta"] = float(row["max_drawdown"]) - float(baseline["max_drawdown"])
            row["profit_factor_delta"] = clean_float(row["profit_factor"]) - clean_float(baseline["profit_factor"])
            row["verdict"] = overlay_verdict(row)
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["verdict", "compounded_return"], ascending=[True, False]).reset_index(drop=True)


def first_signal_by_relationship(signals: DataFrame, relationship: str) -> DataFrame:
    if signals.empty:
        return DataFrame({"trade_id": pd.Series(dtype="int64")})
    subset = signals.loc[signals["relationship"].eq(relationship)].copy()
    if subset.empty:
        return DataFrame({"trade_id": pd.Series(dtype="int64")})
    subset = subset.sort_values(["trade_id", "signal_date"]).drop_duplicates("trade_id", keep="first")
    prefix = "same" if relationship == "same_direction" else "opposite"
    return subset[
        [
            "trade_id",
            "signal_date",
            "remaining_trade_return",
            "signal_score",
        ]
    ].rename(
        columns={
            "signal_date": f"{prefix}_signal_date",
            "remaining_trade_return": f"{prefix}_remaining_return",
            "signal_score": f"{prefix}_signal_score",
        }
    )


def add_after_same(row: Any, fraction: float) -> float:
    base = float(row["profit_ratio"])
    remaining = clean_float(row.get("same_remaining_return"))
    if remaining == 0.0 and pd.isna(row.get("same_signal_date")):
        return base
    return base + fraction * remaining


def exit_at_opposite(row: Any, features: DataFrame) -> float:
    if pd.isna(row.get("opposite_signal_date")):
        return float(row["profit_ratio"])
    return clean_float(direction_return(features, row["open_date"], row["opposite_signal_date"], str(row["direction"])))


def reduce_after_opposite(row: Any, features: DataFrame, fraction_remaining: float) -> float:
    if pd.isna(row.get("opposite_signal_date")):
        return float(row["profit_ratio"])
    prior = clean_float(direction_return(features, row["open_date"], row["opposite_signal_date"], str(row["direction"])))
    remaining = clean_float(row.get("opposite_remaining_return"))
    return (1.0 + prior) * (1.0 + fraction_remaining * remaining) - 1.0


def reduce_after_opposite_value(current_value: float, row: Any, features: DataFrame, fraction_remaining: float) -> float:
    if pd.isna(row.get("opposite_signal_date")):
        return current_value
    base = float(row["profit_ratio"])
    reduced = reduce_after_opposite(row, features, fraction_remaining)
    return current_value - base + reduced


def overlay_summary(overlay_id: str, returns: pd.Series, trades: DataFrame) -> dict[str, Any]:
    returns = pd.to_numeric(returns, errors="coerce").fillna(0.0)
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    wins = returns[returns.gt(0)]
    losses = returns[returns.lt(0)]
    return {
        "overlay_id": overlay_id,
        "trades": int(len(returns)),
        "same_overlap_trades": int(trades["same_signal_count"].gt(0).sum()),
        "opposite_overlap_trades": int(trades["opposite_signal_count"].gt(0).sum()),
        "win_rate": float(returns.gt(0).mean()) if len(returns) else np.nan,
        "avg_return": float(returns.mean()) if len(returns) else np.nan,
        "compounded_return": float(equity.iloc[-1] - 1.0) if len(equity) else 0.0,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "profit_factor": float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 0 else np.nan,
        "return_delta": 0.0,
        "drawdown_delta": 0.0,
        "profit_factor_delta": 0.0,
        "verdict": "baseline" if overlay_id == "baseline_1x" else "unclassified",
    }


def overlay_verdict(row: dict[str, Any]) -> str:
    improved_return = float(row["return_delta"]) > 0.0
    improved_drawdown = float(row["drawdown_delta"]) >= -0.001
    improved_pf = float(row["profit_factor_delta"]) > 0.05
    if improved_return and improved_drawdown and improved_pf:
        return "watchlist"
    if improved_return and improved_pf:
        return "return_helped_check_drawdown"
    return "reject_for_now"


def signal_summary_row(keys: tuple[Any, Any], group: DataFrame) -> dict[str, Any]:
    direction, relationship = keys
    return {
        "trade_direction": direction,
        "relationship": relationship,
        "signals": int(len(group)),
        "parent_trade_win_rate": float(group["trade_win"].mean()),
        "avg_parent_trade_profit": float(group["trade_profit_ratio"].mean()),
        "avg_remaining_trade_return": float(pd.to_numeric(group["remaining_trade_return"], errors="coerce").mean()),
        "avg_future_3h_return": float(pd.to_numeric(group["future_3h_return"], errors="coerce").mean()),
        "avg_future_6h_return": float(pd.to_numeric(group["future_6h_return"], errors="coerce").mean()),
        "avg_future_12h_return": float(pd.to_numeric(group["future_12h_return"], errors="coerce").mean()),
        "median_hours_after_open": float(pd.to_numeric(group["hours_after_open"], errors="coerce").median()),
        "avg_signal_score": float(pd.to_numeric(group["signal_score"], errors="coerce").mean()),
    }


def markdown_report(trade_summary: DataFrame, signal_summary: DataFrame, management_summary: DataFrame) -> str:
    lines = [
        "# Production Alpha Signal Stacking Audit",
        "",
        "## Trader Question",
        "",
        "When a trade is already open and more signals appear, are those signals useful evidence for adding, holding, reducing, or exiting?",
        "",
        "## What We Looked At",
        "",
        "1. Trades from the current promoted Sieve-current-best branch.",
        "2. Candidate signal rows that appeared while each trade was already open.",
        "3. Same-direction signals as possible add/hold evidence.",
        "4. Opposite-direction signals as possible reduce/exit evidence.",
        "",
        "## Trade-Level Summary",
        "",
        trade_summary.to_markdown(index=False),
        "",
        "## Signal-Level Summary",
        "",
        signal_summary.to_markdown(index=False) if not signal_summary.empty else "No overlap signals found.",
        "",
        "## Lightweight Management Simulation",
        "",
        management_summary.to_markdown(index=False),
        "",
        "## Plain-English Reading",
        "",
    ]
    lines.extend(plain_english_findings(trade_summary, signal_summary))
    return "\n".join(lines) + "\n"


def plain_english_findings(trade_summary: DataFrame, signal_summary: DataFrame) -> list[str]:
    findings: list[str] = []
    all_rows = trade_summary.loc[trade_summary["scope"].eq("all_trades")]
    same_only = one_row(all_rows, "same_only")
    opposite_only = one_row(all_rows, "opposite_only")
    no_overlap = one_row(all_rows, "no_overlap")
    if same_only is not None and no_overlap is not None:
        findings.append(
            "1. Same-direction overlap is useful only if its win rate and average return beat no-overlap trades. "
            f"Here same-only win rate was {fmt_pct(same_only['win_rate'])} versus no-overlap {fmt_pct(no_overlap['win_rate'])}."
        )
    if opposite_only is not None and no_overlap is not None:
        findings.append(
            "2. Opposite-direction overlap is useful as a warning only if those trades were worse. "
            f"Here opposite-only win rate was {fmt_pct(opposite_only['win_rate'])} versus no-overlap {fmt_pct(no_overlap['win_rate'])}."
        )
    if not signal_summary.empty:
        for _, row in signal_summary.iterrows():
            findings.append(
                f"3. {row['relationship']} signals during {row['trade_direction']} trades had average remaining trade-direction return "
                f"{fmt_pct(row['avg_remaining_trade_return'])} over {int(row['signals'])} signal events."
            )
    findings.append("4. Verdict should be based on direction-specific rows, not the all-trade average, because long and short failure modes differ.")
    return findings


def one_row(frame: DataFrame, bucket: str) -> dict[str, Any] | None:
    rows = frame.loc[frame["relationship_bucket"].eq(bucket)]
    if rows.empty:
        return None
    return rows.iloc[0].to_dict()


def clean_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(result):
        return 0.0
    return result


def fmt_pct(value: Any) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(value):
        return "n/a"
    return f"{value * 100:.2f}%"


if __name__ == "__main__":
    raise SystemExit(main())
