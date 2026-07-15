from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

from trader_rule_backtests import export_combined_signals, fmt, fmt_pct, num, resolve_exit, safe_name, summarize_trades


USER_DATA_DIR = Path(__file__).resolve().parents[3]
REPORTS_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "reports"
DEFAULT_FEATURES = USER_DATA_DIR / "research_news_data" / "context_features" / "confluence_cache" / "trader_confluence_1h_latest.parquet"
DEFAULT_SIGNALS = REPORTS_DIR / "trading_lead_signals_orderbook_support_removal_short_refinement_20260605_selected.parquet"


@dataclass(frozen=True)
class ExitSpec:
    exit_id: str
    hold_hours: int
    stop_loss: float
    take_profit: float


def main() -> int:
    parser = argparse.ArgumentParser(description="Tailored exit/risk research for orderbook support-removal short leads.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--signals", type=Path, default=DEFAULT_SIGNALS)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--tag", default="support_removal_exit_risk_20260605")
    parser.add_argument("--round-trip-fee", type=float, default=0.0010)
    parser.add_argument("--slippage", type=float, default=0.0002)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if not args.execute:
        print(json.dumps({"mode": "setup-only", "signals": str(args.signals), "features": str(args.features)}, indent=2))
        return 0
    if not args.features.exists():
        raise FileNotFoundError(args.features)
    if not args.signals.exists():
        raise FileNotFoundError(args.signals)

    features = load_features(args.features)
    signals = pd.read_parquet(args.signals)
    signals["date"] = pd.to_datetime(signals["date"], utc=True, errors="coerce")
    signals = signals.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="first").reset_index(drop=True)

    exit_summary, exit_trades, selected = run_exit_grid(features, signals, args.round_trip_fee, args.slippage)
    selected_trades = simulate_selected(features, signals, selected, args.round_trip_fee, args.slippage)
    risk_summary = run_risk_overlays(selected_trades, features)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = safe_name(args.tag)
    exit_summary_path = args.output_dir / f"trading_lead_{tag}_exit_summary.csv"
    exit_trades_path = args.output_dir / f"trading_lead_{tag}_exit_trades.csv"
    selected_path = args.output_dir / f"trading_lead_{tag}_selected_exits.csv"
    selected_trades_path = args.output_dir / f"trading_lead_{tag}_selected_trades.csv"
    risk_path = args.output_dir / f"trading_lead_{tag}_risk_summary.csv"
    signals_path = args.output_dir / f"trading_lead_signals_{tag}_selected.parquet"
    markdown_path = args.output_dir / f"trading_lead_{tag}.md"
    meta_path = args.output_dir / f"trading_lead_{tag}_meta.json"

    exit_summary.to_csv(exit_summary_path, index=False)
    exit_trades.to_csv(exit_trades_path, index=False)
    selected.to_csv(selected_path, index=False)
    selected_trades.to_csv(selected_trades_path, index=False)
    risk_summary.to_csv(risk_path, index=False)
    export_combined_signals(selected_trades, signals_path)
    markdown_path.write_text(markdown_report(exit_summary, selected, selected_trades, risk_summary), encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "features": str(args.features),
                "signals": str(args.signals),
                "exit_summary": str(exit_summary_path),
                "exit_trades": str(exit_trades_path),
                "selected_exits": str(selected_path),
                "selected_trades": str(selected_trades_path),
                "risk_summary": str(risk_path),
                "signals_out": str(signals_path),
                "markdown": str(markdown_path),
                "selected_rows": int(len(selected)),
                "selected_trades_count": int(len(selected_trades)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "exit_summary": str(exit_summary_path),
                "selected_exits": str(selected_path),
                "selected_trades": str(selected_trades_path),
                "risk_summary": str(risk_path),
                "signals": str(signals_path),
                "markdown": str(markdown_path),
                "selected_rows": int(len(selected)),
                "selected_trades": int(len(selected_trades)),
            },
            indent=2,
        )
    )
    return 0


def load_features(path: Path) -> DataFrame:
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    return frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)


def exit_grid() -> tuple[ExitSpec, ...]:
    return (
        ExitSpec("snap_4h_18_18", 4, 0.018, 0.018),
        ExitSpec("quick_6h_14_25", 6, 0.014, 0.025),
        ExitSpec("quick_8h_18_35", 8, 0.018, 0.035),
        ExitSpec("fast_10h_20_50", 10, 0.020, 0.050),
        ExitSpec("balanced_14h_22_45", 14, 0.022, 0.045),
        ExitSpec("balanced_18h_25_50", 18, 0.025, 0.050),
        ExitSpec("wide_24h_25_55", 24, 0.025, 0.055),
        ExitSpec("wide_30h_30_65", 30, 0.030, 0.065),
    )


def run_exit_grid(features: DataFrame, signals: DataFrame, fee: float, slippage: float) -> tuple[DataFrame, DataFrame, DataFrame]:
    rows: list[dict[str, object]] = []
    trade_frames: list[DataFrame] = []
    for rule_id in signals["rule_id"].dropna().astype(str).unique():
        rule_signals = signals[signals["rule_id"].astype(str).eq(rule_id)].copy()
        for exit_spec in exit_grid():
            trades = simulate_rule(features, rule_signals, exit_spec, fee, slippage)
            if not trades.empty:
                trade_frames.append(trades)
            rows.append(
                summarize_trades(
                    trades,
                    f"{rule_id}__{exit_spec.exit_id}",
                    str(rule_signals["concept_id"].iloc[0]) if "concept_id" in rule_signals and len(rule_signals) else rule_id,
                    "short",
                    "Which exit best fits this support-removal short entry?",
                    "Entry fixed; only stop, target, and max hold vary.",
                    features,
                )
                | {
                    "rule_id": rule_id,
                    "exit_id": exit_spec.exit_id,
                    "hold_hours": exit_spec.hold_hours,
                    "stop_loss": exit_spec.stop_loss,
                    "take_profit": exit_spec.take_profit,
                }
            )
    summary = pd.DataFrame(rows)
    all_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    summary["usable"] = summary["trades"].ge(8) & summary["total_return"].gt(0.0) & summary["profit_factor"].ge(1.20)
    summary["rank_score"] = (
        summary["total_return"].fillna(0.0)
        + summary["profit_factor"].replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.0, 5.0) / 10.0
        + summary["win_rate"].fillna(0.0) / 4.0
        + summary["max_drawdown"].fillna(-1.0)
    )
    selected = (
        summary[summary["usable"]]
        .sort_values(["rank_score", "total_return", "profit_factor"], ascending=[False, False, False])
        .groupby("rule_id", as_index=False, group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )
    return summary.sort_values(["usable", "rank_score"], ascending=[False, False]), all_trades, selected


def simulate_rule(features: DataFrame, signals: DataFrame, exit_spec: ExitSpec, fee: float, slippage: float) -> DataFrame:
    by_date = {date: idx for idx, date in enumerate(features["date"])}
    rows: list[dict[str, object]] = []
    open_ = num(features, "open")
    high = num(features, "high")
    low = num(features, "low")
    close = num(features, "close")
    flat_after = -1
    for _, signal in signals.iterrows():
        idx = by_date.get(signal["date"])
        if idx is None or idx <= flat_after or idx > len(features) - exit_spec.hold_hours - 2:
            continue
        entry_idx = idx + 1
        exit_idx = idx + exit_spec.hold_hours
        entry_price = float(open_.iloc[entry_idx])
        if not np.isfinite(entry_price) or entry_price <= 0.0:
            continue
        exit_price, exit_reason = resolve_exit("short", entry_price, high.iloc[entry_idx : exit_idx + 1], low.iloc[entry_idx : exit_idx + 1], close.iloc[exit_idx], exit_spec.stop_loss, exit_spec.take_profit)
        gross = entry_price / exit_price - 1.0
        rows.append(
            {
                "variant_id": f"{signal['rule_id']}__{exit_spec.exit_id}",
                "rule_id": str(signal["rule_id"]),
                "concept_id": str(signal.get("concept_id", "")),
                "direction": "short",
                "threshold": signal.get("threshold", np.nan),
                "signal_date": features["date"].iloc[idx],
                "entry_date": features["date"].iloc[entry_idx],
                "exit_date": features["date"].iloc[exit_idx],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross,
                "net_return": gross - fee - slippage,
                "exit_reason": exit_reason,
                "signal_score": signal.get("signal_score", np.nan),
                "exit_id": exit_spec.exit_id,
                "hold_hours": exit_spec.hold_hours,
                "stop_loss": exit_spec.stop_loss,
                "take_profit": exit_spec.take_profit,
            }
        )
        flat_after = exit_idx
    return pd.DataFrame(rows)


def simulate_selected(features: DataFrame, signals: DataFrame, selected: DataFrame, fee: float, slippage: float) -> DataFrame:
    frames: list[DataFrame] = []
    for _, row in selected.iterrows():
        exit_spec = ExitSpec(str(row["exit_id"]), int(row["hold_hours"]), float(row["stop_loss"]), float(row["take_profit"]))
        rule_signals = signals[signals["rule_id"].astype(str).eq(str(row["rule_id"]))].copy()
        frames.append(simulate_rule(features, rule_signals, exit_spec, fee, slippage))
    if not frames:
        return DataFrame()
    all_trades = pd.concat(frames, ignore_index=True).sort_values("signal_date").reset_index(drop=True)
    rows: list[dict[str, object]] = []
    flat_until = pd.Timestamp.min.tz_localize("UTC")
    for _, row in all_trades.iterrows():
        entry_date = pd.to_datetime(row["entry_date"], utc=True)
        if entry_date <= flat_until:
            continue
        rows.append(row.to_dict())
        flat_until = pd.to_datetime(row["exit_date"], utc=True)
    return pd.DataFrame(rows)


def run_risk_overlays(trades: DataFrame, features: DataFrame) -> DataFrame:
    if trades.empty:
        return DataFrame()
    merged = trades.copy()
    merged["signal_date"] = pd.to_datetime(merged["signal_date"], utc=True, errors="coerce")
    slim = features[[c for c in features.columns if c == "date" or c.startswith(("conf_", "px_", "st_", "ob_"))]].copy()
    merged = merged.merge(slim, left_on="signal_date", right_on="date", how="left")
    overlays = {
        "base_1x": pd.Series(1.0, index=merged.index),
        "score_confidence_0p6_1p4": 0.6 + pd.to_numeric(merged["signal_score"], errors="coerce").fillna(0.5).clip(0.0, 1.0) * 0.8,
        "support_cleared_size": scale_from(merged, "conf_ob_support_cleared", 0.70, 1.35, 0.75),
        "downside_vacuum_size": scale_from(merged, "conf_ob_downside_vacuum_after_support_removed", 0.35, 1.30, 0.80),
        "bounce_warning_reduce": 1.15 - max_existing(merged, ["conf_ob_support_bounce", "conf_ob_breakdown_failure", "conf_ob_bid_absorption"]).clip(0.0, 1.0) * 0.70,
        "bearish_pressure_size": scale_from(merged, "conf_ob_bearish_pressure_agreement", 0.45, 1.25, 0.85),
    }
    rows = []
    base = summarize_scaled(merged["net_return"], overlays["base_1x"], "base_1x", None)
    rows.append(base)
    for name, mult in overlays.items():
        if name == "base_1x":
            continue
        rows.append(summarize_scaled(merged["net_return"], mult.clip(0.25, 1.50), name, base))
    return pd.DataFrame(rows).sort_values(["verdict", "risk_adjusted_delta"], ascending=[True, False]).reset_index(drop=True)


def scale_from(frame: DataFrame, col: str, threshold: float, high: float, low: float) -> Series:
    value = pd.to_numeric(frame[col], errors="coerce").fillna(0.0) if col in frame else pd.Series(0.0, index=frame.index)
    return pd.Series(np.where(value.ge(threshold), high, low), index=frame.index)


def max_existing(frame: DataFrame, cols: list[str]) -> Series:
    parts = [pd.to_numeric(frame[c], errors="coerce") for c in cols if c in frame]
    if not parts:
        return pd.Series(0.0, index=frame.index)
    return pd.concat(parts, axis=1).max(axis=1).fillna(0.0)


def summarize_scaled(returns: Series, multiplier: Series, overlay_id: str, base: dict[str, object] | None) -> dict[str, object]:
    scaled = pd.to_numeric(returns, errors="coerce").fillna(0.0) * multiplier.fillna(1.0)
    equity = (1.0 + scaled).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    wins = scaled[scaled.gt(0.0)]
    losses = scaled[scaled.lt(0.0)]
    pf = float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 0 else np.inf
    row = {
        "overlay_id": overlay_id,
        "trades": int(len(scaled)),
        "avg_multiplier": float(multiplier.mean()),
        "win_rate": float(scaled.gt(0.0).mean()),
        "total_return": float(equity.iloc[-1] - 1.0),
        "max_drawdown": float(drawdown.min()),
        "profit_factor": pf,
    }
    if base is None:
        row.update({"return_delta": 0.0, "drawdown_delta": 0.0, "risk_adjusted_delta": 0.0, "verdict": "baseline"})
        return row
    return_delta = row["total_return"] - float(base["total_return"])
    drawdown_delta = row["max_drawdown"] - float(base["max_drawdown"])
    risk_adjusted_delta = return_delta + drawdown_delta + (row["profit_factor"] - float(base["profit_factor"])) / 5.0
    row.update(
        {
            "return_delta": return_delta,
            "drawdown_delta": drawdown_delta,
            "risk_adjusted_delta": risk_adjusted_delta,
            "verdict": "watchlist" if risk_adjusted_delta > 0.02 and row["total_return"] >= float(base["total_return"]) else "reject_for_now",
        }
    )
    return row


def markdown_report(exit_summary: DataFrame, selected: DataFrame, selected_trades: DataFrame, risk_summary: DataFrame) -> str:
    combined = summarize_trades(
        selected_trades,
        "support_removal_tailored_exit_selected",
        "support_removal_tailored_exit",
        "short",
        "Selected tailored exits for support-removal shorts.",
        "Entry fixed; exits selected per rule.",
        selected_trades.rename(columns={"signal_date": "date"}) if not selected_trades.empty else DataFrame({"date": []}),
    )
    lines = [
        "# Support-Removal Short Exit/Risk Research",
        "",
        "This keeps the orderbook support-removal short entries fixed and tests exits plus simple risk multipliers.",
        "",
        "## Selected Exit Block",
        "",
        f"1. Trades: {combined.get('trades', 0)}",
        f"2. Win rate: {fmt_pct(combined.get('win_rate'))}",
        f"3. Total return: {fmt_pct(combined.get('total_return'))}",
        f"4. Max drawdown: {fmt_pct(combined.get('max_drawdown'))}",
        f"5. Profit factor: {fmt(combined.get('profit_factor'))}",
        "",
        "## Selected Exits",
        "",
    ]
    if selected.empty:
        lines.append("No exit variant passed.")
    else:
        for i, row in selected.iterrows():
            lines.append(f"{i + 1}. `{row['rule_id']}` -> `{row['exit_id']}`")
            lines.append(f"   - Trades: {int(row['trades'])}; win rate: {fmt_pct(row['win_rate'])}; return: {fmt_pct(row['total_return'])}; drawdown: {fmt_pct(row['max_drawdown'])}; PF: {fmt(row['profit_factor'])}.")
    lines.extend(["", "## Risk Overlay Screen", ""])
    if risk_summary.empty:
        lines.append("No risk overlays evaluated.")
    else:
        lines.append(risk_summary[["overlay_id", "trades", "avg_multiplier", "total_return", "max_drawdown", "profit_factor", "risk_adjusted_delta", "verdict"]].to_markdown(index=False))
    lines.extend(["", "## Top Exit Variants", ""])
    for i, row in exit_summary.head(20).reset_index(drop=True).iterrows():
        lines.append(f"{i + 1}. `{row['variant_id']}`")
        lines.append(f"   - Result: {int(row['trades'])} trades, {fmt_pct(row['win_rate'])} wins, {fmt_pct(row['total_return'])} return, {fmt_pct(row['max_drawdown'])} drawdown, PF {fmt(row['profit_factor'])}.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
