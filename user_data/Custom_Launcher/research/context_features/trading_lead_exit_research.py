from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


USER_DATA_DIR = Path(__file__).resolve().parents[3]
REPORTS_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "reports"
DEFAULT_FEATURES = USER_DATA_DIR / "research_news_data" / "context_features" / "confluence_cache" / "trader_confluence_1h_latest.parquet"
DEFAULT_SIGNALS = REPORTS_DIR / "trader_rule_signals_sieve_exact_min10_20260605.parquet"


@dataclass(frozen=True)
class ExitSpec:
    exit_id: str
    hold_hours: int
    stop_loss: float
    take_profit: float
    exit_story: str


def main() -> int:
    parser = argparse.ArgumentParser(description="Sweep tailored exits for selected trader-readable BTC entry leads.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--signals", type=Path, default=DEFAULT_SIGNALS)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--tag", default=datetime.now(timezone.utc).strftime("%Y%m%d"))
    parser.add_argument("--round-trip-fee", type=float, default=0.0010)
    parser.add_argument("--slippage", type=float, default=0.0002)
    parser.add_argument("--min-trades", type=int, default=8)
    args = parser.parse_args()

    if not args.features.exists():
        raise FileNotFoundError(args.features)
    if not args.signals.exists():
        raise FileNotFoundError(args.signals)

    frame = load_frame(args.features)
    signals = load_signals(args.signals)
    summary, trades, selected = run_exit_sweep(
        frame,
        signals,
        round_trip_fee=float(args.round_trip_fee),
        slippage=float(args.slippage),
        min_trades=int(args.min_trades),
    )
    combined = simulate_combined_with_selected_exits(
        frame,
        signals,
        selected,
        round_trip_fee=float(args.round_trip_fee),
        slippage=float(args.slippage),
    )
    combined_summary = summarize_trades(
        combined,
        "combined_selected_entries_tailored_exits",
        "combined",
        "both",
        "Use the same selected entry signals, but apply the best tailored exit found for each rule family.",
        "Entry logic is unchanged; only stop, target, and hold time differ by rule family.",
        frame,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = safe_name(args.tag)
    summary_path = args.output_dir / f"trading_lead_exit_research_{tag}_summary.csv"
    trades_path = args.output_dir / f"trading_lead_exit_research_{tag}_trades.csv"
    selected_path = args.output_dir / f"trading_lead_exit_research_{tag}_selected.csv"
    combined_path = args.output_dir / f"trading_lead_exit_research_{tag}_combined_trades.csv"
    markdown_path = args.output_dir / f"trading_lead_exit_research_{tag}.md"
    meta_path = args.output_dir / f"trading_lead_exit_research_{tag}_meta.json"
    summary.to_csv(summary_path, index=False)
    trades.to_csv(trades_path, index=False)
    selected.to_csv(selected_path, index=False)
    combined.to_csv(combined_path, index=False)
    markdown_path.write_text(markdown_report(summary, selected, combined_summary), encoding="utf-8")
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "signals": str(args.signals),
        "features": str(args.features),
        "entry_signals": int(len(signals)),
        "exit_variants": int(len(summary)),
        "selected_exit_rules": int(len(selected)),
        "combined_summary": combined_summary,
        "outputs": {
            "summary": str(summary_path),
            "trades": str(trades_path),
            "selected": str(selected_path),
            "combined_trades": str(combined_path),
            "markdown": str(markdown_path),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


def load_frame(path: Path) -> DataFrame:
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    for col in ["open", "high", "low", "close"]:
        if col not in frame.columns:
            raise KeyError(f"Missing OHLCV column: {col}")
    return frame


def load_signals(path: Path) -> DataFrame:
    signals = pd.read_parquet(path)
    signals["date"] = pd.to_datetime(signals["date"], utc=True, errors="coerce")
    if "rule_id" in signals.columns:
        signals["rule_id"] = signals["rule_id"].astype(str)
        signals = (
            signals.dropna(subset=["date"])
            .sort_values(["date", "rule_id"])
            .drop_duplicates(["date", "rule_id"], keep="first")
            .reset_index(drop=True)
        )
    else:
        signals = signals.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="first").reset_index(drop=True)
    required = {"date", "enter_long", "enter_short", "rule_id", "concept_id", "threshold", "signal_score"}
    missing = sorted(required - set(signals.columns))
    if missing:
        raise KeyError(f"Missing signal columns: {missing}")
    signals["direction"] = np.where(pd.to_numeric(signals["enter_long"], errors="coerce").fillna(0).gt(0), "long", "short")
    return signals


def run_exit_sweep(frame: DataFrame, signals: DataFrame, *, round_trip_fee: float, slippage: float, min_trades: int) -> tuple[DataFrame, DataFrame, DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    trade_frames: list[DataFrame] = []
    for rule_id, rule_signals in signals.groupby("rule_id", dropna=False):
        for spec in exit_specs_for(str(rule_id)):
            trades = simulate_signals(frame, rule_signals, spec, round_trip_fee, slippage)
            trade_frames.append(trades)
            summary = summarize_trades(
                trades,
                f"{rule_id}__{spec.exit_id}",
                str(rule_signals["concept_id"].iloc[0]),
                str(rule_signals["direction"].iloc[0]),
                question_for_rule(str(rule_id)),
                spec.exit_story,
                frame,
            )
            summary.update({"rule_id": str(rule_id), "exit_id": spec.exit_id, "hold_hours": spec.hold_hours, "stop_loss": spec.stop_loss, "take_profit": spec.take_profit})
            summary_rows.append(summary)
    summary = pd.DataFrame(summary_rows)
    all_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame()
    summary["usable_sample_size"] = summary["trades"].ge(min_trades)
    summary["quality_score"] = (
        summary["total_return"].fillna(0.0)
        + summary["profit_factor"].replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.0, 5.0) / 8.0
        + summary["win_rate"].fillna(0.0) / 4.0
        + summary["max_drawdown"].fillna(-1.0)
        + summary["trades"].fillna(0.0).clip(0, 50) / 500.0
    )
    summary["positive_exit_quality"] = (
        summary["usable_sample_size"]
        & summary["total_return"].gt(0.0)
        & summary["max_drawdown"].gt(-0.25)
        & (summary["profit_factor"].ge(1.20) | summary["win_rate"].ge(0.52))
    )
    selected = (
        summary[summary["positive_exit_quality"]]
        .sort_values(["quality_score", "profit_factor", "total_return"], ascending=[False, False, False])
        .groupby("rule_id", as_index=False, group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )
    return summary.sort_values(["positive_exit_quality", "quality_score"], ascending=[False, False]), all_trades, selected


def exit_specs_for(rule_id: str) -> tuple[ExitSpec, ...]:
    if "mtf_4h_tlv2_res_break_long" in rule_id:
        holds = (6, 8, 12, 18, 24, 36)
        stops = (0.010, 0.014, 0.020, 0.028)
        targets = (0.012, 0.020, 0.035, 0.050, 0.075)
        story = "4h TLV2 resistance-break long exits: fast breakout pop, failed-break control, or hold toward the next resistance/value area."
    elif "mtf_1d_tlv2_res_retest_long" in rule_id:
        holds = (12, 18, 24, 36, 48, 72)
        stops = (0.012, 0.018, 0.025, 0.035)
        targets = (0.020, 0.035, 0.050, 0.075, 0.100)
        story = "Daily TLV2 resistance-retest long exits: give the retest time to continue, but cut if the reclaimed level fails."
    elif "mtf_4h_triangle_upper_break_long" in rule_id:
        holds = (6, 10, 12, 18, 24, 36)
        stops = (0.010, 0.014, 0.020, 0.028)
        targets = (0.012, 0.020, 0.035, 0.050, 0.075)
        story = "4h triangle upper-break long exits: capture squeeze release quickly, or hold while the release keeps expanding."
    elif "mtf_4h_vp_val_accept_short" in rule_id:
        holds = (6, 8, 12, 18, 24, 36)
        stops = (0.010, 0.014, 0.020, 0.028)
        targets = (0.012, 0.020, 0.035, 0.050, 0.075)
        story = "4h VAL acceptance short exits: capture downside value migration, but exit if price reclaims value."
    elif "mtf_1d_rectangle_lower_break_short" in rule_id:
        holds = (6, 10, 12, 18, 24, 36)
        stops = (0.010, 0.014, 0.020, 0.028)
        targets = (0.012, 0.020, 0.035, 0.050, 0.075)
        story = "Daily rectangle lower-break short exits: test fast breakdown continuation versus fakeout control."
    elif "mtf_1d_tlv2_sup_retest_short" in rule_id:
        holds = (12, 18, 24, 36, 48, 72)
        stops = (0.012, 0.018, 0.025, 0.035)
        targets = (0.020, 0.035, 0.050, 0.075, 0.100)
        story = "Daily TLV2 support-retest short exits: give the failed support retest time to continue, but cut if support is reclaimed."
    elif "vacuum" in rule_id or "support_removed" in rule_id:
        holds = (3, 4, 6, 8, 12)
        stops = (0.010, 0.014, 0.018, 0.025)
        targets = (0.012, 0.020, 0.025, 0.035)
        story = "Support-removal/vacuum short exits: quick follow-through, immediate reclaim invalidation, or fast target."
    elif "support_cleared" in rule_id:
        holds = (6, 12, 18, 24, 36)
        stops = (0.014, 0.020, 0.025, 0.035)
        targets = (0.020, 0.035, 0.050, 0.075)
        story = "Support-cleared short exits: allow breakdown continuation, but exit if support reclaim/fade appears."
    elif "rectangle" in rule_id:
        holds = (6, 10, 16, 24, 36, 48)
        stops = (0.012, 0.018, 0.025, 0.035)
        targets = (0.020, 0.035, 0.050, 0.075)
        story = "Rectangle/compression exits: quick fakeout control, continuation target, or next-level target."
    elif "low_break" in rule_id:
        holds = (6, 12, 24, 36, 48, 72)
        stops = (0.012, 0.018, 0.025, 0.035)
        targets = (0.012, 0.020, 0.035, 0.050, 0.075)
        story = "Prior-day low break exits: crash-continuation hold, support-reclaim invalidation, or fast downside target."
    else:
        holds = (6, 12, 24, 36, 48, 72)
        stops = (0.012, 0.018, 0.025, 0.035)
        targets = (0.012, 0.020, 0.035, 0.050)
        story = "Prior-day high break exits: fast continuation target, failed-break invalidation, or time-based fade exit."
    specs: list[ExitSpec] = []
    for hold in holds:
        for stop in stops:
            for target in targets:
                specs.append(ExitSpec(f"h{hold}_sl{int(stop*1000):03d}_tp{int(target*1000):03d}", hold, stop, target, story))
    return tuple(specs)


def simulate_combined_with_selected_exits(frame: DataFrame, signals: DataFrame, selected: DataFrame, *, round_trip_fee: float, slippage: float) -> DataFrame:
    if selected.empty:
        return DataFrame()
    selected_by_rule = {str(row["rule_id"]): ExitSpec(str(row["exit_id"]), int(row["hold_hours"]), float(row["stop_loss"]), float(row["take_profit"]), str(row["visible_market_state"])) for _, row in selected.iterrows()}
    rows: list[dict[str, Any]] = []
    signals = signals.sort_values("date").reset_index(drop=True)
    flat_until = pd.Timestamp.min.tz_localize("UTC")
    for _, signal in signals.iterrows():
        signal_date = pd.to_datetime(signal["date"], utc=True)
        if signal_date <= flat_until:
            continue
        spec = selected_by_rule.get(str(signal["rule_id"]))
        if spec is None:
            continue
        trades = simulate_signals(frame, pd.DataFrame([signal]), spec, round_trip_fee, slippage)
        if trades.empty:
            continue
        trade = trades.iloc[0].to_dict()
        rows.append(trade)
        flat_until = pd.to_datetime(trade["exit_date"], utc=True)
    return pd.DataFrame(rows)


def simulate_signals(frame: DataFrame, signals: DataFrame, spec: ExitSpec, round_trip_fee: float, slippage: float) -> DataFrame:
    frame_index = {date: idx for idx, date in enumerate(frame["date"])}
    rows: list[dict[str, Any]] = []
    open_ = num(frame, "open")
    high = num(frame, "high")
    low = num(frame, "low")
    close = num(frame, "close")
    for _, signal in signals.iterrows():
        signal_date = pd.to_datetime(signal["date"], utc=True)
        i = frame_index.get(signal_date)
        if i is None:
            continue
        if i > len(frame) - spec.hold_hours - 2:
            continue
        entry_idx = i + 1
        exit_idx = i + spec.hold_hours
        entry_price = float(open_.iloc[entry_idx])
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue
        direction = str(signal["direction"])
        exit_price, exit_reason = resolve_exit(direction, entry_price, high.iloc[entry_idx : exit_idx + 1], low.iloc[entry_idx : exit_idx + 1], close.iloc[exit_idx], spec.stop_loss, spec.take_profit)
        if not np.isfinite(exit_price) or exit_price <= 0:
            continue
        gross_return = exit_price / entry_price - 1.0 if direction == "long" else entry_price / exit_price - 1.0
        rows.append(
            {
                "variant_id": f"{signal['rule_id']}__{spec.exit_id}",
                "rule_id": str(signal["rule_id"]),
                "concept_id": str(signal["concept_id"]),
                "direction": direction,
                "exit_id": spec.exit_id,
                "hold_hours": spec.hold_hours,
                "stop_loss": spec.stop_loss,
                "take_profit": spec.take_profit,
                "signal_date": signal_date,
                "entry_date": frame["date"].iloc[entry_idx],
                "exit_date": frame["date"].iloc[exit_idx],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross_return,
                "net_return": gross_return - round_trip_fee - slippage,
                "exit_reason": exit_reason,
                "signal_score": float(signal["signal_score"]) if np.isfinite(float(signal["signal_score"])) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def resolve_exit(direction: str, entry_price: float, highs: Series, lows: Series, fallback_close: float, stop_loss: float, take_profit: float) -> tuple[float, str]:
    if direction == "long":
        stop_price = entry_price * (1.0 - stop_loss)
        take_price = entry_price * (1.0 + take_profit)
        for high, low in zip(highs, lows):
            if np.isfinite(low) and low <= stop_price:
                return stop_price, "stop_loss"
            if np.isfinite(high) and high >= take_price:
                return take_price, "take_profit"
    else:
        stop_price = entry_price * (1.0 + stop_loss)
        take_price = entry_price * (1.0 - take_profit)
        for high, low in zip(highs, lows):
            if np.isfinite(high) and high >= stop_price:
                return stop_price, "stop_loss"
            if np.isfinite(low) and low <= take_price:
                return take_price, "take_profit"
    return float(fallback_close), "time_exit"


def summarize_trades(trades: DataFrame, variant_id: str, concept_id: str, direction: str, trader_question: str, visible_market_state: str, frame: DataFrame) -> dict[str, Any]:
    start = pd.to_datetime(frame["date"].min(), utc=True)
    end = pd.to_datetime(frame["date"].max(), utc=True)
    years = max((end - start).total_seconds() / (365.25 * 24 * 3600), 1e-9)
    buy_hold_return = float(num(frame, "close").iloc[-1] / num(frame, "close").iloc[0] - 1.0)
    if trades.empty:
        return empty_summary(variant_id, concept_id, direction, trader_question, visible_market_state, buy_hold_return)
    returns = pd.to_numeric(trades["net_return"], errors="coerce").dropna()
    if returns.empty:
        return empty_summary(variant_id, concept_id, direction, trader_question, visible_market_state, buy_hold_return)
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    wins = returns[returns.gt(0.0)]
    losses = returns[returns.lt(0.0)]
    profit_factor = float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 0 else np.inf
    exit_counts = trades["exit_reason"].value_counts().to_dict()
    return {
        "variant_id": variant_id,
        "concept_id": concept_id,
        "direction": direction,
        "trader_question": trader_question,
        "visible_market_state": visible_market_state,
        "trades": int(len(returns)),
        "trades_per_year": float(len(returns) / years),
        "win_rate": float(returns.gt(0.0).mean()),
        "avg_trade_return": float(returns.mean()),
        "median_trade_return": float(returns.median()),
        "total_return": float(equity.iloc[-1] - 1.0),
        "max_drawdown": float(drawdown.min()),
        "profit_factor": profit_factor,
        "buy_hold_return": buy_hold_return,
        "take_profit_exits": int(exit_counts.get("take_profit", 0)),
        "stop_loss_exits": int(exit_counts.get("stop_loss", 0)),
        "time_exits": int(exit_counts.get("time_exit", 0)),
    }


def empty_summary(variant_id: str, concept_id: str, direction: str, trader_question: str, visible_market_state: str, buy_hold_return: float) -> dict[str, Any]:
    return {
        "variant_id": variant_id,
        "concept_id": concept_id,
        "direction": direction,
        "trader_question": trader_question,
        "visible_market_state": visible_market_state,
        "trades": 0,
        "trades_per_year": 0.0,
        "win_rate": np.nan,
        "avg_trade_return": np.nan,
        "median_trade_return": np.nan,
        "total_return": 0.0,
        "max_drawdown": 0.0,
        "profit_factor": np.nan,
        "buy_hold_return": buy_hold_return,
        "take_profit_exits": 0,
        "stop_loss_exits": 0,
        "time_exits": 0,
    }


def question_for_rule(rule_id: str) -> str:
    mapping = {
        "mtf_4h_tlv2_res_break_long": "For 4h TLV2 resistance-break longs, which exit captures continuation without letting failed breaks linger?",
        "mtf_1d_tlv2_res_retest_long": "For daily TLV2 resistance-retest longs, which exit gives the retest enough time while controlling failed retests?",
        "mtf_4h_triangle_upper_break_long": "For 4h triangle upper-break longs, which exit captures squeeze release without overholding failed releases?",
        "mtf_4h_vp_val_accept_short": "For 4h VAL acceptance shorts, which exit captures downside value migration without holding through value reclaim?",
        "mtf_1d_rectangle_lower_break_short": "For daily rectangle lower-break shorts, which exit captures breakdown continuation without holding fakeouts?",
        "mtf_1d_tlv2_sup_retest_short": "For daily TLV2 support-retest shorts, which exit captures continuation after failed support retest?",
        "sieve_pattern_rectangle_breakdown_short": "For rectangle/compression breakdown shorts, which exit best captures follow-through without giving back too much?",
        "sieve_pattern_rectangle_breakout_long": "For rectangle/compression breakout longs, which exit best captures continuation without letting fakeouts linger?",
        "sieve_prior_day_high_break_vp_long": "For prior-day high break longs with VP/pressure support, which exit best captures upside continuation?",
        "sieve_prior_day_low_break_vp_short": "For prior-day low break shorts with VP/pressure support, which exit best captures downside continuation or crash momentum?",
        "support_break_support_cleared_short": "For support-cleared shorts, which exit captures breakdown continuation without holding through reclaim?",
        "val_break_downside_vacuum_after_support_removed_short": "For quick VAL/vacuum shorts, which exit captures fast downside follow-through before support can reclaim?",
    }
    return mapping.get(rule_id, f"For {rule_id}, which stop/target/hold combination works best?")


def markdown_report(summary: DataFrame, selected: DataFrame, combined_summary: dict[str, Any]) -> str:
    lines = [
        "# Trading Lead Exit Research",
        "",
        "This keeps the selected entry signals fixed and tests only stop, target, and hold-time choices.",
        "",
        "## Combined Tailored-Exit Block",
        "",
        f"1. Trades: {combined_summary.get('trades', 0)}",
        f"2. Win rate: {fmt_pct(combined_summary.get('win_rate'))}",
        f"3. Total return: {fmt_pct(combined_summary.get('total_return'))}",
        f"4. Average trade return: {fmt_pct(combined_summary.get('avg_trade_return'))}",
        f"5. Max drawdown: {fmt_pct(combined_summary.get('max_drawdown'))}",
        f"6. Profit factor: {fmt(combined_summary.get('profit_factor'))}",
        "",
        "## Selected Exits By Entry Family",
        "",
    ]
    if selected.empty:
        lines.append("No tailored exits passed the minimum quality gate.")
    else:
        for _, row in selected.sort_values("rule_id").iterrows():
            lines.append(f"1. `{row['rule_id']}`")
            lines.append(f"   - Trader question: {row['trader_question']}")
            lines.append(f"   - Exit: hold `{int(row['hold_hours'])}h`, stop `{fmt_pct(row['stop_loss'])}`, target `{fmt_pct(row['take_profit'])}`")
            lines.append(f"   - Result: `{int(row['trades'])}` trades, `{fmt_pct(row['win_rate'])}` win rate, `{fmt_pct(row['total_return'])}` return, `{fmt_pct(row['max_drawdown'])}` drawdown, profit factor `{fmt(row['profit_factor'])}`")
    lines.extend(["", "## Top Exit Variants", ""])
    top = summary.sort_values(["positive_exit_quality", "quality_score"], ascending=[False, False]).head(30)
    for _, row in top.iterrows():
        lines.append(f"1. `{row['variant_id']}`")
        lines.append(f"   - Exit: hold `{int(row['hold_hours'])}h`, stop `{fmt_pct(row['stop_loss'])}`, target `{fmt_pct(row['take_profit'])}`")
        lines.append(f"   - Result: `{int(row['trades'])}` trades, `{fmt_pct(row['win_rate'])}` win rate, `{fmt_pct(row['total_return'])}` return, `{fmt_pct(row['max_drawdown'])}` drawdown, profit factor `{fmt(row['profit_factor'])}`")
    return "\n".join(lines) + "\n"


def num(frame: DataFrame, column: str) -> Series:
    return pd.to_numeric(frame[column], errors="coerce")


def fmt(value: Any) -> str:
    try:
        if value is None or not np.isfinite(float(value)):
            return "n/a"
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def fmt_pct(value: Any) -> str:
    try:
        if value is None or not np.isfinite(float(value)):
            return "n/a"
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "n/a"


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value)).strip("_")[:120] or "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
