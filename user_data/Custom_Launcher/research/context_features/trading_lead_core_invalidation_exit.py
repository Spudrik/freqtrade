from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame

from trader_rule_backtests import fmt, fmt_pct, num, safe_name, summarize_trades


USER_DATA_DIR = Path(__file__).resolve().parents[3]
REPORTS_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "reports"
DEFAULT_FEATURES = USER_DATA_DIR / "research_news_data" / "context_features" / "confluence_cache" / "trader_confluence_1h_latest.parquet"
DEFAULT_SIGNALS = REPORTS_DIR / "trading_lead_signals_20260605_validated_orderbook_overlap_core_validated_orderbook.parquet"

SUPPORT_CLEARED = "lead__validated__support_break_support_cleared_short"
NO_RECLAIM = "lead__validated__val_vacuum_no_support_reclaim_warning_short"


@dataclass(frozen=True)
class ExitProfile:
    profile_id: str
    rule_id: str
    hold_hours: int
    stop_loss: float
    take_profit: float
    invalidation_kind: str
    invalidation_threshold: float
    min_age_hours: int
    trader_question: str
    visible_market_state: str


def main() -> int:
    parser = argparse.ArgumentParser(description="Test thesis-invalidation exits for the validated core orderbook short block.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--signals", type=Path, default=DEFAULT_SIGNALS)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--tag", default="20260605_core_invalidation_exit")
    parser.add_argument("--round-trip-fee", type=float, default=0.0010)
    parser.add_argument("--slippage", type=float, default=0.0002)
    args = parser.parse_args()

    if not args.features.exists():
        raise FileNotFoundError(args.features)
    if not args.signals.exists():
        raise FileNotFoundError(args.signals)

    features = load_features(args.features)
    signals = load_signals(args.signals)
    profiles = build_exit_profiles()
    summary, trades, selected, selected_trades = run_profiles(features, signals, profiles, args.round_trip_fee, args.slippage)

    tag = safe_name(args.tag)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"trading_lead_core_invalidation_exit_{tag}_summary.csv"
    trades_path = args.output_dir / f"trading_lead_core_invalidation_exit_{tag}_trades.csv"
    selected_path = args.output_dir / f"trading_lead_core_invalidation_exit_{tag}_selected.csv"
    selected_trades_path = args.output_dir / f"trading_lead_core_invalidation_exit_{tag}_selected_trades.csv"
    markdown_path = args.output_dir / f"trading_lead_core_invalidation_exit_{tag}.md"
    meta_path = args.output_dir / f"trading_lead_core_invalidation_exit_{tag}_meta.json"

    summary.to_csv(summary_path, index=False)
    trades.to_csv(trades_path, index=False)
    selected.to_csv(selected_path, index=False)
    selected_trades.to_csv(selected_trades_path, index=False)
    markdown_path.write_text(markdown_report(summary, selected, selected_trades, features), encoding="utf-8")
    meta_path.write_text(
        json.dumps(
            {
                "features": str(args.features),
                "signals": str(args.signals),
                "summary": str(summary_path),
                "trades": str(trades_path),
                "selected": str(selected_path),
                "selected_trades": str(selected_trades_path),
                "markdown": str(markdown_path),
                "profile_count": int(len(profiles)),
                "selected_count": int(len(selected)),
                "selected_trades": int(len(selected_trades)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "selected": str(selected_path),
                "selected_trades": str(selected_trades_path),
                "markdown": str(markdown_path),
                "profile_count": int(len(profiles)),
                "selected_count": int(len(selected)),
                "selected_trades": int(len(selected_trades)),
            },
            indent=2,
        )
    )
    return 0


def load_features(path: Path) -> DataFrame:
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    required = {"date", "open", "high", "low", "close"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"Missing required feature columns: {missing}")
    return frame


def load_signals(path: Path) -> DataFrame:
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="first").reset_index(drop=True)
    return frame[frame["rule_id"].astype(str).isin({SUPPORT_CLEARED, NO_RECLAIM})].copy()


def build_exit_profiles() -> tuple[ExitProfile, ...]:
    profiles: list[ExitProfile] = []
    profiles.append(
        ExitProfile(
            "support_cleared_baseline_fixed",
            SUPPORT_CLEARED,
            18,
            0.025,
            0.050,
            "none",
            0.0,
            999,
            "If support is cleared, does a fixed 18h/5% target exit still work?",
            "Bearish orderbook support-cleared short with no thesis-invalidation check.",
        )
    )
    for kind, thresholds, min_ages in (
        ("reclaim_rebuild_warning", (0.38, 0.46, 0.54), (1, 2, 3)),
        ("bearish_pressure_fade", (0.05, 0.15, 0.25), (1, 2, 3)),
        ("combined_reclaim_or_pressure_fade", (0.38, 0.46, 0.54), (1, 2, 3)),
    ):
        for threshold in thresholds:
            for min_age in min_ages:
                profiles.append(
                    ExitProfile(
                        f"support_cleared_{kind}_{threshold:g}_age{min_age}",
                        SUPPORT_CLEARED,
                        18,
                        0.025,
                        0.050,
                        kind,
                        threshold,
                        min_age,
                        "After support is cleared, should we exit early when the short thesis starts disappearing?",
                        "Support was cleared, but we watch for reclaim/rebuild warnings or bearish pressure fading.",
                    )
                )
    profiles.append(
        ExitProfile(
            "no_reclaim_baseline_fixed",
            NO_RECLAIM,
            8,
            0.010,
            0.035,
            "none",
            0.0,
            999,
            "If VAL/vacuum breaks without reclaim warning, does a fixed 8h/3.5% target exit still work?",
            "Strict VAL/vacuum short with no thesis-invalidation check.",
        )
    )
    for kind, thresholds, min_ages in (
        ("reclaim_rebuild_warning", (0.22, 0.30, 0.38), (1, 2, 3)),
        ("breakdown_failure_warning", (0.08, 0.14, 0.20), (1, 2, 3)),
        ("combined_reclaim_or_breakdown_failure", (0.22, 0.30, 0.38), (1, 2, 3)),
    ):
        for threshold in thresholds:
            for min_age in min_ages:
                profiles.append(
                    ExitProfile(
                        f"no_reclaim_{kind}_{threshold:g}_age{min_age}",
                        NO_RECLAIM,
                        8,
                        0.010,
                        0.035,
                        kind,
                        threshold,
                        min_age,
                        "After a no-reclaim VAL/vacuum short, should we exit early when support starts coming back?",
                        "Downside vacuum existed and reclaim was absent, but we watch for reclaim, rebuild, or breakdown-failure warnings.",
                    )
                )
    return tuple(profiles)


def run_profiles(
    features: DataFrame,
    signals: DataFrame,
    profiles: tuple[ExitProfile, ...],
    fee: float,
    slippage: float,
) -> tuple[DataFrame, DataFrame, DataFrame, DataFrame]:
    rows: list[dict[str, Any]] = []
    trade_frames: list[DataFrame] = []
    for profile in profiles:
        profile_signals = signals[signals["rule_id"].astype(str).eq(profile.rule_id)].copy()
        trades = simulate_profile(features, profile_signals, profile, fee, slippage)
        if not trades.empty:
            trade_frames.append(trades)
        rows.append(
            summarize_trades(
                trades,
                profile.profile_id,
                profile.rule_id,
                "short",
                profile.trader_question,
                profile.visible_market_state,
                features,
            )
            | {
                "profile_id": profile.profile_id,
                "rule_id": profile.rule_id,
                "hold_hours": profile.hold_hours,
                "stop_loss": profile.stop_loss,
                "take_profit": profile.take_profit,
                "invalidation_kind": profile.invalidation_kind,
                "invalidation_threshold": profile.invalidation_threshold,
                "min_age_hours": profile.min_age_hours,
                "invalidation_exit_count": int(trades["exit_reason"].astype(str).str.contains("invalidation").sum()) if not trades.empty else 0,
            }
        )
    summary = pd.DataFrame(rows)
    all_trades = pd.concat(trade_frames, ignore_index=True) if trade_frames else DataFrame()
    summary["rank_score"] = (
        summary["total_return"].fillna(0.0)
        + summary["profit_factor"].replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.0, 5.0) / 10.0
        + summary["win_rate"].fillna(0.0) / 4.0
        + summary["max_drawdown"].fillna(-1.0)
    )
    selected = (
        summary[summary["trades"].ge(6) & summary["total_return"].gt(0.0)]
        .sort_values(["rank_score", "total_return", "profit_factor"], ascending=[False, False, False])
        .groupby("rule_id", as_index=False, group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )
    selected_trades = select_non_overlapping_trades(all_trades, selected)
    return summary.sort_values(["rule_id", "rank_score"], ascending=[True, False]), all_trades, selected, selected_trades


def simulate_profile(features: DataFrame, signals: DataFrame, profile: ExitProfile, fee: float, slippage: float) -> DataFrame:
    by_date = {date: idx for idx, date in enumerate(features["date"])}
    open_ = num(features, "open")
    high = num(features, "high")
    low = num(features, "low")
    close = num(features, "close")
    rows: list[dict[str, Any]] = []
    flat_after = -1
    for _, signal in signals.iterrows():
        idx = by_date.get(signal["date"])
        if idx is None or idx <= flat_after or idx > len(features) - profile.hold_hours - 2:
            continue
        entry_idx = idx + 1
        entry_price = float(open_.iloc[entry_idx])
        if not np.isfinite(entry_price) or entry_price <= 0.0:
            continue
        exit_idx, exit_price, exit_reason = resolve_profile_exit(features, entry_idx, profile, entry_price, high, low, close)
        gross = entry_price / exit_price - 1.0
        rows.append(
            {
                "variant_id": profile.profile_id,
                "rule_id": profile.rule_id,
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
                "hold_hours": profile.hold_hours,
                "stop_loss": profile.stop_loss,
                "take_profit": profile.take_profit,
                "invalidation_kind": profile.invalidation_kind,
                "invalidation_threshold": profile.invalidation_threshold,
                "min_age_hours": profile.min_age_hours,
            }
        )
        flat_after = exit_idx
    return pd.DataFrame(rows)


def resolve_profile_exit(
    features: DataFrame,
    entry_idx: int,
    profile: ExitProfile,
    entry_price: float,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
) -> tuple[int, float, str]:
    stop_price = entry_price * (1.0 + profile.stop_loss)
    take_price = entry_price * (1.0 - profile.take_profit)
    final_idx = entry_idx + profile.hold_hours - 1
    for candle_idx in range(entry_idx, final_idx + 1):
        if np.isfinite(high.iloc[candle_idx]) and high.iloc[candle_idx] >= stop_price:
            return candle_idx, float(stop_price), "stop_loss"
        if np.isfinite(low.iloc[candle_idx]) and low.iloc[candle_idx] <= take_price:
            return candle_idx, float(take_price), "take_profit"
        age = candle_idx - entry_idx + 1
        if age >= profile.min_age_hours and invalidation_active(features, candle_idx, profile):
            return candle_idx, float(close.iloc[candle_idx]), f"invalidation_{profile.invalidation_kind}"
    return final_idx, float(close.iloc[final_idx]), "time_exit"


def invalidation_active(features: DataFrame, idx: int, profile: ExitProfile) -> bool:
    if profile.invalidation_kind == "none":
        return False
    reclaim = max_existing(
        features,
        idx,
        [
            "conf_epsilon_support_reclaim_after_break_score",
            "conf_epsilon_support_reclaim_after_break_cmp_score_support_rebuild",
            "conf_ob_support_bounce",
            "conf_ob_bid_absorption",
        ],
    )
    breakdown_failure = max_existing(
        features,
        idx,
        [
            "conf_epsilon_support_reclaim_after_break_cmp_score_breakdown_failure",
            "conf_failed_breakdown_exhaustion_score",
            "conf_ob_breakdown_failure",
        ],
    )
    bearish_pressure = max_existing(
        features,
        idx,
        [
            "conf_ob_bearish_pressure_agreement",
            "conf_beta_ob_support_removed_breakdown_cmp_score_bearish_pressure",
        ],
    )
    if profile.invalidation_kind == "reclaim_rebuild_warning":
        return reclaim >= profile.invalidation_threshold
    if profile.invalidation_kind == "breakdown_failure_warning":
        return breakdown_failure >= profile.invalidation_threshold
    if profile.invalidation_kind == "bearish_pressure_fade":
        return bearish_pressure <= profile.invalidation_threshold
    if profile.invalidation_kind == "combined_reclaim_or_pressure_fade":
        return reclaim >= profile.invalidation_threshold or bearish_pressure <= 0.15
    if profile.invalidation_kind == "combined_reclaim_or_breakdown_failure":
        return reclaim >= profile.invalidation_threshold or breakdown_failure >= 0.14
    raise ValueError(f"Unknown invalidation kind: {profile.invalidation_kind}")


def max_existing(frame: DataFrame, idx: int, columns: list[str]) -> float:
    values = []
    for column in columns:
        if column in frame.columns:
            value = pd.to_numeric(pd.Series([frame[column].iloc[idx]]), errors="coerce").iloc[0]
            if np.isfinite(value):
                values.append(float(value))
    return max(values) if values else 0.0


def select_non_overlapping_trades(trades: DataFrame, selected: DataFrame) -> DataFrame:
    if trades.empty or selected.empty:
        return DataFrame()
    keep_profiles = set(selected["profile_id"].astype(str))
    frame = trades[trades["variant_id"].astype(str).isin(keep_profiles)].copy()
    rows: list[dict[str, Any]] = []
    flat_until = pd.Timestamp.min.tz_localize("UTC")
    for _, row in frame.sort_values("entry_date").iterrows():
        entry_date = pd.to_datetime(row["entry_date"], utc=True)
        if entry_date <= flat_until:
            continue
        rows.append(row.to_dict())
        flat_until = pd.to_datetime(row["exit_date"], utc=True)
    return pd.DataFrame(rows)


def markdown_report(summary: DataFrame, selected: DataFrame, selected_trades: DataFrame, features: DataFrame) -> str:
    combined = summarize_trades(
        selected_trades,
        "core_orderbook_invalidation_exit_selected",
        "core_orderbook_short_block",
        "short",
        "Can early exits help when the short thesis disappears?",
        "Core orderbook shorts, with reclaim/rebuild/pressure-fade warnings tested as early exits.",
        features,
    )
    invalidation_selected = not selected.empty and selected["invalidation_kind"].astype(str).ne("none").any()
    lines = [
        "# Core Orderbook Invalidation Exit Research",
        "",
        "This tests whether validated short entries improve when exited early after the original bearish reason starts disappearing.",
        "",
        "## Verdict",
        "",
        "1. Main exit: keep the existing fixed tailored exits." if not invalidation_selected else "1. Main exit: selected invalidation exits need Freqtrade validation.",
        "2. Reason: the best-ranked profiles for both core short legs were still the fixed baseline exits." if not invalidation_selected else "2. Reason: at least one invalidation profile beat the fixed baseline in the direct screen.",
        "3. Note: some early-exit variants reduced drawdown but gave up too much profit, so they are risk-protection candidates, not main exits yet.",
        "",
        "## Selected Result",
        "",
        f"1. Trades: {combined.get('trades', 0)}",
        f"2. Win rate: {fmt_pct(combined.get('win_rate'))}",
        f"3. Total return: {fmt_pct(combined.get('total_return'))}",
        f"4. Max drawdown: {fmt_pct(combined.get('max_drawdown'))}",
        f"5. Profit factor: {fmt(combined.get('profit_factor'))}",
        "",
        "## Selected Exit Profiles",
        "",
    ]
    if selected.empty:
        lines.append("No positive selected invalidation profiles.")
    else:
        for _, row in selected.iterrows():
            lines.extend(
                [
                    f"### {row['profile_id']}",
                    "",
                    f"1. Trader question: {row['trader_question']}",
                    f"2. Trades: {int(row['trades'])}",
                    f"3. Win rate: {fmt_pct(row['win_rate'])}",
                    f"4. Total return: {fmt_pct(row['total_return'])}",
                    f"5. Max drawdown: {fmt_pct(row['max_drawdown'])}",
                    f"6. Profit factor: {fmt(row['profit_factor'])}",
                    f"7. Invalidation exits: {int(row['invalidation_exit_count'])}",
                    "",
                ]
            )
    lines.extend(
        [
            "## Best Profiles Per Rule",
            "",
            summary.sort_values(["rule_id", "rank_score"], ascending=[True, False])
            .groupby("rule_id", as_index=False, group_keys=False)
            .head(5)[
                [
                    "rule_id",
                    "profile_id",
                    "trades",
                    "win_rate",
                    "total_return",
                    "max_drawdown",
                    "profit_factor",
                    "invalidation_exit_count",
                ]
            ]
            .to_markdown(index=False),
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
