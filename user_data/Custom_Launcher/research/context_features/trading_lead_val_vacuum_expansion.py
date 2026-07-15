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

from trader_rule_backtests import fmt_pct, num, resolve_exit, safe_name, summarize_trades


USER_DATA_DIR = Path(__file__).resolve().parents[3]
REPORTS_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "reports"
DEFAULT_FEATURES = USER_DATA_DIR / "research_news_data" / "context_features" / "confluence_cache" / "trader_confluence_1h_latest.parquet"
DEFAULT_SIGNALS = REPORTS_DIR / "trading_lead_signals_orderbook_support_removal_short_refinement_20260605_selected.parquet"
RULE_ID = "val_break_downside_vacuum_after_support_removed_short"


@dataclass(frozen=True)
class ExitSpec:
    exit_id: str
    hold_hours: int
    stop_loss: float
    take_profit: float


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    trader_question: str
    visible_market_state: str
    reclaim_max: float | None = None
    rebuild_max: float | None = None
    failure_max: float | None = None
    vacuum_min: float | None = None
    pressure_min: float | None = None
    panic_min: float | None = None


EXIT_SPECS = (
    ExitSpec("baseline_6h_sl014_tp025", 6, 0.014, 0.025),
    ExitSpec("tuned_8h_sl010_tp035", 8, 0.010, 0.035),
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Expand and validate no-reclaim VAL/vacuum quick-short variants.")
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--signals", type=Path, default=DEFAULT_SIGNALS)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--tag", default=datetime.now(timezone.utc).strftime("%Y%m%d_val_vacuum_expansion"))
    parser.add_argument("--round-trip-fee", type=float, default=0.0010)
    parser.add_argument("--slippage", type=float, default=0.0002)
    parser.add_argument("--min-trades", type=int, default=8)
    args = parser.parse_args()

    if not args.features.exists():
        raise FileNotFoundError(args.features)
    if not args.signals.exists():
        raise FileNotFoundError(args.signals)

    features = load_features(args.features)
    signals = load_signals(args.signals)
    frame = signals.merge(features, on="date", how="left", suffixes=("", "_feature"))
    variants = build_variants()
    summary, trades = run_variants(features, frame, variants, args.round_trip_fee, args.slippage)
    selected = select_variants(summary, int(args.min_trades))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = safe_name(args.tag)
    summary_path = args.output_dir / f"trading_lead_val_vacuum_expansion_{tag}_summary.csv"
    trades_path = args.output_dir / f"trading_lead_val_vacuum_expansion_{tag}_trades.csv"
    selected_path = args.output_dir / f"trading_lead_val_vacuum_expansion_{tag}_selected.csv"
    markdown_path = args.output_dir / f"trading_lead_val_vacuum_expansion_{tag}.md"
    meta_path = args.output_dir / f"trading_lead_val_vacuum_expansion_{tag}_meta.json"
    summary.to_csv(summary_path, index=False)
    trades.to_csv(trades_path, index=False)
    selected.to_csv(selected_path, index=False)

    signal_paths: dict[str, str] = {}
    for _, row in selected.iterrows():
        variant_id = str(row["variant_id"])
        subset = frame[variant_mask(frame, variant_by_id(variants, variant_id))].copy()
        signal_path = args.output_dir / f"trading_lead_signals_{tag}_{safe_name(variant_id)}.parquet"
        export_signal_rows(signals_from_subset(subset), signal_path)
        signal_paths[variant_id] = str(signal_path)

    markdown_path.write_text(markdown_report(summary, selected, signal_paths), encoding="utf-8")
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "features": str(args.features),
        "signals": str(args.signals),
        "input_rule_signals": int(len(signals)),
        "variant_count": int(len(variants)),
        "summary_rows": int(len(summary)),
        "selected_rows": int(len(selected)),
        "signal_exports": signal_paths,
        "outputs": {
            "summary": str(summary_path),
            "trades": str(trades_path),
            "selected": str(selected_path),
            "markdown": str(markdown_path),
        },
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=2))
    return 0


def load_features(path: Path) -> DataFrame:
    frame = pd.read_parquet(path)
    frame["date"] = pd.to_datetime(frame["date"], utc=True, errors="coerce")
    return frame.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def load_signals(path: Path) -> DataFrame:
    signals = pd.read_parquet(path)
    signals["date"] = pd.to_datetime(signals["date"], utc=True, errors="coerce")
    signals = signals.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="first").reset_index(drop=True)
    signals = signals[signals["rule_id"].astype(str).eq(RULE_ID)].copy()
    signals["direction"] = "short"
    return signals


def build_variants() -> tuple[VariantSpec, ...]:
    variants: list[VariantSpec] = [
        VariantSpec(
            "base_val_vacuum",
            "Does the unfiltered VAL/vacuum quick short work?",
            "Price broke below value after support removal and downside liquidity looked thin.",
        )
    ]
    for threshold in (0.22, 0.26, 0.30, 0.34, 0.38, 0.42, 0.46, 0.50):
        variants.append(
            VariantSpec(
                f"no_reclaim_lte_{int(threshold * 100):02d}",
                "Does quality improve when reclaim-warning pressure stays below this threshold?",
                "VAL/vacuum short with no strong broken-support reclaim warning.",
                reclaim_max=threshold,
            )
        )
    for threshold in (0.08, 0.12, 0.16, 0.20, 0.25):
        variants.append(
            VariantSpec(
                f"no_rebuild_lte_{int(threshold * 100):02d}",
                "Does quality improve when support does not rebuild underneath price?",
                "VAL/vacuum short with limited support rebuild warning.",
                rebuild_max=threshold,
            )
        )
    for threshold in (0.05, 0.10, 0.15, 0.20, 0.25):
        variants.append(
            VariantSpec(
                f"breakdown_failure_lte_{int(threshold * 100):02d}",
                "Does quality improve when breakdown-failure warning stays low?",
                "VAL/vacuum short without a strong failed-breakdown warning.",
                failure_max=threshold,
            )
        )
    for threshold in (0.20, 0.23, 0.26, 0.30):
        variants.append(
            VariantSpec(
                f"vacuum_gte_{int(threshold * 100):02d}",
                "Does quality improve when downside vacuum is still open?",
                "VAL/vacuum short with thin liquidity below price.",
                vacuum_min=threshold,
            )
        )
    for threshold in (0.10, 0.20, 0.33, 0.50):
        variants.append(
            VariantSpec(
                f"pressure_gte_{int(threshold * 100):02d}",
                "Does quality improve when bearish pressure remains present?",
                "VAL/vacuum short with continuing sell pressure.",
                pressure_min=threshold,
            )
        )
    for threshold in (0.22, 0.26, 0.30, 0.34):
        variants.append(
            VariantSpec(
                f"panic_gte_{int(threshold * 100):02d}",
                "Does quality improve when the orderbook looks like downside acceleration after the break?",
                "VAL/vacuum short with downside panic or acceleration state.",
                panic_min=threshold,
            )
        )
    for reclaim in (0.34, 0.38, 0.42, 0.46):
        for rebuild in (0.12, 0.16, 0.20, 0.25):
            variants.append(
                VariantSpec(
                    f"no_reclaim_{int(reclaim * 100):02d}_no_rebuild_{int(rebuild * 100):02d}",
                    "Does quality improve when reclaim and support-rebuild warnings are both limited?",
                    "VAL/vacuum short where broken support is not reclaiming or rebuilding.",
                    reclaim_max=reclaim,
                    rebuild_max=rebuild,
                )
            )
    for reclaim in (0.34, 0.38, 0.42, 0.46):
        for pressure in (0.20, 0.33, 0.50):
            variants.append(
                VariantSpec(
                    f"no_reclaim_{int(reclaim * 100):02d}_pressure_{int(pressure * 100):02d}",
                    "Does quality improve when no-reclaim state combines with bearish pressure?",
                    "VAL/vacuum short where support is not reclaiming and sell pressure remains present.",
                    reclaim_max=reclaim,
                    pressure_min=pressure,
                )
            )
    return tuple(variants)


def run_variants(features: DataFrame, frame: DataFrame, variants: tuple[VariantSpec, ...], fee: float, slippage: float) -> tuple[DataFrame, DataFrame]:
    rows: list[dict[str, Any]] = []
    trade_frames: list[DataFrame] = []
    for variant in variants:
        subset = frame[variant_mask(frame, variant)].copy()
        for exit_spec in EXIT_SPECS:
            trades = simulate(features, subset, variant, exit_spec, fee, slippage)
            if not trades.empty:
                trade_frames.append(trades)
            summary = summarize_trades(
                trades,
                f"{variant.variant_id}__{exit_spec.exit_id}",
                "val_break_orderbook_vacuum_short",
                "short",
                variant.trader_question,
                variant.visible_market_state,
                features,
            )
            summary.update(
                {
                    "variant_id": variant.variant_id,
                    "exit_id": exit_spec.exit_id,
                    "hold_hours": exit_spec.hold_hours,
                    "stop_loss": exit_spec.stop_loss,
                    "take_profit": exit_spec.take_profit,
                    "signal_rows": int(len(subset)),
                    "reclaim_max": variant.reclaim_max,
                    "rebuild_max": variant.rebuild_max,
                    "failure_max": variant.failure_max,
                    "vacuum_min": variant.vacuum_min,
                    "pressure_min": variant.pressure_min,
                    "panic_min": variant.panic_min,
                }
            )
            summary.update(stability_summary(trades))
            rows.append(summary)
    summary_frame = pd.DataFrame(rows)
    trades_frame = pd.concat(trade_frames, ignore_index=True) if trade_frames else DataFrame()
    summary_frame = add_comparisons(summary_frame)
    return summary_frame.sort_values(["verdict_rank", "quality_score", "total_return"], ascending=[False, False, False]), trades_frame


def variant_mask(frame: DataFrame, variant: VariantSpec) -> Series:
    mask = pd.Series(True, index=frame.index)
    if variant.reclaim_max is not None:
        mask &= reclaim_warning(frame).le(float(variant.reclaim_max))
    if variant.rebuild_max is not None:
        mask &= support_rebuild_warning(frame).le(float(variant.rebuild_max))
    if variant.failure_max is not None:
        mask &= breakdown_failure_warning(frame).le(float(variant.failure_max))
    if variant.vacuum_min is not None:
        mask &= downside_vacuum(frame).ge(float(variant.vacuum_min))
    if variant.pressure_min is not None:
        mask &= bearish_pressure(frame).ge(float(variant.pressure_min))
    if variant.panic_min is not None:
        mask &= panic_state(frame).ge(float(variant.panic_min))
    return mask.fillna(False)


def simulate(features: DataFrame, signals: DataFrame, variant: VariantSpec, exit_spec: ExitSpec, fee: float, slippage: float) -> DataFrame:
    by_date = {date: idx for idx, date in enumerate(features["date"])}
    open_ = num(features, "open")
    high = num(features, "high")
    low = num(features, "low")
    close = num(features, "close")
    rows: list[dict[str, Any]] = []
    flat_after = -1
    for _, signal in signals.sort_values("date").iterrows():
        idx = by_date.get(signal["date"])
        if idx is None or idx <= flat_after or idx > len(features) - exit_spec.hold_hours - 2:
            continue
        entry_idx = idx + 1
        exit_idx = idx + exit_spec.hold_hours
        entry_price = float(open_.iloc[entry_idx])
        if not np.isfinite(entry_price) or entry_price <= 0.0:
            continue
        exit_price, exit_reason = resolve_exit(
            "short",
            entry_price,
            high.iloc[entry_idx : exit_idx + 1],
            low.iloc[entry_idx : exit_idx + 1],
            close.iloc[exit_idx],
            exit_spec.stop_loss,
            exit_spec.take_profit,
        )
        gross = entry_price / exit_price - 1.0
        rows.append(
            {
                "variant_id": variant.variant_id,
                "exit_id": exit_spec.exit_id,
                "rule_id": RULE_ID,
                "concept_id": str(signal.get("concept_id", "")),
                "direction": "short",
                "signal_date": features["date"].iloc[idx],
                "entry_date": features["date"].iloc[entry_idx],
                "exit_date": features["date"].iloc[exit_idx],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross,
                "net_return": gross - fee - slippage,
                "exit_reason": exit_reason,
                "signal_score": signal.get("signal_score", np.nan),
            }
        )
        flat_after = exit_idx
    return pd.DataFrame(rows)


def add_comparisons(summary: DataFrame) -> DataFrame:
    base = summary[summary["variant_id"].eq("base_val_vacuum")].copy()
    if base.empty:
        summary["base_trades"] = 0
        summary["base_win_rate"] = 0.0
        summary["base_total_return"] = 0.0
        summary["base_profit_factor"] = 0.0
        summary["base_avg_trade_return"] = 0.0
    else:
        base_by_exit = base.set_index("exit_id")
        summary["base_trades"] = summary["exit_id"].map(base_by_exit["trades"]).fillna(0).astype(int)
        summary["base_win_rate"] = summary["exit_id"].map(base_by_exit["win_rate"]).fillna(0.0)
        summary["base_total_return"] = summary["exit_id"].map(base_by_exit["total_return"]).fillna(0.0)
        summary["base_profit_factor"] = summary["exit_id"].map(base_by_exit["profit_factor"]).fillna(0.0)
        summary["base_avg_trade_return"] = summary["exit_id"].map(base_by_exit["avg_trade_return"]).fillna(0.0)
    summary["quality_delta"] = (
        (summary["profit_factor"] - summary["base_profit_factor"])
        + ((summary["win_rate"] - summary["base_win_rate"]) * 1.5)
        + ((summary["avg_trade_return"] - summary["base_avg_trade_return"]) * 30.0)
    )
    summary["keeps_enough_rows"] = summary["trades"].ge(8)
    summary["beats_base"] = (
        summary["keeps_enough_rows"]
        & summary["total_return"].gt(summary["base_total_return"])
        & summary["profit_factor"].gt(summary["base_profit_factor"])
        & summary["max_drawdown"].ge(summary["max_drawdown"].quantile(0.25))
    )
    summary["verdict"] = np.where(summary["beats_base"], "worth_more_testing", np.where(summary["keeps_enough_rows"], "watchlist", "too_few_trades"))
    summary.loc[summary["variant_id"].eq("base_val_vacuum"), "verdict"] = "baseline"
    rank_map = {"worth_more_testing": 3, "watchlist": 2, "baseline": 1, "too_few_trades": 0}
    summary["verdict_rank"] = summary["verdict"].map(rank_map).fillna(0).astype(int)
    summary["quality_score"] = (
        summary["total_return"].fillna(0.0)
        + summary["profit_factor"].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(0.0, 10.0) / 10.0
        + summary["win_rate"].fillna(0.0) / 4.0
        + summary["max_drawdown"].fillna(-1.0)
    )
    return summary


def select_variants(summary: DataFrame, min_trades: int) -> DataFrame:
    candidates = summary[
        summary["trades"].ge(min_trades)
        & summary["variant_id"].ne("base_val_vacuum")
        & summary["total_return"].gt(summary["base_total_return"])
        & summary["profit_factor"].gt(summary["base_profit_factor"])
    ].copy()
    if candidates.empty:
        return candidates
    quality = (
        candidates.sort_values(["quality_score", "trades", "total_return"], ascending=[False, False, False])
        .groupby("exit_id", as_index=False, group_keys=False)
        .head(3)
    )
    expanded = (
        candidates.sort_values(["trades", "quality_score", "total_return"], ascending=[False, False, False])
        .groupby("exit_id", as_index=False, group_keys=False)
        .head(3)
    )
    named_review = candidates[candidates["variant_id"].isin(["no_reclaim_lte_46"])].copy()
    return (
        pd.concat([quality, expanded, named_review], ignore_index=True)
        .drop_duplicates(["variant_id", "exit_id"], keep="first")
        .sort_values(["quality_score", "trades", "total_return"], ascending=[False, False, False])
        .reset_index(drop=True)
    )


def signals_from_subset(subset: DataFrame) -> DataFrame:
    cols = ["date", "enter_long", "enter_short", "enter_tag", "rule_id", "concept_id", "threshold", "signal_score"]
    available = [col for col in cols if col in subset.columns]
    signals = subset[available].copy()
    if "date" in signals.columns:
        signals["date"] = pd.to_datetime(signals["date"], utc=True, errors="coerce")
    if "enter_long" not in signals.columns:
        signals["enter_long"] = 0
    if "enter_short" not in signals.columns:
        signals["enter_short"] = 1
    if "enter_tag" not in signals.columns and "rule_id" in signals.columns:
        signals["enter_tag"] = signals["rule_id"].astype(str)
    return signals


def export_signal_rows(signals: DataFrame, output_path: Path) -> None:
    columns = ["date", "enter_long", "enter_short", "enter_tag", "rule_id", "concept_id", "threshold", "signal_score"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if signals.empty:
        pd.DataFrame(columns=columns).to_parquet(output_path, index=False)
        return
    for col in columns:
        if col not in signals.columns:
            signals[col] = np.nan
    signals = signals[columns].dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="first")
    signals.to_parquet(output_path, index=False)


def stability_summary(trades: DataFrame) -> dict[str, Any]:
    if trades.empty:
        return {
            "active_years": 0,
            "positive_years": 0,
            "positive_year_rate": 0.0,
            "worst_year_return": 0.0,
            "best_year_return": 0.0,
            "active_months": 0,
            "positive_months": 0,
            "positive_month_rate": 0.0,
        }
    trades = trades.copy()
    trades["entry_date"] = pd.to_datetime(trades["entry_date"], utc=True, errors="coerce")
    trades = trades.dropna(subset=["entry_date"])
    if trades.empty:
        return {
            "active_years": 0,
            "positive_years": 0,
            "positive_year_rate": 0.0,
            "worst_year_return": 0.0,
            "best_year_return": 0.0,
            "active_months": 0,
            "positive_months": 0,
            "positive_month_rate": 0.0,
        }
    year_returns = trades.groupby(trades["entry_date"].dt.year)["net_return"].sum()
    month_returns = trades.groupby(trades["entry_date"].dt.strftime("%Y-%m"))["net_return"].sum()
    return {
        "active_years": int(len(year_returns)),
        "positive_years": int(year_returns.gt(0.0).sum()),
        "positive_year_rate": float(year_returns.gt(0.0).mean()) if len(year_returns) else 0.0,
        "worst_year_return": float(year_returns.min()) if len(year_returns) else 0.0,
        "best_year_return": float(year_returns.max()) if len(year_returns) else 0.0,
        "active_months": int(len(month_returns)),
        "positive_months": int(month_returns.gt(0.0).sum()),
        "positive_month_rate": float(month_returns.gt(0.0).mean()) if len(month_returns) else 0.0,
    }


def variant_by_id(variants: tuple[VariantSpec, ...], variant_id: str) -> VariantSpec:
    for variant in variants:
        if variant.variant_id == variant_id:
            return variant
    raise KeyError(variant_id)


def reclaim_warning(frame: DataFrame) -> Series:
    return max_existing(
        frame,
        [
            "conf_epsilon_support_reclaim_after_break_score",
            "conf_epsilon_support_reclaim_after_break_cmp_score_breakdown_failure",
            "conf_epsilon_downside_first_break_exhaustion_cmp_score_reclaim_pressure",
            "conf_failed_breakdown_exhaustion_score",
        ],
    )


def support_rebuild_warning(frame: DataFrame) -> Series:
    return max_existing(
        frame,
        [
            "conf_epsilon_support_reclaim_after_break_cmp_score_support_rebuild",
            "conf_epsilon_downside_first_break_exhaustion_cmp_score_support_rebuild",
        ],
    )


def breakdown_failure_warning(frame: DataFrame) -> Series:
    return max_existing(
        frame,
        [
            "conf_ob_breakdown_failure",
            "conf_failed_breakdown_exhaustion_score",
            "conf_epsilon_support_reclaim_after_break_cmp_score_breakdown_failure",
            "ob_spot_obts_breakdown_failure_score",
            "ob_linear_obts_breakdown_failure_score",
            "ob_inverse_obts_breakdown_failure_score",
        ],
    )


def downside_vacuum(frame: DataFrame) -> Series:
    return max_existing(
        frame,
        [
            "conf_ob_downside_vacuum_after_support_removed",
            "conf_epsilon_orderbook_panic_after_break_cmp_score_downside_vacuum",
            "conf_beta_ob_support_removed_breakdown_cmp_score_downside_vacuum",
            "conf_delta_orderbook_support_weak_drawdown_cmp_score_downside_vacuum",
        ],
    )


def bearish_pressure(frame: DataFrame) -> Series:
    return pd.concat(
        [
            max_existing(
                frame,
                [
                    "conf_ob_bearish_pressure_agreement",
                    "conf_beta_ob_support_removed_breakdown_cmp_score_bearish_pressure",
                ],
            ),
            -num(frame, "px_volume_pressure_6h"),
            -num(frame, "px_volume_pressure_24h"),
            -num(frame, "st_volume_pressure_6h"),
            -num(frame, "st_volume_pressure_24h"),
        ],
        axis=1,
    ).max(axis=1)


def panic_state(frame: DataFrame) -> Series:
    return max_existing(
        frame,
        [
            "conf_epsilon_orderbook_panic_after_break_score",
            "conf_epsilon_orderbook_panic_after_break_cmp_score_support_removed",
            "conf_epsilon_orderbook_panic_after_break_cmp_score_downside_vacuum",
            "conf_ob_spread_fragility",
        ],
    )


def max_existing(frame: DataFrame, cols: list[str]) -> Series:
    existing = [num(frame, col) for col in cols if col in frame.columns]
    if not existing:
        return pd.Series(0.0, index=frame.index)
    return pd.concat(existing, axis=1).max(axis=1).fillna(0.0)


def markdown_report(summary: DataFrame, selected: DataFrame, signal_paths: dict[str, str]) -> str:
    lines = [
        "# VAL/Vacuum No-Reclaim Expansion",
        "",
        "Trader question: can the sparse no-reclaim quick-short lead be expanded without losing the quality lift?",
        "",
        "## Selected Variants",
        "",
    ]
    if selected.empty:
        lines.append("No expanded variant beat the unfiltered baseline with enough trades.")
    else:
        display_cols = [
            "variant_id",
            "exit_id",
            "trades",
            "win_rate",
            "total_return",
            "max_drawdown",
            "profit_factor",
            "active_years",
            "positive_years",
            "positive_months",
            "active_months",
            "base_trades",
            "base_total_return",
            "base_profit_factor",
            "verdict",
        ]
        lines.append(selected[display_cols].to_markdown(index=False))
    lines.extend(["", "## Exported Signal Sets", ""])
    if not signal_paths:
        lines.append("No signal exports.")
    else:
        for variant_id, path in signal_paths.items():
            lines.append(f"1. `{variant_id}`: `{path}`")
    lines.extend(["", "## Top Rows", ""])
    top_cols = ["variant_id", "exit_id", "trades", "win_rate", "total_return", "max_drawdown", "profit_factor", "verdict"]
    lines.append(summary.head(20)[top_cols].to_markdown(index=False))
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
