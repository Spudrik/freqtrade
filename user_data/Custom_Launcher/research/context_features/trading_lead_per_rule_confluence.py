from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas import DataFrame, Series


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from trading_lead_confluence_checks import build_filters, confluence_combos  # noqa: E402


USER_DATA_DIR = Path(__file__).resolve().parents[3]
REPORTS_DIR = USER_DATA_DIR / "research_news_data" / "context_features" / "reports"
DEFAULT_FEATURES = USER_DATA_DIR / "research_news_data" / "context_features" / "confluence_cache" / "trader_confluence_1h_latest.parquet"
DEFAULT_SIGNALS = REPORTS_DIR / "trader_rule_signals_sieve_exact_min10_20260605.parquet"


@dataclass(frozen=True)
class ExitProfile:
    profile_id: str
    rule_id: str
    hold_hours: int
    stop_loss: float
    take_profit: float
    description: str


BASELINE_EXITS: dict[str, tuple[int, float, float]] = {
    "sieve_pattern_rectangle_breakdown_short": (10, 0.020, 0.050),
    "sieve_pattern_rectangle_breakout_long": (10, 0.020, 0.050),
    "sieve_prior_day_high_break_vp_long": (48, 0.020, 0.020),
    "sieve_prior_day_low_break_vp_short": (48, 0.020, 0.020),
}

TAILORED_EXITS: dict[str, tuple[int, float, float]] = {
    "sieve_pattern_rectangle_breakdown_short": (10, 0.025, 0.050),
    "sieve_pattern_rectangle_breakout_long": (48, 0.035, 0.075),
    "sieve_prior_day_high_break_vp_long": (72, 0.035, 0.020),
    "sieve_prior_day_low_break_vp_short": (36, 0.035, 0.050),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run per-rule confluence screens with reusable filtered signal exports.")
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
    merged = signals.merge(frame, on="date", how="left", suffixes=("", "_feature"))
    summary, selected, exports = run_per_rule_confluence(
        frame,
        merged,
        round_trip_fee=float(args.round_trip_fee),
        slippage=float(args.slippage),
        min_trades=int(args.min_trades),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tag = safe_name(args.tag)
    summary_path = args.output_dir / f"trading_lead_per_rule_confluence_{tag}_summary.csv"
    selected_path = args.output_dir / f"trading_lead_per_rule_confluence_{tag}_selected.csv"
    markdown_path = args.output_dir / f"trading_lead_per_rule_confluence_{tag}.md"
    meta_path = args.output_dir / f"trading_lead_per_rule_confluence_{tag}_meta.json"
    summary.to_csv(summary_path, index=False)
    selected.to_csv(selected_path, index=False)

    signal_paths: dict[str, str] = {}
    for export_id, export_signals in exports.items():
        path = args.output_dir / f"trading_lead_signals_{tag}_{safe_name(export_id)}.parquet"
        write_signals(export_signals, path)
        signal_paths[export_id] = str(path)

    markdown_path.write_text(markdown_report(summary, selected, signal_paths), encoding="utf-8")
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "features": str(args.features),
        "signals": str(args.signals),
        "input_signals": int(len(signals)),
        "summary_rows": int(len(summary)),
        "selected_rows": int(len(selected)),
        "signal_exports": signal_paths,
        "outputs": {
            "summary": str(summary_path),
            "selected": str(selected_path),
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
    for col in ("open", "high", "low", "close"):
        if col not in frame.columns:
            raise KeyError(f"Missing required OHLCV column: {col}")
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
    signals["direction"] = np.where(pd.to_numeric(signals["enter_long"], errors="coerce").fillna(0).gt(0), "long", "short")
    return signals


def run_per_rule_confluence(frame: DataFrame, merged_signals: DataFrame, *, round_trip_fee: float, slippage: float, min_trades: int) -> tuple[DataFrame, DataFrame, dict[str, DataFrame]]:
    filter_masks = build_filter_masks(merged_signals)
    exit_profiles = build_exit_profiles(merged_signals)
    rows: list[dict[str, Any]] = []
    all_signal_variants: dict[tuple[str, str, str], DataFrame] = {}

    for profile in exit_profiles:
        rule_signals = merged_signals[merged_signals["rule_id"].eq(profile.rule_id)].copy()
        if rule_signals.empty:
            continue
        base_mask = pd.Series(True, index=rule_signals.index)
        base_stats = simulate_and_summarize(frame, rule_signals, base_mask, profile, round_trip_fee, slippage)
        rows.append(result_row(profile, "base_no_extra_filter", "No extra confluence filter.", "base", base_stats, base_stats, min_trades))
        all_signal_variants[(profile.profile_id, profile.rule_id, "base_no_extra_filter")] = rule_signals

        for filter_id, filter_info in filter_masks.items():
            mask = filter_info["mask"].reindex(rule_signals.index).fillna(False).astype(bool)
            stats = simulate_and_summarize(frame, rule_signals, mask, profile, round_trip_fee, slippage)
            rows.append(result_row(profile, filter_id, filter_info["question"], filter_info["family"], stats, base_stats, min_trades))
            if stats["trades"] > 0:
                all_signal_variants[(profile.profile_id, profile.rule_id, filter_id)] = rule_signals[mask]

    summary = pd.DataFrame(rows)
    summary = summary.sort_values(["verdict_rank", "quality_delta", "filtered_trades"], ascending=[False, False, False]).reset_index(drop=True)
    selected = select_filters(summary)
    exports = build_exports(merged_signals, selected, all_signal_variants)
    return summary, selected, exports


def build_filter_masks(frame: DataFrame) -> dict[str, dict[str, Any]]:
    filters = build_filters(frame)
    masks: dict[str, dict[str, Any]] = {}
    for spec in filters:
        masks[spec.filter_id] = {
            "mask": spec.mask_builder(frame).fillna(False).astype(bool),
            "question": spec.trader_question,
            "family": spec.data_family,
        }
    for combo_id, parts, question in confluence_combos():
        if all(part in masks for part in parts):
            combo_mask = pd.Series(True, index=frame.index)
            families = []
            for part in parts:
                combo_mask &= masks[part]["mask"]
                families.append(masks[part]["family"])
            masks[combo_id] = {"mask": combo_mask, "question": question, "family": "+".join(sorted(set(families)))}
    return masks


def build_exit_profiles(signals: DataFrame) -> tuple[ExitProfile, ...]:
    profiles: list[ExitProfile] = []
    for profile_id, source in (("baseline_exit", BASELINE_EXITS), ("tailored_exit", TAILORED_EXITS)):
        for rule_id, (hold, stop, target) in source.items():
            profiles.append(
                ExitProfile(
                    profile_id=profile_id,
                    rule_id=rule_id,
                    hold_hours=hold,
                    stop_loss=stop,
                    take_profit=target,
                    description=f"{profile_id}: hold {hold}h, stop {stop:.1%}, target {target:.1%}",
                )
            )
    known = {(profile.profile_id, profile.rule_id) for profile in profiles}
    extra_rules = sorted(set(signals["rule_id"].dropna().astype(str)) - {rule_id for _, rule_id in known})
    for rule_id in extra_rules:
        for profile in inferred_profiles_for_rule(rule_id):
            if (profile.profile_id, profile.rule_id) not in known:
                profiles.append(profile)
                known.add((profile.profile_id, profile.rule_id))
    return tuple(profiles)


def inferred_profiles_for_rule(rule_id: str) -> tuple[ExitProfile, ...]:
    mtf_profiles: dict[str, tuple[int, float, float, str]] = {
        "mtf_4h_tlv2_res_break_long": (
            12,
            0.020,
            0.050,
            "validated_exit: 4h TLV2 resistance break long, hold 12h, stop 2.0%, target 5.0%",
        ),
        "mtf_1d_tlv2_res_retest_long": (
            24,
            0.020,
            0.050,
            "validated_exit: daily TLV2 resistance retest long, hold 24h, stop 2.0%, target 5.0%",
        ),
        "mtf_4h_triangle_upper_break_long": (
            12,
            0.020,
            0.050,
            "validated_exit: 4h triangle upper-break long, hold 12h, stop 2.0%, target 5.0%",
        ),
        "mtf_4h_vp_val_accept_short": (
            12,
            0.020,
            0.050,
            "validated_exit: 4h VAL acceptance short, hold 12h, stop 2.0%, target 5.0%",
        ),
        "mtf_1d_rectangle_lower_break_short": (
            12,
            0.020,
            0.050,
            "validated_exit: daily rectangle lower-break short, hold 12h, stop 2.0%, target 5.0%",
        ),
        "mtf_1d_tlv2_sup_retest_short": (
            24,
            0.020,
            0.050,
            "validated_exit: daily TLV2 support retest short, hold 24h, stop 2.0%, target 5.0%",
        ),
    }
    if rule_id in mtf_profiles:
        hold, stop, target, description = mtf_profiles[rule_id]
        return (
            ExitProfile(
                profile_id="validated_exit",
                rule_id=rule_id,
                hold_hours=hold,
                stop_loss=stop,
                take_profit=target,
                description=description,
            ),
        )
    if rule_id == "support_break_support_cleared_short":
        return (
            ExitProfile(
                profile_id="validated_exit",
                rule_id=rule_id,
                hold_hours=18,
                stop_loss=0.025,
                take_profit=0.050,
                description="validated_exit: support-cleared short, hold 18h, stop 2.5%, target 5.0%",
            ),
        )
    if rule_id == "val_break_downside_vacuum_after_support_removed_short":
        return (
            ExitProfile(
                profile_id="validated_exit",
                rule_id=rule_id,
                hold_hours=6,
                stop_loss=0.014,
                take_profit=0.025,
                description="validated_exit: quick VAL/vacuum short, hold 6h, stop 1.4%, target 2.5%",
            ),
        )
    direction = infer_rule_direction(rule_id)
    profile = ExitProfile(
        profile_id="generic_research_exit",
        rule_id=rule_id,
        hold_hours=24,
        stop_loss=0.020,
        take_profit=0.030,
        description=f"generic_research_exit: inferred {direction}, hold 24h, stop 2.0%, target 3.0%",
    )
    return (profile,)


def infer_rule_direction(rule_id: str) -> str:
    text_rule = rule_id.lower()
    if "short" in text_rule or "breakdown" in text_rule or "reject" in text_rule:
        return "short"
    if "long" in text_rule or "breakout" in text_rule or "bounce" in text_rule:
        return "long"
    return "both"


def simulate_and_summarize(frame: DataFrame, signals: DataFrame, mask: Series, profile: ExitProfile, round_trip_fee: float, slippage: float) -> dict[str, float]:
    filtered = signals[mask].copy()
    trades = simulate_signals(frame, filtered, profile, round_trip_fee, slippage)
    return summarize_trades(trades)


def simulate_signals(frame: DataFrame, signals: DataFrame, profile: ExitProfile, round_trip_fee: float, slippage: float) -> DataFrame:
    frame_index = {date: idx for idx, date in enumerate(frame["date"])}
    open_ = num(frame, "open")
    high = num(frame, "high")
    low = num(frame, "low")
    close = num(frame, "close")
    rows: list[dict[str, Any]] = []
    for _, signal in signals.sort_values("date").iterrows():
        i = frame_index.get(signal["date"])
        if i is None or i > len(frame) - profile.hold_hours - 2:
            continue
        entry_idx = i + 1
        exit_idx = i + profile.hold_hours
        entry_price = float(open_.iloc[entry_idx])
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue
        direction = str(signal["direction"])
        exit_price, exit_reason = resolve_exit(
            direction,
            entry_price,
            high.iloc[entry_idx : exit_idx + 1],
            low.iloc[entry_idx : exit_idx + 1],
            close.iloc[exit_idx],
            profile.stop_loss,
            profile.take_profit,
        )
        gross_return = exit_price / entry_price - 1.0 if direction == "long" else entry_price / exit_price - 1.0
        rows.append(
            {
                "rule_id": profile.rule_id,
                "direction": direction,
                "signal_date": signal["date"],
                "entry_date": frame["date"].iloc[entry_idx],
                "exit_date": frame["date"].iloc[exit_idx],
                "net_return": gross_return - round_trip_fee - slippage,
                "exit_reason": exit_reason,
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


def summarize_trades(trades: DataFrame) -> dict[str, float]:
    if trades.empty:
        return {"trades": 0.0, "win_rate": 0.0, "avg_return": 0.0, "total_return": 0.0, "profit_factor": 0.0, "max_drawdown": 0.0}
    returns = pd.to_numeric(trades["net_return"], errors="coerce").fillna(0.0)
    equity = (1.0 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    gross_loss = abs(float(losses.sum()))
    pf = float(wins.sum()) / gross_loss if gross_loss > 0 else (999.0 if float(wins.sum()) > 0 else 0.0)
    return {
        "trades": float(len(returns)),
        "win_rate": float((returns > 0).mean()),
        "avg_return": float(returns.mean()),
        "total_return": float(equity.iloc[-1] - 1.0),
        "profit_factor": pf,
        "max_drawdown": float(drawdown.min()),
    }


def result_row(profile: ExitProfile, filter_id: str, question: str, family: str, stats: dict[str, float], base: dict[str, float], min_trades: int) -> dict[str, Any]:
    quality_delta = (
        (stats["profit_factor"] - base["profit_factor"])
        + ((stats["win_rate"] - base["win_rate"]) * 1.5)
        + ((stats["avg_return"] - base["avg_return"]) * 40.0)
        + ((stats["max_drawdown"] - base["max_drawdown"]) * 2.0)
    )
    enough = stats["trades"] >= min_trades
    if filter_id == "base_no_extra_filter":
        verdict = "baseline"
        rank = 1
    elif enough and quality_delta > 0.15 and stats["profit_factor"] > base["profit_factor"] and stats["avg_return"] >= base["avg_return"]:
        verdict = "worth_more_testing"
        rank = 4
    elif enough and quality_delta > 0.0:
        verdict = "watchlist"
        rank = 3
    elif stats["trades"] > 0 and not enough:
        verdict = "too_few_trades"
        rank = 2
    else:
        verdict = "reject_for_now"
        rank = 0
    return {
        "exit_profile": profile.profile_id,
        "rule_id": profile.rule_id,
        "filter_id": filter_id,
        "trader_question": question,
        "data_family": family,
        "hold_hours": profile.hold_hours,
        "stop_loss": profile.stop_loss,
        "take_profit": profile.take_profit,
        "filtered_trades": int(stats["trades"]),
        "base_trades": int(base["trades"]),
        "win_rate": round(stats["win_rate"], 4),
        "base_win_rate": round(base["win_rate"], 4),
        "avg_return": round(stats["avg_return"], 5),
        "base_avg_return": round(base["avg_return"], 5),
        "total_return": round(stats["total_return"], 5),
        "base_total_return": round(base["total_return"], 5),
        "profit_factor": round(stats["profit_factor"], 4),
        "base_profit_factor": round(base["profit_factor"], 4),
        "max_drawdown": round(stats["max_drawdown"], 5),
        "base_max_drawdown": round(base["max_drawdown"], 5),
        "quality_delta": round(float(quality_delta), 4),
        "verdict": verdict,
        "verdict_rank": rank,
        "plain_english_result": plain_result(profile, filter_id, stats, base, verdict),
    }


def plain_result(profile: ExitProfile, filter_id: str, stats: dict[str, float], base: dict[str, float], verdict: str) -> str:
    if filter_id == "base_no_extra_filter":
        return "This is the unfiltered version of the entry family."
    if stats["trades"] <= 0:
        return "The filter removed every decision point, so it is not useful."
    direction = "improved" if verdict in {"worth_more_testing", "watchlist"} else "did not improve"
    return (
        f"This filter kept {int(stats['trades'])} of {int(base['trades'])} decision points and {direction} the setup: "
        f"win rate {stats['win_rate']:.1%} vs {base['win_rate']:.1%}, "
        f"average return {stats['avg_return']:.2%} vs {base['avg_return']:.2%}, "
        f"profit factor {stats['profit_factor']:.2f} vs {base['profit_factor']:.2f}."
    )


def select_filters(summary: DataFrame) -> DataFrame:
    candidates = summary[summary["verdict"].isin(["worth_more_testing", "watchlist"])].copy()
    if candidates.empty:
        return candidates
    candidates = candidates.sort_values(["verdict_rank", "quality_delta", "filtered_trades"], ascending=[False, False, False])
    return candidates.groupby(["exit_profile", "rule_id"], as_index=False, group_keys=False).head(2).reset_index(drop=True)


def build_exports(merged_signals: DataFrame, selected: DataFrame, variants: dict[tuple[str, str, str], DataFrame]) -> dict[str, DataFrame]:
    exports: dict[str, DataFrame] = {}
    if selected.empty:
        return exports
    for profile_id, profile_rows in selected.groupby("exit_profile"):
        combined_parts: list[DataFrame] = []
        for _, row in profile_rows.iterrows():
            key = (str(row["exit_profile"]), str(row["rule_id"]), str(row["filter_id"]))
            part = variants.get(key)
            if part is not None and not part.empty:
                combined_parts.append(part)
        if combined_parts:
            exports[f"{profile_id}_selected_confluence"] = pd.concat(combined_parts, ignore_index=True)
    best_rows = selected.sort_values(["verdict_rank", "quality_delta", "filtered_trades"], ascending=[False, False, False]).groupby("rule_id", as_index=False, group_keys=False).head(1)
    best_parts: list[DataFrame] = []
    for _, row in best_rows.iterrows():
        key = (str(row["exit_profile"]), str(row["rule_id"]), str(row["filter_id"]))
        part = variants.get(key)
        if part is not None and not part.empty:
            best_parts.append(part)
    if best_parts:
        exports["best_per_rule_selected_confluence"] = pd.concat(best_parts, ignore_index=True)
    return exports


def write_signals(signals: DataFrame, path: Path) -> None:
    columns = ["date", "enter_long", "enter_short", "enter_tag", "rule_id", "concept_id", "threshold", "signal_score"]
    if signals.empty:
        pd.DataFrame(columns=columns).to_parquet(path, index=False)
        return
    out = signals.copy()
    for column in columns:
        if column not in out.columns:
            out[column] = 0 if column.startswith("enter_") else ""
    out = out[columns].sort_values("date").drop_duplicates("date", keep="first").reset_index(drop=True)
    out.to_parquet(path, index=False)


def markdown_report(summary: DataFrame, selected: DataFrame, signal_paths: dict[str, str]) -> str:
    lines = [
        "# Per-Rule Confluence Report",
        "",
        "This tests confluence filters separately for each selected entry family. It avoids the earlier mistake of applying one global filter to every setup.",
        "",
        "## Selected Filters",
        "",
    ]
    if selected.empty:
        lines.append("No per-rule filters passed the watchlist gate.")
    else:
        display = selected[
            [
                "exit_profile",
                "rule_id",
                "filter_id",
                "filtered_trades",
                "base_trades",
                "win_rate",
                "base_win_rate",
                "avg_return",
                "base_avg_return",
                "profit_factor",
                "base_profit_factor",
                "quality_delta",
                "verdict",
                "plain_english_result",
            ]
        ]
        lines.append(display.to_markdown(index=False))
    lines.extend(["", "## Exported Signal Sets", ""])
    if signal_paths:
        for export_id, path in signal_paths.items():
            lines.append(f"1. `{export_id}`: `{path}`")
    else:
        lines.append("No filtered signal sets exported.")
    lines.extend(["", "## Top Results", ""])
    top = summary[summary["filter_id"].ne("base_no_extra_filter")].sort_values(["verdict_rank", "quality_delta"], ascending=[False, False]).head(40)
    if not top.empty:
        lines.append(top[["exit_profile", "rule_id", "filter_id", "filtered_trades", "win_rate", "avg_return", "profit_factor", "quality_delta", "verdict", "plain_english_result"]].to_markdown(index=False))
    return "\n".join(lines) + "\n"


def num(frame: DataFrame, column: str) -> Series:
    return pd.to_numeric(frame[column], errors="coerce")


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in str(value)).strip("_")[:140] or "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
