from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


ROOT = Path("C:/FreqTradeStuff")
DEFAULT_BACKTEST_ZIP = ROOT / "user_data/backtest_results/backtest-result-2026-06-06_23-04-43.zip"
DEFAULT_FEATURES = (
    ROOT
    / "user_data/research_news_data/context_features/confluence_cache/trader_confluence_1h_latest.parquet"
)
DEFAULT_REPORT_DIR = ROOT / "user_data/research_news_data/context_features/reports"

FEATURES = [
    "conf_ob_bearish_pressure_agreement",
    "conf_ob_support_removed_strength",
    "conf_ob_support_cleared",
    "conf_ob_downside_vacuum_after_support_removed",
    "conf_ob_downside_vacuum",
    "conf_ob_spread_fragility",
    "conf_ob_pressure_divergence",
    "conf_ob_breakout_failure",
    "conf_ob_breakdown_failure",
    "conf_ob_ask_absorption",
    "conf_ob_bid_absorption",
    "conf_structure_breakdown_trigger_score",
    "conf_structure_bearish_state_score",
    "conf_structure_breakout_trigger_score",
    "conf_structure_bullish_state_score",
    "st_failed_breakout_structure_risk",
    "st_failed_breakdown_structure_risk",
    "st_breakout_structure_setup",
    "st_breakdown_structure_setup",
    "conf_failed_breakout_exhaustion_score",
    "conf_failed_breakdown_exhaustion_score",
    "px_close_breakdown_6h",
    "px_close_breakdown_24h",
    "px_close_breakout_6h",
    "px_close_breakout_24h",
    "conf_volume_bearish_confirmation",
    "conf_volume_bullish_confirmation",
    "px_volume_pressure_6h",
    "px_volume_pressure_24h",
    "st_1h_vp_below_value_area",
    "st_4h_vp_below_value_area",
    "st_1d_vp_below_value_area",
    "st_1h_vp_above_value_area",
    "st_4h_vp_above_value_area",
    "st_1d_vp_above_value_area",
]


def _load_trades(backtest_zip: Path) -> tuple[str, pd.DataFrame]:
    with zipfile.ZipFile(backtest_zip) as archive:
        result_name = next(name for name in archive.namelist() if name.endswith(".json") and "_config" not in name)
        payload = json.loads(archive.read(result_name))
    strategy_name, result = next(iter(payload["strategy"].items()))
    trades = pd.DataFrame(result["trades"])
    trades["open_date"] = pd.to_datetime(trades["open_date"], utc=True).dt.floor("h")
    trades["close_date"] = pd.to_datetime(trades["close_date"], utc=True).dt.floor("h")
    trades["is_loss"] = trades["profit_ratio"] < 0
    trades["direction"] = trades["is_short"].map({True: "short", False: "long"})
    return strategy_name, trades


def _load_features(feature_path: Path) -> pd.DataFrame:
    schema_cols = set(pq.ParquetFile(feature_path).schema.names)
    available = ["date"] + [column for column in FEATURES if column in schema_cols]
    features = pd.read_parquet(feature_path, columns=available)
    features["date"] = pd.to_datetime(features["date"], utc=True).dt.floor("h")
    return features


def _truthy_rate(series: pd.Series) -> float:
    clean = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return float((clean > 0.0).mean()) if len(clean) else 0.0


def _family_summary(trades: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tag, direction), group in trades.groupby(["enter_tag", "direction"], dropna=False):
        losses = group.loc[group["is_loss"]]
        wins = group.loc[~group["is_loss"]]
        rows.append(
            {
                "enter_tag": tag,
                "direction": direction,
                "trades": int(len(group)),
                "wins": int(len(wins)),
                "losses": int(len(losses)),
                "win_rate": round(float((~group["is_loss"]).mean()), 4),
                "total_profit_pct": round(float(group["profit_ratio"].sum() * 100.0), 4),
                "avg_profit_pct": round(float(group["profit_ratio"].mean() * 100.0), 4),
                "avg_loss_pct": round(float(losses["profit_ratio"].mean() * 100.0), 4) if len(losses) else 0.0,
                "worst_loss_pct": round(float(group["profit_ratio"].min() * 100.0), 4),
                "avg_duration_hours": round(float(group["trade_duration"].mean() / 60.0), 2),
            }
        )
    return pd.DataFrame(rows).sort_values(["losses", "total_profit_pct"], ascending=[False, True])


def _signal_diagnostics(trades_with_features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    suffixes = {"entry": "_entry", "exit": "_exit"}
    for (tag, direction), group in trades_with_features.groupby(["enter_tag", "direction"], dropna=False):
        if len(group) < 3:
            continue
        losses = group.loc[group["is_loss"]]
        wins = group.loc[~group["is_loss"]]
        if len(losses) == 0 or len(wins) == 0:
            continue
        for phase, suffix in suffixes.items():
            for feature in FEATURES:
                column = f"{feature}{suffix}"
                if column not in group:
                    continue
                loss_rate = _truthy_rate(losses[column])
                win_rate = _truthy_rate(wins[column])
                loss_mean = float(pd.to_numeric(losses[column], errors="coerce").fillna(0.0).mean())
                win_mean = float(pd.to_numeric(wins[column], errors="coerce").fillna(0.0).mean())
                rows.append(
                    {
                        "enter_tag": tag,
                        "direction": direction,
                        "phase": phase,
                        "feature": feature,
                        "trades": int(len(group)),
                        "losses": int(len(losses)),
                        "wins": int(len(wins)),
                        "loss_active_rate": round(loss_rate, 4),
                        "win_active_rate": round(win_rate, 4),
                        "active_rate_gap_loss_minus_win": round(loss_rate - win_rate, 4),
                        "loss_mean": round(loss_mean, 6),
                        "win_mean": round(win_mean, 6),
                        "mean_gap_loss_minus_win": round(loss_mean - win_mean, 6),
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["active_rate_gap_loss_minus_win", "mean_gap_loss_minus_win", "losses"],
        ascending=[False, False, False],
    )


def _write_markdown(
    path: Path,
    strategy_name: str,
    family_summary: pd.DataFrame,
    signal_summary: pd.DataFrame,
    backtest_zip: Path,
    feature_path: Path,
) -> None:
    lines = [
        "# Current Best Family Failure Diagnostics",
        "",
        f"Generated: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`",
        "",
        "## Scope",
        "",
        f"1. Strategy: `{strategy_name}`",
        f"2. Backtest archive: `{backtest_zip}`",
        f"3. Feature cache: `{feature_path}`",
        "4. Purpose: identify entry families where failure states differ between winners and losers.",
        "",
        "## Plain-English Reading",
        "",
        "1. Entry-phase rows ask: what was visible when the trade opened?",
        "2. Exit-phase rows ask: what was visible near the eventual close hour?",
        "3. A useful exit candidate is not just common in losers; it should be less common in winners from the same entry family.",
        "4. This report is diagnostic evidence only. It should guide family-specific exit tests, not become a shared broad exit rule.",
        "",
        "## Families With Most Losses",
        "",
        "| Entry family | Side | Trades | Losses | Win rate | Total profit % | Worst loss % | Suggested next action |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in family_summary.head(12).itertuples(index=False):
        action = "family-specific exit diagnostic"
        if row.losses <= 1:
            action = "park; too few losses"
        elif row.total_profit_pct < 0:
            action = "consider removal or strict filter"
        lines.append(
            f"| `{row.enter_tag}` | {row.direction} | {row.trades} | {row.losses} | {row.win_rate:.2%} | "
            f"{row.total_profit_pct:.2f} | {row.worst_loss_pct:.2f} | {action} |"
        )
    lines.extend(
        [
            "",
            "## Strongest Loser-Skewed Failure States",
            "",
            "| Entry family | Phase | Feature | Loss active | Winner active | Gap | Meaning |",
            "| --- | --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    plain = {
        "conf_ob_bearish_pressure_agreement": "orderbook leaned bearish",
        "conf_ob_support_removed_strength": "support liquidity was being removed",
        "conf_ob_support_cleared": "support looked cleared",
        "conf_ob_downside_vacuum_after_support_removed": "support was removed and downside looked thin",
        "conf_ob_spread_fragility": "book looked fragile/wider spread",
        "conf_structure_breakdown_trigger_score": "price structure was breaking down",
        "conf_structure_bearish_state_score": "chart state looked bearish",
        "st_failed_breakout_structure_risk": "breakout/follow-through looked like it was failing",
        "conf_failed_breakout_exhaustion_score": "breakout exhaustion/failure pattern was visible",
        "conf_volume_bearish_confirmation": "volume was confirming downside pressure",
        "px_close_breakdown_6h": "recent close broke below its local range",
    }
    top = signal_summary.loc[
        (signal_summary["phase"] == "exit")
        & (signal_summary["trades"] >= 8)
        & (signal_summary["losses"] >= 3)
        & (signal_summary["active_rate_gap_loss_minus_win"] >= 0.25)
    ].head(20)
    for row in top.itertuples(index=False):
        meaning = plain.get(row.feature, row.feature.replace("_", " "))
        lines.append(
            f"| `{row.enter_tag}` ({row.direction}) | {row.phase} | `{row.feature}` | "
            f"{row.loss_active_rate:.2%} | {row.win_active_rate:.2%} | "
            f"{row.active_rate_gap_loss_minus_win:.2%} | {meaning} |"
        )
    if top.empty:
        lines.append("| _None_ |  |  |  |  |  | No strong loser-skewed exit state found with current thresholds. |")
    focus_families = family_summary.loc[family_summary["losses"] >= 5, ["enter_tag", "direction"]].head(8)
    lines.extend(
        [
            "",
            "## Top Loss Families: Best Exit-State Clues",
            "",
            "| Entry family | Feature | Loss active | Winner active | Gap | Practical reading |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )
    for family_row in focus_families.itertuples(index=False):
        family = family_row.enter_tag
        direction = family_row.direction
        family_rows = signal_summary.loc[
            (signal_summary["enter_tag"] == family)
            & (signal_summary["direction"] == direction)
            & (signal_summary["phase"] == "exit")
            & (signal_summary["active_rate_gap_loss_minus_win"] > 0.0)
        ].head(3)
        if family_rows.empty:
            lines.append(
                f"| `{family}` ({direction}) | _None_ |  |  |  | No clean loser-skewed state found in selected features. |"
            )
            continue
        for row in family_rows.itertuples(index=False):
            meaning = plain.get(row.feature, row.feature.replace("_", " "))
            lines.append(
                f"| `{row.enter_tag}` ({row.direction}) | `{row.feature}` | {row.loss_active_rate:.2%} | "
                f"{row.win_active_rate:.2%} | {row.active_rate_gap_loss_minus_win:.2%} | {meaning} |"
            )
    lines.extend(
        [
            "",
            "## Next Tests Suggested",
            "",
            "1. Test family-specific exits only for rows above, starting with exit-phase signals that are common in losers and uncommon in winners.",
            "2. Do not promote shared long orderbook-stress exits from this report; previous shared tests removed too much edge.",
            "3. For profitable families with many losses, prefer a targeted early-exit rule over removal.",
            "4. For net-negative families, test removal or stricter entry filters before adding complicated exits.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backtest-zip", type=Path, default=DEFAULT_BACKTEST_ZIP)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--stem", default="production_alpha_current_best_family_diagnostics_20260607")
    args = parser.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)
    strategy_name, trades = _load_trades(args.backtest_zip)
    features = _load_features(args.features)

    feature_cols = [column for column in features.columns if column != "date"]
    entry_features = features.rename(columns={column: f"{column}_entry" for column in feature_cols})
    exit_features = features.rename(columns={column: f"{column}_exit" for column in feature_cols})
    merged = trades.merge(entry_features, left_on="open_date", right_on="date", how="left").drop(columns=["date"])
    merged = merged.merge(exit_features, left_on="close_date", right_on="date", how="left").drop(columns=["date"])

    family = _family_summary(merged)
    signals = _signal_diagnostics(merged)

    stem = args.stem
    family_path = args.report_dir / f"{stem}.csv"
    signal_path = args.report_dir / f"{stem}_signals.csv"
    merged_path = args.report_dir / f"{stem}_trades.csv"
    md_path = args.report_dir / f"{stem}.md"

    family.to_csv(family_path, index=False)
    signals.to_csv(signal_path, index=False)
    keep = [
        "pair",
        "open_date",
        "close_date",
        "enter_tag",
        "direction",
        "profit_ratio",
        "profit_abs",
        "trade_duration",
        "exit_reason",
    ]
    merged[keep].to_csv(merged_path, index=False)
    _write_markdown(md_path, strategy_name, family, signals, args.backtest_zip, args.features)

    print(f"wrote {family_path}")
    print(f"wrote {signal_path}")
    print(f"wrote {merged_path}")
    print(f"wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
